import argparse
import json
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


BOARD_SIZE = 6
EMPTY = 0
BLACK = 1
WHITE = -1
DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


class Reversi6x6:
    def __init__(self) -> None:
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        c = BOARD_SIZE // 2
        self.board[c - 1, c - 1] = WHITE
        self.board[c - 1, c] = BLACK
        self.board[c, c - 1] = BLACK
        self.board[c, c] = WHITE
        self.current_player = BLACK
        self.pass_count = 0

    def copy(self) -> "Reversi6x6":
        env = Reversi6x6()
        env.board = self.board.copy()
        env.current_player = self.current_player
        env.pass_count = self.pass_count
        return env

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

    def _flips_for_move(self, r: int, c: int, player: int) -> List[Tuple[int, int]]:
        if self.board[r, c] != EMPTY:
            return []
        opponent = -player
        all_flips: List[Tuple[int, int]] = []
        for dr, dc in DIRECTIONS:
            rr, cc = r + dr, c + dc
            flips: List[Tuple[int, int]] = []
            if not self.in_bounds(rr, cc) or self.board[rr, cc] != opponent:
                continue
            while self.in_bounds(rr, cc) and self.board[rr, cc] == opponent:
                flips.append((rr, cc))
                rr += dr
                cc += dc
            if self.in_bounds(rr, cc) and self.board[rr, cc] == player:
                all_flips.extend(flips)
        return all_flips

    def legal_moves(self, player: Optional[int] = None) -> List[int]:
        player = self.current_player if player is None else player
        moves: List[int] = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self._flips_for_move(r, c, player):
                    moves.append(r * BOARD_SIZE + c)
        return moves

    def apply_move(self, move: Optional[int]) -> bool:
        if move is None:
            if self.legal_moves(self.current_player):
                return False
            self.pass_count += 1
            self.current_player *= -1
            return True

        r, c = divmod(move, BOARD_SIZE)
        flips = self._flips_for_move(r, c, self.current_player)
        if not flips:
            return False
        self.board[r, c] = self.current_player
        for rr, cc in flips:
            self.board[rr, cc] = self.current_player
        self.current_player *= -1
        self.pass_count = 0
        return True

    def is_terminal(self) -> bool:
        if np.all(self.board != EMPTY):
            return True
        if self.pass_count >= 2:
            return True
        if not self.legal_moves(BLACK) and not self.legal_moves(WHITE):
            return True
        return False

    def score(self) -> Tuple[int, int]:
        black = int(np.sum(self.board == BLACK))
        white = int(np.sum(self.board == WHITE))
        return black, white

    def winner(self) -> int:
        black, white = self.score()
        if black > white:
            return BLACK
        if white > black:
            return WHITE
        return 0

    def to_state(self, perspective: int) -> np.ndarray:
        own = (self.board == perspective).astype(np.float32)
        opp = (self.board == -perspective).astype(np.float32)
        turn = np.full((BOARD_SIZE, BOARD_SIZE), 1.0 if self.current_player == perspective else 0.0, dtype=np.float32)
        return np.stack([own, opp, turn], axis=0)


class QNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(BOARD_SIZE * BOARD_SIZE * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, BOARD_SIZE * BOARD_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    legal_mask: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000) -> None:
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, tr: Transition) -> None:
        self.buffer.append(tr)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

    def save(self, path: str) -> None:
        serializable = [
            {
                "state": t.state.tolist(),
                "action": t.action,
                "reward": t.reward,
                "next_state": t.next_state.tolist(),
                "done": t.done,
                "legal_mask": t.legal_mask.tolist(),
            }
            for t in self.buffer
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        for t in loaded:
            self.push(
                Transition(
                    state=np.array(t["state"], dtype=np.float32),
                    action=int(t["action"]),
                    reward=float(t["reward"]),
                    next_state=np.array(t["next_state"], dtype=np.float32),
                    done=bool(t["done"]),
                    legal_mask=np.array(t["legal_mask"], dtype=np.float32),
                )
            )


class DQNAgent:
    def __init__(self, lr: float = 1e-3, gamma: float = 0.99, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.q_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma

    def select_action(self, state: np.ndarray, legal_moves: List[int], epsilon: float) -> Optional[int]:
        if not legal_moves:
            return None
        if random.random() < epsilon:
            return random.choice(legal_moves)
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(x).squeeze(0).cpu().numpy()
        masked = np.full_like(q_values, -1e9)
        masked[legal_moves] = q_values[legal_moves]
        return int(np.argmax(masked))

    def train_step(self, buffer: ReplayBuffer, batch_size: int = 64) -> Optional[float]:
        if len(buffer) < batch_size:
            return None
        batch = buffer.sample(batch_size)

        states = torch.tensor(np.stack([b.state for b in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([b.action for b in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.stack([b.next_state for b in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([b.done for b in batch], dtype=torch.float32, device=self.device)
        masks = torch.tensor(np.stack([b.legal_mask for b in batch]), dtype=torch.float32, device=self.device)

        q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_all = self.target_net(next_states)
            invalid = (masks == 0)
            next_q_all = next_q_all.masked_fill(invalid, -1e9)
            next_q = torch.max(next_q_all, dim=1).values
            next_q = torch.where(torch.any(masks > 0, dim=1), next_q, torch.zeros_like(next_q))
            target = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.functional.mse_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def update_target(self) -> None:
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path: str) -> None:
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str) -> None:
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())


def create_legal_mask(moves: List[int]) -> np.ndarray:
    mask = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
    mask[moves] = 1.0
    return mask


def self_play_train(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    agent = DQNAgent(lr=args.lr, gamma=args.gamma, device=device)
    buffer = ReplayBuffer(capacity=args.buffer_size)

    if args.replay_log and os.path.exists(args.replay_log):
        buffer.load(args.replay_log)
        print(f"Loaded replay log: {args.replay_log} ({len(buffer)} transitions)")

    epsilon = args.epsilon_start
    step = 0
    for episode in range(1, args.episodes + 1):
        env = Reversi6x6()
        episode_data = {BLACK: [], WHITE: []}

        while not env.is_terminal():
            player = env.current_player
            state = env.to_state(player)
            legal = env.legal_moves(player)
            action = agent.select_action(state, legal, epsilon)

            env.apply_move(action)
            done = env.is_terminal()

            if action is not None:
                episode_data[player].append((state, action, env.to_state(player), done, create_legal_mask(env.legal_moves(env.current_player))))

        winner = env.winner()
        for player, transitions in episode_data.items():
            final_reward = 1.0 if winner == player else -1.0 if winner == -player else 0.0
            for state, action, next_state, done, legal_mask in transitions:
                reward = final_reward if done else 0.0
                buffer.push(Transition(state, action, reward, next_state, done, legal_mask))

        for _ in range(args.train_steps_per_episode):
            loss = agent.train_step(buffer, batch_size=args.batch_size)
            step += 1
            if loss is not None and step % args.target_update_interval == 0:
                agent.update_target()

        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)

        if episode % args.log_interval == 0:
            black, white = env.score()
            print(f"Episode {episode}/{args.episodes}, epsilon={epsilon:.3f}, buffer={len(buffer)}, score(B/W)={black}/{white}")

    model_path = os.path.join(args.out_dir, args.model_name)
    agent.save(model_path)
    print(f"Saved model to {model_path}")

    if args.save_replay_log:
        replay_path = os.path.join(args.out_dir, args.save_replay_log)
        buffer.save(replay_path)
        print(f"Saved replay log to {replay_path}")


def print_board(board: np.ndarray) -> None:
    symbols = {BLACK: "●", WHITE: "○", EMPTY: "."}
    print("  " + " ".join(str(i) for i in range(BOARD_SIZE)))
    for r in range(BOARD_SIZE):
        row = " ".join(symbols[int(board[r, c])] for c in range(BOARD_SIZE))
        print(f"{r} {row}")


def human_vs_cpu(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    agent = DQNAgent(device=device)
    agent.load(args.model)
    env = Reversi6x6()

    human_color = BLACK if args.human_color.lower() == "black" else WHITE

    while not env.is_terminal():
        print_board(env.board)
        player = env.current_player
        legal = env.legal_moves(player)
        turn_name = "BLACK" if player == BLACK else "WHITE"

        if not legal:
            print(f"{turn_name} は合法手がないためパス")
            env.apply_move(None)
            continue

        if player == human_color:
            print(f"あなたの手番 ({turn_name})。合法手: {[divmod(m, BOARD_SIZE) for m in legal]}")
            while True:
                raw = input("行 列 を入力 (例: 2 3): ").strip().split()
                if len(raw) != 2:
                    print("2つの整数を入力してください。")
                    continue
                try:
                    r, c = map(int, raw)
                except ValueError:
                    print("整数を入力してください。")
                    continue
                move = r * BOARD_SIZE + c
                if move in legal:
                    env.apply_move(move)
                    break
                print("合法手ではありません。")
        else:
            state = env.to_state(player)
            move = agent.select_action(state, legal, epsilon=0.0)
            env.apply_move(move)
            if move is not None:
                r, c = divmod(move, BOARD_SIZE)
                print(f"CPU ({turn_name}) は ({r}, {c}) に着手")

    print_board(env.board)
    black, white = env.score()
    print(f"最終スコア BLACK={black}, WHITE={white}")
    winner = env.winner()
    if winner == 0:
        print("引き分け")
    elif winner == human_color:
        print("あなたの勝ち")
    else:
        print("CPUの勝ち")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="6x6 Reversi with DQN self-play")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Self-play training")
    train.add_argument("--episodes", type=int, default=1000)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--buffer-size", type=int, default=100_000)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--gamma", type=float, default=0.99)
    train.add_argument("--epsilon-start", type=float, default=1.0)
    train.add_argument("--epsilon-end", type=float, default=0.05)
    train.add_argument("--epsilon-decay", type=float, default=0.995)
    train.add_argument("--target-update-interval", type=int, default=200)
    train.add_argument("--train-steps-per-episode", type=int, default=20)
    train.add_argument("--log-interval", type=int, default=20)
    train.add_argument("--out-dir", type=str, default="artifacts")
    train.add_argument("--model-name", type=str, default="dqn_reversi6x6.pt")
    train.add_argument("--replay-log", type=str, default=None, help="load existing replay log JSON")
    train.add_argument("--save-replay-log", type=str, default="replay_log.json", help="save replay log filename")
    train.add_argument("--cpu", action="store_true")

    play = sub.add_parser("play", help="Human vs CPU")
    play.add_argument("--model", type=str, required=True)
    play.add_argument("--human-color", choices=["black", "white"], default="black")
    play.add_argument("--cpu", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        self_play_train(args)
    elif args.command == "play":
        human_vs_cpu(args)


if __name__ == "__main__":
    main()
