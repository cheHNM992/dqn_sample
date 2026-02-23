# dqn_sample

6×6オセロ（Reversi）環境と、DQNによる自己対戦学習エージェントのサンプル実装です。

## できること

- 6×6オセロのルールを実装（合法手判定、反転、パス、終了判定、勝敗判定）
- DQNによる自己対戦学習
- 自己対戦リプレイログ（経験再生バッファ）の保存・読み込み
- 人間 vs CPU（人間先攻/後攻の両対応）

## 動作環境

- Python 3.9+
- PyTorch
- NumPy

## インストール例

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy
```

## 学習

```bash
python reversi_dqn.py train --episodes 1000 --out-dir artifacts
```

既存ログを使って学習を継続する例:

```bash
python reversi_dqn.py train \
  --episodes 500 \
  --replay-log artifacts/replay_log.json \
  --out-dir artifacts
```

出力されるもの:

- `artifacts/dqn_reversi6x6.pt`（モデル）
- `artifacts/replay_log.json`（自己対戦ログ）

## 対戦（人間 vs CPU）

人間先攻（黒）:

```bash
python reversi_dqn.py play --model artifacts/dqn_reversi6x6.pt --human-color black
```

人間後攻（白）:

```bash
python reversi_dqn.py play --model artifacts/dqn_reversi6x6.pt --human-color white
```
