# Simple Go CNN Training System

This folder contains a minimalist yet stable supervised training pipeline for predicting the next move in a Go game using the cleaned `.data` files.

## Components

- `datasets.py` – streaming dataset that rebuilds board states and emits `(features, move)` pairs.
- `model.py` – a compact CNN with residual blocks and a policy head.
- `metrics.py` – helpers for tracking average loss and top-k accuracy.
- `trainer.py` – high-level training loop with SGD, cosine LR schedule, checkpointing, and evaluation.
- `train.py` – command-line entry point.
- `config.py`, `utils.py` – configuration helpers, logging, checkpoint utilities.

## Data assumptions

- `.data` files live under `../Training_data/` (relative to this CNN directory).
- The system automatically detects and uses all `.data` files in the Training_data directory.
- Each line contains a JSON array representing board positions, not move sequences.
- Data format: `[{"B":[x,y]}, {"W":[x,y]}, ...]` where each dictionary represents a stone on the board.
- Coordinates are 1-indexed (typical Go notation).

## Data structure changes

**Important:** The current data format differs from the original format. Previously, the system expected move sequences `[color, x, y]`, but now it receives static board positions. The training logic has been adapted accordingly:

- Each `.data` file now represents a complete board position rather than a game sequence.
- The system generates training samples by finding all legal next moves from each position.
- This approach provides more training data per file but represents a different learning strategy.

## Quick start (sanity check)

Run a short training session on 19×19 data:

```bash
cd /Users/laochou/Desktop/编程/项目/TinyGo/CNN
python -m CNN.train \
  --board-size 19 \
  --output-dir ./output/debug \
  --epochs 1 \
  --steps-per-epoch 200 \
  --eval-steps 50 \
  --batch-size 128 \
  --num-workers 2
```

Note: The system will automatically detect and use all data files from `../Training_data/`.

## Recommended full training

```bash
python -m CNN.train \
  --board-size 19 \
  --output-dir ./output/full_training \
  --epochs 10 \
  --steps-per-epoch 4000 \
  --eval-steps 800 \
  --batch-size 256 \
  --num-workers 8
```

You can also specify custom data paths if needed:

```bash
python -m CNN.train \
  --board-size 19 \
  --data-paths ../Training_data/1.data ../Training_data/10.data \
  --output-dir ./output/custom_data \
  --epochs 5 \
  --steps-per-epoch 1000 \
  --eval-steps 200
```

Adjust `steps-per-epoch` and `eval-steps` to control training duration per epoch.

## Notes

- Checkpoints are saved every epoch (configurable via `--save-every`).
- Training resumes with `--resume /path/to/checkpoint_latest.pt`.
- Logs are written both to stdout and `<output_dir>/train.log`.
- The dataset loader partitions games deterministically based on their index: roughly 10% for validation.

Feel free to extend the pipeline to mix midgame data, add richer feature planes, or introduce value heads when you move beyond the minimal baseline.
