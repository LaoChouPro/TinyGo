from __future__ import annotations

import argparse
from pathlib import Path

from config import TrainingConfig
from trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train simple Go CNN policy network')
    parser.add_argument('--board-size', type=int, default=19)
    parser.add_argument('--data-paths', nargs='*', default=None, help='List of .data files (default 19x19)')
    parser.add_argument('--output-dir', type=str, default='~/data/go_AI_runs/simple_cnn')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--steps-per-epoch', type=int, default=4000)
    parser.add_argument('--eval-steps', type=int, default=800)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save-every', type=int, default=1)
    parser.add_argument('--device', type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainingConfig(
        board_size=args.board_size,
        data_paths=[Path(p) for p in args.data_paths] if args.data_paths else None,
        output_dir=Path(args.output_dir).expanduser(),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        eval_steps=args.eval_steps,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        seed=args.seed,
        save_every=args.save_every,
        device=args.device or ('cuda' if __import__('torch').cuda.is_available() else 'cpu'),
    )
    trainer = Trainer(cfg)

    # 如果没有指定resume参数，自动检查输出目录中是否有checkpoint
    if args.resume is None:
        checkpoint_path = cfg.output_dir / 'checkpoint_latest.pt'
        if checkpoint_path.exists():
            print(f"自动发现checkpoint: {checkpoint_path}")
            trainer.maybe_load_checkpoint(checkpoint_path)
        else:
            print("未发现checkpoint，从头开始训练")
    else:
        trainer.maybe_load_checkpoint(Path(args.resume).expanduser())

    trainer.run()


if __name__ == '__main__':
    main()
