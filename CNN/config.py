from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class TrainingConfig:
    board_size: int = 19
    data_paths: List[Path] = None
    output_dir: Path = Path('~/data/go_AI_runs/simple_cnn').expanduser()
    batch_size: int = 256
    num_workers: int = 4
    epochs: int = 5
    steps_per_epoch: int = 4000
    eval_steps: int = 800
    learning_rate: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    save_every: int = 1

    def resolve_data_paths(self) -> List[Path]:
        if self.data_paths is None:
            # 使用.data文件
            base_dir = Path(__file__).parent.parent
            training_data_dir = base_dir / 'Training_data'
            if training_data_dir.exists():
                data_files = sorted(training_data_dir.glob('*.data'))
                return data_files
            return [Path('~/data/go_AI_data/pure_data/19x19.data').expanduser()]
        return [Path(p).expanduser() for p in self.data_paths]
