from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_output_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _to_serialisable(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_to_serialisable(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_serialisable(v) for v in value)
    if isinstance(value, dict):
        return {k: _to_serialisable(v) for k, v in value.items()}
    return value


def save_config(cfg: Any, path: Path) -> None:
    serialisable = {key: _to_serialisable(value) for key, value in asdict(cfg).items()}
    path.write_text(json.dumps(serialisable, indent=2), encoding='utf-8')


def configure_logging(log_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(),
        ],
    )


def save_checkpoint(state: Dict[str, Any], path: Path) -> None:
    torch.save(state, path)


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    return torch.load(path, map_location=device)
