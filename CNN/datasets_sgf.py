from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import IterableDataset
from sgfmill import sgf
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

# 重用GoGameState类
from datasets import GoGameState


Color = str  # alias for readability


@dataclass
class DatasetConfig:
    board_size: int
    data_files: Sequence[Path]
    val_ratio: float = 0.1
    limit_games: Optional[int] = None  # optional cap for debugging


class SgfGoMoveDataset(IterableDataset):
    """Stream go board states and next-move labels from SGF files."""

    def __init__(self, cfg: DatasetConfig, mode: str) -> None:
        if mode not in {"train", "val"}:
            raise ValueError("mode must be 'train' or 'val'")
        self.cfg = cfg
        self.mode = mode
        self.board_size = cfg.board_size

    def _selected(self, game_index: int) -> bool:
        if self.cfg.val_ratio <= 0:
            return self.mode == "train"
        bucket = game_index % 1000
        threshold = int(self.cfg.val_ratio * 1000)
        return bucket < threshold if self.mode == "val" else bucket >= threshold

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        games_seen = 0
        for file_index, sgf_path in enumerate(self.cfg.data_files):
            if file_index % num_workers != worker_id:
                continue
            if self.cfg.limit_games is not None and games_seen >= self.cfg.limit_games:
                return
            if not self._selected(file_index):
                continue

            try:
                with open(sgf_path, 'rb') as f:
                    game = sgf.Sgf_game.from_bytes(f.read())

                # 检查棋盘大小
                if game.get_size() != self.board_size:
                    continue

                # 获取主序列
                main_sequence = game.get_main_sequence()

                # 从空棋盘开始重建游戏状态
                state = GoGameState(self.board_size)

                # 跳过根节点，处理每一步棋
                for i, node in enumerate(main_sequence[1:], 1):  # 跳过根节点
                    move = node.get_move()
                    if move[0] is None:  # PASS
                        continue

                    color, pos = move
                    x, y = pos  # SGF使用0-indexed坐标

                    # 验证坐标范围
                    if not (0 <= x < self.board_size and 0 <= y < self.board_size):
                        continue

                    # 生成训练样本：在当前状态下预测这一步
                    features = state.make_features(color)
                    action = y * self.board_size + x  # 转换为平面索引
                    target = torch.tensor(action, dtype=torch.long)
                    yield torch.from_numpy(features), target

                    # 应用这步棋到棋盘状态
                    try:
                        state.play_move(color, (x, y))
                    except ValueError:
                        # 如果是非法走法，停止处理这个游戏
                        break

                games_seen += 1

            except Exception:
                # 忽略损坏的SGF文件
                continue


def build_sgf_dataloader(
    cfg: DatasetConfig,
    mode: str,
    batch_size: int,
    num_workers: int,
) -> torch.utils.data.DataLoader:
    dataset = SgfGoMoveDataset(cfg, mode)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )