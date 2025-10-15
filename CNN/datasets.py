from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset


Color = str  # alias for readability


class GoGameState:
    """Minimal Go board state to rebuild positions and apply captures."""

    def __init__(self, size: int) -> None:
        if size <= 0 or size > 25:
            raise ValueError(f"Unsupported board size: {size}")
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)  # 0 empty, 1 black, -1 white

    def apply_setup(self, color: Color, coords: Sequence[Tuple[int, int]]) -> None:
        value = 1 if color == 'B' else -1
        for x, y in coords:
            # 转换1-indexed坐标到0-indexed
            if 1 <= x <= self.size and 1 <= y <= self.size:
                self.board[y-1, x-1] = value

    def apply_empty(self, coords: Sequence[Tuple[int, int]]) -> None:
        for x, y in coords:
            # 转换1-indexed坐标到0-indexed
            if 1 <= x <= self.size and 1 <= y <= self.size:
                self.board[y-1, x-1] = 0

    def play_move(self, color: Color, coord: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = coord
        if self.board[y, x] != 0:
            raise ValueError("attempt to play on occupied point")
        value = 1 if color == 'B' else -1
        self.board[y, x] = value

        captured_groups: List[List[Tuple[int, int]]] = []
        for nx, ny in self._neighbors(x, y):
            if self.board[ny, nx] == -value:
                group, liberties = self._collect_group(nx, ny)
                if liberties == 0:
                    captured_groups.append(group)

        # 记录所有被提子的位置
        captured_stones = []
        for group in captured_groups:
            for gx, gy in group:
                self.board[gy, gx] = 0
                captured_stones.append((gx, gy))

        _, liberties = self._collect_group(x, y)
        if liberties == 0:
            # suicide move, revert captures and raise
            for group in captured_groups:
                for gx, gy in group:
                    self.board[gy, gx] = -value
            self.board[y, x] = 0
            raise ValueError("suicide move")

        return captured_stones

    def make_features(self, to_play: Color) -> np.ndarray:
        black = (self.board == 1).astype(np.float32)
        white = (self.board == -1).astype(np.float32)
        to_play_plane = np.full_like(black, 1.0 if to_play == 'B' else 0.0)
        return np.stack([black, white, to_play_plane], axis=0)

    def _neighbors(self, x: int, y: int) -> Iterator[Tuple[int, int]]:
        if x > 0:
            yield x - 1, y
        if x + 1 < self.size:
            yield x + 1, y
        if y > 0:
            yield x, y - 1
        if y + 1 < self.size:
            yield x, y + 1

    def _collect_group(self, x: int, y: int) -> Tuple[List[Tuple[int, int]], int]:
        value = self.board[y, x]
        stack = [(x, y)]
        visited = set(stack)
        group: List[Tuple[int, int]] = []
        liberties = set()
        while stack:
            cx, cy = stack.pop()
            group.append((cx, cy))
            for nx, ny in self._neighbors(cx, cy):
                v = self.board[ny, nx]
                if v == 0:
                    liberties.add((nx, ny))
                elif v == value and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    stack.append((nx, ny))
        return group, len(liberties)


def sgf_coord_to_xy(coord: Sequence[int]) -> Tuple[int, int]:
    x, y = coord
    return x - 1, y - 1


@dataclass
class DatasetConfig:
    board_size: int
    data_files: Sequence[Path]
    val_ratio: float = 0.1
    limit_games: Optional[int] = None  # optional cap for debugging


class GoMoveDataset(IterableDataset):
    """Stream go board states and next-move labels from .data files."""

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
        for path in self.cfg.data_files:
            with path.open('r', encoding='utf-8') as fh:
                for game_index, line in enumerate(fh):
                    if game_index % num_workers != worker_id:
                        continue
                    if self.cfg.limit_games is not None and games_seen >= self.cfg.limit_games:
                        return
                    games_seen += 1
                    if not self._selected(game_index):
                        continue
                    line = line.strip()
                    if not line:
                        continue
                    moves = json.loads(line)
                    state = GoGameState(self.board_size)
                    try:
                        yield from self._game_to_samples(state, moves)
                    except ValueError:
                        # ignore corrupted games
                        continue

    def _game_to_samples(
        self,
        state: GoGameState,
        moves: List[dict],
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        处理序列数据：围棋对局的走棋序列
        对于包含N步棋的对局，生成N个训练样本：
        样本i：使用前i-1步棋的盘面状态，预测第i步棋的位置
        """
        # 从空棋盘开始（假设没有让子）
        # state已经是初始化的空棋盘

        for i, move in enumerate(moves):
            if not isinstance(move, dict) or len(move) != 1:
                continue

            # 提取棋子颜色和位置
            color = list(move.keys())[0]
            coords = move[color]
            if not isinstance(coords, list) or len(coords) != 2:
                continue

            x, y = coords  # 1-indexed坐标

            # 验证坐标范围
            if not (1 <= x <= self.board_size and 1 <= y <= self.board_size):
                continue

            # 生成训练样本：用当前盘面状态预测这一步棋
            features = state.make_features(color)
            action = (y - 1) * self.board_size + (x - 1)  # 转换为0-indexed平面索引
            target = torch.tensor(action, dtype=torch.long)
            yield torch.from_numpy(features), target

            # 应用这步棋到盘面状态，为下一步做准备
            try:
                state.play_move(color, (x - 1, y - 1))
            except ValueError:
                # 如果是非法走法，停止处理这个对局
                break


def build_dataloader(
    cfg: DatasetConfig,
    mode: str,
    batch_size: int,
    num_workers: int,
) -> torch.utils.data.DataLoader:
    dataset = GoMoveDataset(cfg, mode)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
