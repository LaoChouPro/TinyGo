from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AverageMeter:
    name: str
    fmt: str = ':.4f'

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0

    def __str__(self) -> str:
        return f"{self.name} {self.val:{self.fmt}} (avg {self.avg:{self.fmt}})"


def topk_accuracy(logits: torch.Tensor, target: torch.Tensor, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = logits.topk(maxk, dim=1)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            results.append((correct_k * 100.0 / target.size(0)).item())
        return results
