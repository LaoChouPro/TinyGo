from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import TrainingConfig
from datasets import DatasetConfig
from datasets import build_dataloader
from metrics import AverageMeter, topk_accuracy
from model import SimplePolicyNet
from utils import (
    configure_logging,
    load_checkpoint,
    prepare_output_dir,
    save_checkpoint,
    save_config,
    set_seed,
)


LOGGER = logging.getLogger(__name__)


class Trainer:
    def __init__(self, cfg: TrainingConfig) -> None:
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        self.output_dir = prepare_output_dir(cfg.output_dir)
        configure_logging(self.output_dir / 'train.log')
        LOGGER.info("Training configuration: %s", cfg)

        data_paths = cfg.resolve_data_paths()
        dataset_cfg = DatasetConfig(
            board_size=cfg.board_size,
            data_files=data_paths,
            val_ratio=0.0,  # 临时禁用验证集以确保SGF数据正常工作
        )
        self.train_loader = build_dataloader(
            dataset_cfg,
            mode='train',
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )
        # 暂时不创建验证加载器
        self.val_loader = None

        self.model = SimplePolicyNet(board_size=cfg.board_size)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.epochs * cfg.steps_per_epoch)
        self.start_epoch = 0

        save_config(cfg, self.output_dir / 'config.json')

    def maybe_load_checkpoint(self, path: Optional[Path]) -> None:
        if path is None or not path.exists():
            return
        checkpoint = load_checkpoint(path, self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.start_epoch = checkpoint.get('epoch', 0)
        LOGGER.info("Resumed from checkpoint %s at epoch %d", path, self.start_epoch)

    def run(self) -> None:
        global_step = self.start_epoch * self.cfg.steps_per_epoch
        for epoch in range(self.start_epoch, self.cfg.epochs):
            train_metrics, global_step = self.train_one_epoch(epoch, global_step)
            LOGGER.info(
                "Epoch %d train loss %.4f acc@1 %.2f%% acc@5 %.2f%%",
                epoch + 1,
                train_metrics['loss'],
                train_metrics['acc1'],
                train_metrics['acc5'],
            )
            val_metrics = self.evaluate(epoch)
            if val_metrics is not None:
                LOGGER.info(
                    "Epoch %d val   loss %.4f acc@1 %.2f%% acc@5 %.2f%%",
                    epoch + 1,
                    val_metrics['loss'],
                    val_metrics['acc1'],
                    val_metrics['acc5'],
                )
            if (epoch + 1) % self.cfg.save_every == 0:
                self.save_checkpoint(epoch)

    def train_one_epoch(self, epoch: int, global_step: int):
        self.model.train()
        loss_meter = AverageMeter('loss')
        acc1_meter = AverageMeter('acc1')
        acc5_meter = AverageMeter('acc5')

        iterator = iter(self.train_loader)

        # 创建进度条
        pbar = tqdm(total=self.cfg.steps_per_epoch,
                    desc=f'Epoch {epoch+1}',
                    unit='batch')

        steps = 0
        while steps < self.cfg.steps_per_epoch:
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(self.train_loader)
                inputs, targets = next(iterator)
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits = self.model(inputs)
            loss = self.criterion(logits, targets)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            acc1, acc5 = topk_accuracy(logits, targets, topk=(1, 5))
            batch_size = inputs.size(0)
            loss_meter.update(loss.item(), batch_size)
            acc1_meter.update(acc1, batch_size)
            acc5_meter.update(acc5, batch_size)

            steps += 1
            global_step += 1

            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss_meter.avg:.4f}',
                'Acc@1': f'{acc1_meter.avg:.2f}%',
                'Acc@5': f'{acc5_meter.avg:.2f}%'
            })
            pbar.update(1)

        pbar.close()
        return (
            {'loss': loss_meter.avg, 'acc1': acc1_meter.avg, 'acc5': acc5_meter.avg},
            global_step,
        )

    def evaluate(self, epoch: int):
        if self.val_loader is None:
            LOGGER.info("Skipping evaluation - no validation data available")
            return None

        self.model.eval()
        loss_meter = AverageMeter('loss')
        acc1_meter = AverageMeter('acc1')
        acc5_meter = AverageMeter('acc5')
        iterator = iter(self.val_loader)
        steps = 0
        with torch.no_grad():
            while steps < self.cfg.eval_steps:
                try:
                    inputs, targets = next(iterator)
                except StopIteration:
                    iterator = iter(self.val_loader)
                    inputs, targets = next(iterator)
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)
                acc1, acc5 = topk_accuracy(logits, targets, topk=(1, 5))
                batch_size = inputs.size(0)
                loss_meter.update(loss.item(), batch_size)
                acc1_meter.update(acc1, batch_size)
                acc5_meter.update(acc5, batch_size)
                steps += 1
        return {'loss': loss_meter.avg, 'acc1': acc1_meter.avg, 'acc5': acc5_meter.avg}

    def save_checkpoint(self, epoch: int) -> None:
        checkpoint = {
            'epoch': epoch + 1,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        path = self.output_dir / f'checkpoint_epoch_{epoch+1}.pt'
        save_checkpoint(checkpoint, path)
        latest = self.output_dir / 'checkpoint_latest.pt'
        save_checkpoint(checkpoint, latest)
        LOGGER.info('Checkpoint saved to %s', path)
