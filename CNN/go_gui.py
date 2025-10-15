#!/usr/bin/env python3
"""
围棋GUI程序 - 基于tkinter实现与AI模型对弈
"""

from __future__ import annotations

import argparse
import logging
import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path
from typing import Optional, Tuple, List
import threading
import time

import numpy as np
import torch

from model import SimplePolicyNet
from datasets import GoGameState
from utils import load_checkpoint

LOGGER = logging.getLogger(__name__)


class GoBoardCanvas(tk.Canvas):
    """围棋棋盘Canvas组件"""

    def __init__(self, parent, size: int = 19, cell_size: int = 30):
        self.size = size
        self.cell_size = cell_size
        self.board_size = (size - 1) * cell_size
        canvas_size = self.board_size + 2 * cell_size

        super().__init__(
            parent,
            width=canvas_size,
            height=canvas_size,
            bg='#DEB887',  # 棋盘颜色
            highlightthickness=1,
            highlightbackground='black'
        )

        self.stones = {}  # 存储棋子位置 {(x,y): color}
        self.last_move = None  # 最后一步棋
        self.captured_stones = []  # 被提子的位置（用于显示效果）
        self.capture_animation_timer = None

        self.draw_board()
        self.bind('<Button-1>', self.on_click)
        self.bind('<Motion>', self.on_mouse_move)

    def draw_board(self):
        """绘制棋盘"""
        self.delete("all")

        # 绘制网格线
        offset = self.cell_size
        for i in range(self.size):
            # 垂直线
            x = offset + i * self.cell_size
            self.create_line(x, offset, x, offset + self.board_size, width=1, fill='black')
            # 水平线
            y = offset + i * self.cell_size
            self.create_line(offset, y, offset + self.board_size, y, width=1, fill='black')

        # 绘制星位点
        star_points = []
        if self.size == 19:
            star_points = [(3,3), (3,9), (3,15), (9,3), (9,9), (9,15), (15,3), (15,9), (15,15)]
        elif self.size == 13:
            star_points = [(3,3), (3,9), (6,6), (9,3), (9,9)]
        elif self.size == 9:
            star_points = [(2,2), (2,6), (4,4), (6,2), (6,6)]

        for x, y in star_points:
            px = offset + x * self.cell_size
            py = offset + y * self.cell_size
            self.create_oval(px-3, py-3, px+3, py+3, fill='black', outline='black')

        # 绘制坐标标签
        for i in range(self.size):
            # 横坐标
            x = offset + i * self.cell_size
            self.create_text(x, offset//2, text=str(i+1), font=('Arial', 8))
            self.create_text(x, offset + self.board_size + offset//2, text=str(i+1), font=('Arial', 8))
            # 纵坐标
            y = offset + i * self.cell_size
            self.create_text(offset//2, y, text=str(self.size-i), font=('Arial', 8))
            self.create_text(offset + self.board_size + offset//2, y, text=str(self.size-i), font=('Arial', 8))

    def redraw(self):
        """重新绘制棋盘和棋子"""
        self.draw_board()

        # 绘制被提子的位置（红色X标记）
        for x, y in self.captured_stones:
            px, py = self.coord_to_pixel(x, y)
            # 绘制红色X标记
            size = self.cell_size // 4
            self.create_line(px-size, py-size, px+size, py+size, fill='red', width=3)
            self.create_line(px-size, py+size, px+size, py-size, fill='red', width=3)

        # 绘制棋子
        for (x, y), color in self.stones.items():
            px, py = self.coord_to_pixel(x, y)
            stone_color = 'black' if color == 1 else 'white'
            outline_color = 'white' if color == 1 else 'black'

            self.create_oval(
                px - self.cell_size//3, py - self.cell_size//3,
                px + self.cell_size//3, py + self.cell_size//3,
                fill=stone_color, outline=outline_color, width=2
            )

        # 标记最后一步棋
        if self.last_move:
            px, py = self.coord_to_pixel(*self.last_move)
            self.create_oval(px-4, py-4, px+4, py+4, fill='red', outline='red')

    def coord_to_pixel(self, x: int, y: int) -> Tuple[int, int]:
        """棋盘坐标转换为像素坐标"""
        px = self.cell_size + x * self.cell_size
        py = self.cell_size + y * self.cell_size
        return px, py

    def pixel_to_coord(self, px: int, py: int) -> Optional[Tuple[int, int]]:
        """像素坐标转换为棋盘坐标"""
        x = round((px - self.cell_size) / self.cell_size)
        y = round((py - self.cell_size) / self.cell_size)

        if 0 <= x < self.size and 0 <= y < self.size:
            return x, y
        return None

    def add_stone(self, x: int, y: int, color: int, captured_stones: List[Tuple[int, int]] = None):
        """在棋盘上放置棋子"""
        self.stones[(x, y)] = color
        self.last_move = (x, y)

        # 移除被提子
        if captured_stones:
            for cx, cy in captured_stones:
                if (cx, cy) in self.stones:
                    del self.stones[(cx, cy)]

            # 显示提子效果
            self.show_capture_effect(captured_stones)
        else:
            self.redraw()

    def show_capture_effect(self, captured_stones: List[Tuple[int, int]]):
        """显示提子效果"""
        self.captured_stones = captured_stones
        self.redraw()

        # 2秒后清除提子标记
        if self.capture_animation_timer:
            self.after_cancel(self.capture_animation_timer)
        self.capture_animation_timer = self.after(2000, self._clear_capture_effect)

    def _clear_capture_effect(self):
        """清除提子效果"""
        self.captured_stones = []
        self.redraw()


    def on_click(self, event):
        """鼠标点击事件"""
        coord = self.pixel_to_coord(event.x, event.y)
        if coord and hasattr(self, 'click_callback'):
            self.click_callback(coord[0], coord[1])

    def on_mouse_move(self, event):
        """鼠标移动事件（可以添加悬停效果）"""
        pass


class GoGameGUI:
    """围棋游戏GUI主类"""

    def __init__(self, checkpoint_path: str, board_size: int = 19, human_color: str = 'B'):
        self.root = tk.Tk()
        self.root.title("围棋AI对弈")
        self.root.resizable(False, False)

        # 游戏参数
        self.board_size = board_size
        self.human_color = human_color.upper()
        self.ai_color = 'W' if self.human_color == 'B' else 'B'
        self.current_player = 'B'
        self.game_active = True
        self.move_count = 0

        # 加载AI模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(checkpoint_path)

        # 游戏状态
        self.game_state = GoGameState(board_size)

        # 创建UI组件
        self._create_widgets()

        # 如果AI先手，立即下棋
        if self.current_player == self.ai_color:
            self.root.after(1000, self._ai_move)

    def _load_model(self, checkpoint_path: str) -> SimplePolicyNet:
        """加载AI模型"""
        try:
            checkpoint_path = Path(checkpoint_path).expanduser()
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"找不到模型文件: {checkpoint_path}")

            model = SimplePolicyNet(board_size=self.board_size)
            state_dict = load_checkpoint(checkpoint_path, self.device)

            if 'model' in state_dict:
                model.load_state_dict(state_dict['model'])
            else:
                model.load_state_dict(state_dict)

            model.to(self.device)
            model.eval()

            LOGGER.info(f"成功加载模型: {checkpoint_path}")
            return model

        except Exception as e:
            messagebox.showerror("错误", f"加载模型失败: {e}")
            raise

    def _create_widgets(self):
        """创建UI组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 棋盘
        self.board_canvas = GoBoardCanvas(main_frame, size=self.board_size)
        self.board_canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=10)
        self.board_canvas.click_callback = self._on_board_click

        # 游戏信息面板
        info_frame = ttk.LabelFrame(main_frame, text="游戏信息", padding="5")
        info_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        # 状态标签
        self.status_label = ttk.Label(info_frame, text="", font=('Arial', 12, 'bold'))
        self.status_label.grid(row=0, column=0, columnspan=3, pady=5)

        # 控制按钮
        control_frame = ttk.Frame(info_frame)
        control_frame.grid(row=1, column=0, columnspan=3, pady=10)

        ttk.Button(control_frame, text="Pass", command=self._human_pass).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="重新开始", command=self._restart_game).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="退出", command=self.root.quit).grid(row=0, column=2, padx=5)

        # 更新状态显示
        self._update_status()

    def _update_status(self):
        """更新状态显示"""
        if not self.game_active:
            self.status_label.config(text="游戏结束")
            return

        current_color_name = "黑棋" if self.current_player == 'B' else "白棋"
        player_type = "你" if self.current_player == self.human_color else "AI"

        self.status_label.config(text=f"当前轮到: {current_color_name} ({player_type})")

    def _on_board_click(self, x: int, y: int):
        """棋盘点击处理"""
        if not self.game_active or self.current_player != self.human_color:
            return

        try:
            # 检查位置是否已被占用
            if (x, y) in self.board_canvas.stones:
                messagebox.showwarning("警告", "此位置已有棋子")
                return

            # 尝试下棋
            captured_stones = self.game_state.play_move(self.current_player, (x, y))
            self.board_canvas.add_stone(x, y, 1 if self.current_player == 'B' else -1, captured_stones)

            # 切换玩家
            self._switch_player()
            self.move_count += 1

            # AI回应
            if self.game_active and self.current_player == self.ai_color:
                self.root.after(500, self._ai_move)

        except ValueError as e:
            messagebox.showerror("错误", f"非法落子: {e}")

    def _human_pass(self):
        """人类玩家Pass"""
        if not self.game_active or self.current_player != self.human_color:
            return

        self._switch_player()
        self.move_count += 1

        if self.game_active and self.current_player == self.ai_color:
            self.root.after(500, self._ai_move)

    def _ai_move(self):
        """AI下棋"""
        if not self.game_active or self.current_player != self.ai_color:
            return

        try:
            # 获取AI落子
            coord, _ = self._get_ai_move()

            # 执行AI落子
            if coord is None:
                # AI选择Pass
                pass
            else:
                x, y = coord
                captured_stones = self.game_state.play_move(self.current_player, coord)
                self.board_canvas.add_stone(x, y, 1 if self.current_player == 'B' else -1, captured_stones)

            # 切换玩家
            self._switch_player()
            self.move_count += 1

        except Exception as e:
            messagebox.showerror("错误", f"AI下棋出错: {e}")
            self.game_active = False
            self._update_status()

    def _get_ai_move(self) -> Tuple[Optional[Tuple[int, int]], List[Tuple[int, int, float]]]:
        """获取AI推荐落子位置"""
        features = self.game_state.make_features(self.current_player)
        tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # 按概率排序
        order = np.argsort(probs)[::-1]
        suggestions = []
        best_move = None

        for idx in order:
            y, x = divmod(idx, self.board_size)  # 注意这里的坐标转换

            # 跳过已占用的位置
            if self.game_state.board[y, x] != 0:
                continue

            prob = float(probs[idx])
            suggestions.append((x + 1, y + 1, prob))  # 转换为1-based坐标用于显示

            # 验证落子合法性
            if best_move is None:
                try:
                    temp_state = GoGameState(self.board_size)
                    temp_state.board = self.game_state.board.copy()
                    temp_state.play_move(self.current_player, (x, y))
                    best_move = (x, y)
                except ValueError:
                    continue  # 非法位置，尝试下一个

        return best_move, suggestions

    def _switch_player(self):
        """切换当前玩家"""
        self.current_player = 'W' if self.current_player == 'B' else 'B'
        self._update_status()

        # 检查游戏是否结束
        if self.move_count >= self.board_size * self.board_size:
            self.game_active = False
            self._update_status()

    def _restart_game(self):
        """重新开始游戏"""
        self.game_state = GoGameState(self.board_size)
        self.board_canvas.stones = {}
        self.board_canvas.last_move = None
        self.board_canvas.captured_stones = []
        if self.board_canvas.capture_animation_timer:
            self.board_canvas.after_cancel(self.board_canvas.capture_animation_timer)
        self.board_canvas.redraw()

        self.current_player = 'B'
        self.game_active = True
        self.move_count = 0
        self._update_status()

        # 如果AI先手，立即下棋
        if self.current_player == self.ai_color:
            self.root.after(1000, self._ai_move)

    def run(self):
        """运行GUI程序"""
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="围棋GUI对弈程序")
    parser.add_argument("--checkpoint", required=True, help="模型检查点文件路径")
    parser.add_argument("--board-size", type=int, default=19, choices=[9, 13, 19], help="棋盘大小")
    parser.add_argument("--human-color", choices=["B", "W", "black", "white"], default="B", help="人类玩家颜色")
    return parser.parse_args()


def main():
    """主函数"""
    logging.basicConfig(level=logging.INFO)

    args = parse_args()

    try:
        app = GoGameGUI(
            checkpoint_path=args.checkpoint,
            board_size=args.board_size,
            human_color=args.human_color
        )
        app.run()

    except Exception as e:
        print(f"程序启动失败: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())