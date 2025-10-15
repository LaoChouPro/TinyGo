from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from model import SimplePolicyNet
from datasets import GoGameState
from utils import load_checkpoint

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Go against the trained CNN policy")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_latest.pt or specific epoch checkpoint")
    parser.add_argument("--board-size", type=int, default=19)
    parser.add_argument("--human-color", choices=["B", "W", "black", "white"], default="B")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda or cpu (defaults to auto)")
    parser.add_argument("--topk", type=int, default=5, help="Show top-k AI move suggestions")
    return parser.parse_args()


def to_xy(index: int, size: int) -> Tuple[int, int]:
    y, x = divmod(index, size)
    return x, y


def to_index(x: int, y: int, size: int) -> int:
    return y * size + x


def format_board(state: GoGameState) -> str:
    size = state.size
    header = "    " + " ".join(f"{i:2d}" for i in range(1, size + 1))
    rows: List[str] = [header]
    display_map = {0: ".", 1: "X", -1: "O"}
    for row in range(size - 1, -1, -1):
        line = [f"{row + 1:2d} |"]
        for col in range(size):
            val = display_map[state.board[row, col]]
            line.append(val)
        rows.append(" ".join(line))
    return "\n".join(rows)


def human_move(state: GoGameState, color: str, move_str: str) -> bool:
    move_str = move_str.strip().lower()
    if move_str in {"pass", "p"}:
        return True
    if move_str in {"quit", "exit", "resign"}:
        raise SystemExit("Game ended by user.")
    parts = move_str.replace(",", " ").split()
    if len(parts) != 2:
        print("请输入坐标，如 '4 4' 或输入 pass")
        return False
    try:
        x = int(parts[0])
        y = int(parts[1])
    except ValueError:
        print("坐标需要为整数")
        return False
    if not (1 <= x <= state.size and 1 <= y <= state.size):
        print("坐标超出棋盘范围")
        return False
    try:
        state.play_move(color, (x - 1, y - 1))
    except ValueError as exc:
        print(f"非法落子: {exc}")
        return False
    return True


def ai_move(
    model: SimplePolicyNet,
    state: GoGameState,
    color: str,
    device: torch.device,
    topk: int,
) -> Tuple[Optional[Tuple[int, int]], List[Tuple[int, int, float]]]:
    features = state.make_features(color)
    tensor = torch.from_numpy(features).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    size = state.size
    order = np.argsort(probs)[::-1]
    suggestions: List[Tuple[int, int, float]] = []
    move_coord: Optional[Tuple[int, int]] = None
    for idx in order:
        x, y = to_xy(idx, size)
        if state.board[y, x] != 0:
            continue
        prob = float(probs[idx])
        if len(suggestions) < topk:
            suggestions.append((x + 1, y + 1, prob))
        try:
            # simulate move on copy to ensure legality
            temp = GoGameState(size)
            temp.board = state.board.copy()
            temp.play_move(color, (x, y))
        except ValueError:
            continue
        move_coord = (x, y)
        break
    return move_coord, suggestions


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    model = SimplePolicyNet(board_size=args.board_size)
    state_dict = load_checkpoint(checkpoint_path, device)
    if 'model' in state_dict:
        model.load_state_dict(state_dict['model'])
    else:
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"加载模型: {checkpoint_path}")
    print(f"使用设备: {device}")

    state = GoGameState(args.board_size)
    human_color = 'B' if args.human_color.lower().startswith('b') else 'W'
    current = 'B'
    move_count = 0

    while True:
        print("\n当前棋盘:")
        print(format_board(state))
        if current == human_color:
            move_input = input(f"轮到你 ({current})，请输入坐标或 pass: ")
            if human_move(state, current, move_input):
                current = 'W' if current == 'B' else 'B'
                move_count += 1
        else:
            coord, suggestions = ai_move(model, state, current, device, args.topk)
            print(f"AI ({current}) 建议: ")
            for i, (sx, sy, prob) in enumerate(suggestions, start=1):
                print(f"  Top{i}: ({sx:2d}, {sy:2d}) 概率 {prob:.4f}")
            if coord is None:
                print("AI 无合法落子，选择 PASS")
            else:
                cx, cy = coord
                print(f"AI 落子 ({cx + 1}, {cy + 1})")
                state.play_move(current, coord)
            current = 'W' if current == 'B' else 'B'
            move_count += 1

        if move_count >= args.board_size * args.board_size * 2:
            print("达到设定步数上限，结束对局。")
            break


if __name__ == '__main__':
    main()
