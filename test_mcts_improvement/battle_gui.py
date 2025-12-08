import tkinter as tk
from tkinter import ttk
import threading
import sys
import os
import torch
import numpy as np
import time

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.board import GoBoard, BLACK, WHITE, SIZE, EMPTY
from src_large.model import TinyGoNet
from win_rate.src_small.model import ValueNetSmall
from src_mcts.mcts import MCTS
from src_large.inference import predict_move as predict_move_large

class BattleArena:
    def __init__(self, root):
        self.root = root
        self.root.title("MCTS (Black) vs PolicyNet (White)")
        
        # UI Layout
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Board Canvas
        self.cell_size = 25
        self.margin = 25
        self.board_size = SIZE
        self.canvas_size = self.margin * 2 + self.cell_size * (self.board_size - 1)
        
        self.canvas = tk.Canvas(self.main_frame, width=self.canvas_size, height=self.canvas_size, bg="#DDBB88")
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Info Panel
        self.info_panel = tk.Frame(self.main_frame, width=250)
        self.info_panel.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)
        
        # Title
        tk.Label(self.info_panel, text="MCTS (Black)\nvs\nPolicyNet (White)", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self.info_panel, textvariable=self.status_var, font=("Arial", 11)).pack(pady=5)
        
        # Win Rate Graph (Simplified as Text Bars for now)
        tk.Label(self.info_panel, text="Win Rate Estimate (Black)", font=("Arial", 10, "bold")).pack(pady=(20, 5))
        self.wr_bar = ttk.Progressbar(self.info_panel, orient="horizontal", length=200, mode="determinate")
        self.wr_bar.pack(pady=5)
        self.wr_text = tk.StringVar(value="50%")
        tk.Label(self.info_panel, textvariable=self.wr_text).pack()
        
        # Controls
        self.btn_frame = tk.Frame(self.info_panel)
        self.btn_frame.pack(pady=20)
        
        self.start_btn = tk.Button(self.btn_frame, text="Start Battle", command=self.start_battle, bg="#4CAF50", fg="white")
        self.start_btn.pack(fill=tk.X, pady=2)
        
        self.stop_btn = tk.Button(self.btn_frame, text="Stop", command=self.stop_battle, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=2)
        
        # Game State
        self.board = GoBoard()
        self.is_running = False
        self.game_thread = None
        self.last_move = None
        
        # Models
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_models()
        
        self.draw_board()
        
    def load_models(self):
        print(f"Loading models on {self.device}...")
        policy_path = os.path.join(project_root, 'best_model_large.pth')
        value_path = os.path.join(project_root, 'win_rate', 'src_small', 'best_model.pth')
        
        # 1. MCTS Policy
        self.mcts_policy = TinyGoNet(num_blocks=10, channels=128).to(self.device)
        self.mcts_policy.load_state_dict(torch.load(policy_path, map_location=self.device)['model_state_dict'])
        self.mcts_policy.eval()
        
        # 2. MCTS Value
        self.mcts_value = ValueNetSmall(num_blocks=6, channels=64, input_planes=2).to(self.device)
        self.mcts_value.load_state_dict(torch.load(value_path, map_location=self.device)['model_state_dict'])
        self.mcts_value.eval()
        
        # 3. Pure Policy (Same weights as MCTS Policy, separate instance just in case)
        self.pure_policy = TinyGoNet(num_blocks=10, channels=128).to(self.device)
        self.pure_policy.load_state_dict(torch.load(policy_path, map_location=self.device)['model_state_dict'])
        self.pure_policy.eval()
        
        self.mcts = MCTS(self.mcts_policy, self.mcts_value, device=self.device, num_simulations=100) # Stronger MCTS
        
    def draw_board(self):
        self.canvas.delete("all")
        # Grid
        for i in range(self.board_size):
            x = self.margin + i * self.cell_size
            y = self.margin + i * self.cell_size
            self.canvas.create_line(self.margin, y, self.canvas_size - self.margin, y)
            self.canvas.create_line(x, self.margin, x, self.canvas_size - self.margin)
        
        # Stars
        stars = [3, 9, 15]
        for r in stars:
            for c in stars:
                x = self.margin + c * self.cell_size
                y = self.margin + r * self.cell_size
                self.canvas.create_oval(x-2, y-2, x+2, y+2, fill="black")
                
        # Stones
        for r in range(self.board_size):
            for c in range(self.board_size):
                color = self.board.board[r, c]
                if color != EMPTY:
                    x = self.margin + c * self.cell_size
                    y = self.margin + r * self.cell_size
                    fill_color = "black" if color == BLACK else "white"
                    self.canvas.create_oval(x-11, y-11, x+11, y+11, fill=fill_color, outline="black")
                    
        # Last Move
        if self.last_move:
            r, c = self.last_move
            x = self.margin + c * self.cell_size
            y = self.margin + r * self.cell_size
            color = self.board.board[r, c]
            mark = "white" if color == BLACK else "black"
            self.canvas.create_rectangle(x-3, y-3, x+3, y+3, fill=mark, outline="")

    def start_battle(self):
        if self.is_running: return
        self.is_running = True
        self.board.reset()
        self.last_move = None
        self.draw_board()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        self.game_thread = threading.Thread(target=self.game_loop, daemon=True)
        self.game_thread.start()
        
    def stop_battle(self):
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Stopped")
        
    def game_loop(self):
        move_count = 0
        passes = 0
        
        while self.is_running and move_count < 300 and passes < 2:
            current_color = self.board.next_player
            
            # Update Status
            player_name = "MCTS (Black)" if current_color == BLACK else "PolicyNet (White)"
            self.status_var.set(f"Thinking: {player_name}")
            
            start_time = time.time()
            
            if current_color == BLACK:
                # MCTS
                actions, probs = self.mcts.run(self.board, BLACK, temperature=0.5)
                best_idx = np.argmax(probs)
                move = actions[best_idx]
                win_rate = self.mcts._get_value(self.board) # WR for Black
            else:
                # Pure Policy
                moves = predict_move_large(self.pure_policy, self.board, WHITE, self.device)
                # moves is list of ((r,c), prob)
                # Pick top 1 (Greedy)
                if not moves:
                    move = None # Pass
                else:
                    move = moves[0][0]
                
                # Evaluate position for graph (using MCTS Value net for consistency)
                win_rate = self.mcts._get_value(self.board) # WR for White (Next player is Black?)
                # Wait, _get_value(board) returns Win Prob for board.next_player.
                # If current is WHITE, board.next_player is WHITE.
                # So it returns WR for White.
                # Black WR = 1 - White WR.
                win_rate = 1.0 - win_rate
            
            elapsed = time.time() - start_time
            
            # Apply Move
            if move is None:
                passes += 1
                self.status_var.set(f"{player_name} Passed")
            else:
                passes = 0
                r, c = move
                if self.board.play(r, c, current_color) == -1:
                    print(f"Invalid move by {player_name}: {r},{c}")
                    break
                self.last_move = (r, c)
                self.board.next_player = 3 - current_color
                
            move_count += 1
            
            # Update UI
            self.root.after(0, lambda wr=win_rate: self.update_ui_threadsafe(wr))
            
            # Small delay for visual
            time.sleep(0.1)
            
        self.is_running = False
        self.root.after(0, lambda: self.stop_battle())
        
    def update_ui_threadsafe(self, win_rate):
        self.draw_board()
        # Update bar (Black WR)
        self.wr_bar['value'] = win_rate * 100
        self.wr_text.set(f"Black Win Rate: {win_rate:.1%}")

if __name__ == "__main__":
    root = tk.Tk()
    app = BattleArena(root)
    root.mainloop()
