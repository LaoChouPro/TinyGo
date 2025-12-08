import tkinter as tk
from tkinter import messagebox
import torch
import numpy as np
import os
import threading
import sys

# Ensure imports work
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.board import GoBoard, BLACK, WHITE, SIZE, EMPTY
from src_large.model import TinyGoNet
from win_rate.src_small.model import ValueNetSmall
from src_mcts.mcts import MCTS

class GoGuiMCTS:
    def __init__(self, root, policy_path, value_path):
        self.root = root
        self.root.title("TinyGo MCTS AI")
        
        self.cell_size = 30
        self.margin = 30
        self.board_size = SIZE
        self.canvas_size = self.margin * 2 + self.cell_size * (self.board_size - 1)
        
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg="#DDBB88")
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.info_panel = tk.Frame(root)
        self.info_panel.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)
        
        self.status_label = tk.Label(self.info_panel, text="Status: Ready", font=("Arial", 12))
        self.status_label.pack(pady=5)
        
        self.turn_label = tk.Label(self.info_panel, text="Turn: Black (You)", font=("Arial", 12, "bold"))
        self.turn_label.pack(pady=5)
        
        self.ai_info_label = tk.Label(self.info_panel, text="", font=("Arial", 10), justify=tk.LEFT)
        self.ai_info_label.pack(pady=10)
        
        self.pass_btn = tk.Button(self.info_panel, text="Pass", command=self.human_pass)
        self.pass_btn.pack(pady=5)
        
        # self.undo_btn = tk.Button(self.info_panel, text="Undo", command=self.undo_move)
        # self.undo_btn.pack(pady=5)
        
        btn_frame = tk.Frame(self.info_panel)
        btn_frame.pack(pady=5)
        
        self.new_black_btn = tk.Button(btn_frame, text="New Game (Black)", command=lambda: self.reset_game(BLACK))
        self.new_black_btn.pack(side=tk.LEFT, padx=2)
        
        self.new_white_btn = tk.Button(btn_frame, text="New Game (White)", command=lambda: self.reset_game(WHITE))
        self.new_white_btn.pack(side=tk.LEFT, padx=2)
        
        # MCTS Config
        config_frame = tk.LabelFrame(self.info_panel, text="MCTS Config")
        config_frame.pack(pady=10, fill=tk.X)
        
        tk.Label(config_frame, text="Simulations:").pack(side=tk.LEFT)
        self.sim_var = tk.StringVar(value="100")
        tk.Entry(config_frame, textvariable=self.sim_var, width=5).pack(side=tk.LEFT)
        
        self.board = GoBoard()
        self.human_color = BLACK
        self.history = [] 
        
        self.game_over = False
        self.last_move = None
        self.is_ai_thinking = False
        
        # Load Models
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
            
        self.policy_path = policy_path
        self.value_path = value_path
        self.mcts = None
        
        self.load_ai_models()
        
        self.canvas.bind("<Button-1>", self.on_click)
        self.draw_board()
        
    def load_ai_models(self):
        self.status_label.config(text="Loading AI...")
        self.root.update()
        
        try:
            # Policy
            policy_model = TinyGoNet(num_blocks=10, channels=128)
            p_state = torch.load(self.policy_path, map_location=self.device)
            if isinstance(p_state, dict) and 'model_state_dict' in p_state:
                policy_model.load_state_dict(p_state['model_state_dict'])
            else:
                policy_model.load_state_dict(p_state)
            policy_model.to(self.device)
            
            # Value
            value_model = ValueNetSmall(num_blocks=6, channels=64, input_planes=2)
            v_state = torch.load(self.value_path, map_location=self.device)
            if isinstance(v_state, dict) and 'model_state_dict' in v_state:
                value_model.load_state_dict(v_state['model_state_dict'])
            else:
                value_model.load_state_dict(v_state)
            value_model.to(self.device)
            
            self.mcts = MCTS(policy_model, value_model, device=self.device, num_simulations=100)
            
            self.status_label.config(text="AI Ready")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {e}")
            # self.root.destroy()
            
    def reset_game(self, human_color=BLACK):
        if self.is_ai_thinking: return
        self.board.reset()
        self.human_color = human_color
        self.history = []
        self.game_over = False
        self.last_move = None
        self.update_ui()
        
        if self.human_color == WHITE:
            self.run_ai_move()
            
    def update_ui(self):
        self.draw_board()
        turn_str = "Black" if self.board.next_player == BLACK else "White"
        if not self.game_over:
            if self.board.next_player == self.human_color:
                self.turn_label.config(text=f"Turn: {turn_str} (You)", fg="black")
            else:
                self.turn_label.config(text=f"Turn: {turn_str} (AI)", fg="red")
        else:
            self.turn_label.config(text="Game Over", fg="blue")
            
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
                    self.canvas.create_oval(x-14, y-14, x+14, y+14, fill=fill_color, outline="black")
                    
        # Last Move Marker
        if self.last_move:
            r, c = self.last_move
            x = self.margin + c * self.cell_size
            y = self.margin + r * self.cell_size
            stone_color = self.board.board[r, c]
            mark_color = "white" if stone_color == BLACK else "black"
            self.canvas.create_rectangle(x-4, y-4, x+4, y+4, fill=mark_color, outline="")
            
    def on_click(self, event):
        if self.game_over or self.is_ai_thinking:
            return
        
        if self.board.next_player != self.human_color:
            return
            
        # Convert coords
        c = round((event.x - self.margin) / self.cell_size)
        r = round((event.y - self.margin) / self.cell_size)
        
        if 0 <= r < self.board_size and 0 <= c < self.board_size:
            if self.board.play(r, c, self.human_color) != -1:
                self.last_move = (r, c)
                self.board.next_player = 3 - self.human_color
                self.update_ui()
                self.run_ai_move()
            else:
                self.status_label.config(text="Invalid Move!")
                
    def human_pass(self):
        if self.game_over or self.is_ai_thinking: return
        if self.board.next_player != self.human_color: return
        
        # Pass
        self.board.next_player = 3 - self.human_color
        self.last_move = None # or indicate pass
        self.update_ui()
        self.run_ai_move()
        
    def run_ai_move(self):
        if self.game_over: return
        self.is_ai_thinking = True
        self.status_label.config(text="AI Thinking...")
        self.root.update()
        
        # Run in thread
        threading.Thread(target=self._ai_worker, daemon=True).start()
        
    def _ai_worker(self):
        try:
            sims = int(self.sim_var.get())
        except:
            sims = 100
        self.mcts.num_simulations = sims
        
        actions, probs = self.mcts.run(self.board, self.board.next_player, temperature=0.1)
        
        # Pick best
        best_idx = np.argmax(probs)
        r, c = actions[best_idx]
        prob = probs[best_idx]
        
        # Get Value from root?
        # Re-eval root value for display
        val = self.mcts._get_value(self.board) # Value for current player
        
        def update_on_main():
            self.board.play(r, c, self.board.next_player)
            self.last_move = (r, c)
            self.board.next_player = 3 - self.board.next_player
            self.is_ai_thinking = False
            self.status_label.config(text="Status: Ready")
            self.ai_info_label.config(text=f"AI Move: ({r}, {c})\nConfidence: {prob:.2%}\nWinRate Est: {val:.2%}")
            self.update_ui()
            
        self.root.after(0, update_on_main)

if __name__ == "__main__":
    root = tk.Tk()
    
    # Paths
    policy_path = os.path.join(project_root, 'best_model_large.pth')
    value_path = os.path.join(project_root, 'win_rate', 'src_small', 'best_model.pth')
    
    gui = GoGuiMCTS(root, policy_path, value_path)
    root.mainloop()
