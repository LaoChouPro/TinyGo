import tkinter as tk
from tkinter import messagebox
import torch
import numpy as np
import os
import threading
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.board import GoBoard, BLACK, WHITE, SIZE, EMPTY
# Use large model inference
from src_large.inference import load_model, predict_move

class GoGui:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("TinyGo AI (Large) - Play against Best Model")
        
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
        
        self.undo_btn = tk.Button(self.info_panel, text="Undo (2 steps)", command=self.undo_move)
        self.undo_btn.pack(pady=5)
        
        btn_frame = tk.Frame(self.info_panel)
        btn_frame.pack(pady=5)
        
        self.new_black_btn = tk.Button(btn_frame, text="New Game (Black)", command=lambda: self.reset_game(BLACK))
        self.new_black_btn.pack(side=tk.LEFT, padx=2)
        
        self.new_white_btn = tk.Button(btn_frame, text="New Game (White)", command=lambda: self.reset_game(WHITE))
        self.new_white_btn.pack(side=tk.LEFT, padx=2)
        
        self.board = GoBoard()
        self.human_color = BLACK
        self.history = [] # Stack of (board_copy, next_player, ko_check_state, last_move)
        
        self.game_over = False
        self.last_move = None
        self.ko_check_state = None # Stores board state before opponent's move
        self.is_ai_thinking = False # Lock for UI
        
        # Load Model
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
            
        self.model = None
        self.model_path = model_path
        self.load_ai_model()
        
        self.canvas.bind("<Button-1>", self.on_click)
        self.draw_board()
        
    def load_ai_model(self):
        if not os.path.exists(self.model_path):
            messagebox.showerror("Error", f"Model file not found: {self.model_path}\nPlease train the large model first (python src_large/train.py).")
            self.root.destroy()
            return
            
        self.status_label.config(text="Loading Large AI...")
        self.root.update()
        
        try:
            self.model = load_model(self.model_path, self.device)
            self.status_label.config(text="Large AI Ready")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            self.root.destroy()

    def reset_game(self, human_color=BLACK):
        if self.is_ai_thinking:
            return
            
        self.board.reset()
        self.human_color = human_color
        self.history = []
        self.game_over = False
        self.last_move = None
        self.ko_check_state = None
        self.update_ui()
        self.status_label.config(text="New Game Started")
        
        if self.human_color == WHITE:
            self.status_label.config(text="AI (Black) is thinking...")
            self.root.after(500, self.ai_move)
            
    def save_state(self):
        # Deep copy necessary parts
        state = (
            self.board.board.copy(),
            self.board.next_player,
            self.ko_check_state.copy() if self.ko_check_state is not None else None,
            self.last_move
        )
        self.history.append(state)
        
    def undo_move(self):
        if self.is_ai_thinking:
            return

        if self.game_over:
            pass
            
        if len(self.history) < 2:
            self.status_label.config(text="Cannot undo (Not enough moves)")
            return
            
        # Revert 2 moves (AI + Human)
        self.history.pop() # Discard last state (before AI move)
        state = self.history.pop() # Get state before Human move
        
        self.restore_state(state)
        self.game_over = False # Reset game over if we undid
        self.status_label.config(text="Undid 2 moves")
        self.update_ui()
        
    def restore_state(self, state):
        board_arr, next_player, ko_state, last_move = state
        self.board.board = board_arr
        self.board.next_player = next_player
        self.ko_check_state = ko_state
        self.last_move = last_move
        
    def draw_board(self):
        self.canvas.delete("all")
        
        # Draw grid
        for i in range(self.board_size):
            # Horizontal
            start_x = self.margin
            end_x = self.margin + (self.board_size - 1) * self.cell_size
            y = self.margin + i * self.cell_size
            self.canvas.create_line(start_x, y, end_x, y)
            
            # Vertical
            start_y = self.margin
            end_y = self.margin + (self.board_size - 1) * self.cell_size
            x = self.margin + i * self.cell_size
            self.canvas.create_line(x, start_y, x, end_y)
            
        # Draw star points (hoshi)
        star_points = [3, 9, 15]
        for r in star_points:
            for c in star_points:
                cx = self.margin + c * self.cell_size
                cy = self.margin + r * self.cell_size
                r_dot = 2
                self.canvas.create_oval(cx-r_dot, cy-r_dot, cx+r_dot, cy+r_dot, fill="black")

        # Draw stones
        for r in range(self.board_size):
            for c in range(self.board_size):
                color = self.board.board[r, c]
                if color != EMPTY:
                    self.draw_stone(r, c, color)
                    
        # Draw last move marker
        if self.last_move:
            r, c = self.last_move
            cx = self.margin + c * self.cell_size
            cy = self.margin + r * self.cell_size
            # Small red circle
            self.canvas.create_oval(cx-3, cy-3, cx+3, cy+3, fill="red", outline="red")

    def draw_stone(self, r, c, color):
        cx = self.margin + c * self.cell_size
        cy = self.margin + r * self.cell_size
        radius = self.cell_size // 2 - 2
        
        fill_color = "black" if color == BLACK else "white"
        outline_color = "black"
        
        self.canvas.create_oval(cx-radius, cy-radius, cx+radius, cy+radius, fill=fill_color, outline=outline_color)

    def update_ui(self):
        self.draw_board()
        
        current_color_name = "Black" if self.board.next_player == BLACK else "White"
        
        if self.board.next_player == self.human_color:
            turn_text = f"{current_color_name} (You)"
        else:
            turn_text = f"{current_color_name} (AI)"
            
        self.turn_label.config(text=f"Turn: {turn_text}")
        self.root.update()

    def on_click(self, event):
        if self.game_over or self.board.next_player != self.human_color or self.is_ai_thinking:
            return
            
        # Convert pixel to grid coords
        c = int(round((event.x - self.margin) / self.cell_size))
        r = int(round((event.y - self.margin) / self.cell_size))
        
        if 0 <= r < self.board_size and 0 <= c < self.board_size:
            # Try play
            if self.board.board[r, c] == EMPTY:
                # Save state before move (for Undo)
                self.save_state()
                
                # Save state before move for Ko checking (for next player)
                prev_state = self.board.board.copy()
                
                res = self.board.play(r, c, self.human_color)
                if res != -1:
                    self.ko_check_state = prev_state
                    self.last_move = (r, c)
                    self.update_ui()
                    # Trigger AI
                    self.root.after(100, self.ai_move)
                else:
                    self.status_label.config(text="Invalid Move!")
                    self.history.pop()

    def human_pass(self):
        if self.game_over or self.board.next_player != self.human_color or self.is_ai_thinking:
            return
        
        self.save_state()
        
        self.board.next_player = WHITE if self.human_color == BLACK else BLACK
        self.status_label.config(text="You passed.")
        self.update_ui()
        self.root.after(500, self.ai_move)

    def ai_move(self):
        if self.game_over or self.is_ai_thinking:
            return
            
        self.is_ai_thinking = True
        self.status_label.config(text="AI Thinking...")
        self.root.update()
        
        ai_color = WHITE if self.human_color == BLACK else BLACK
        
        # Predict
        try:
            moves = predict_move(self.model, self.board, ai_color, self.device)
            
            # Try top moves until valid
            played = False
            ai_stats = "Top Moves (Large):\n"
            
            for i, (move, prob) in enumerate(moves):
                r, c = move
                
                # Check for Ko
                is_ko = False
                if self.ko_check_state is not None:
                    # Simulate on a temp board
                    temp_board = GoBoard(self.board_size)
                    temp_board.board = self.board.board.copy()
                    temp_board.next_player = ai_color
                    res_sim = temp_board.play(r, c, ai_color)
                    if res_sim != -1:
                        if np.array_equal(temp_board.board, self.ko_check_state):
                            is_ko = True
                
                if is_ko:
                    ai_stats += f"({r},{c}): {prob:.1%} [Ko]\n"
                    continue

                ai_stats += f"({r},{c}): {prob:.1%}\n"
                
                if not played:
                    # Save state before playing (Undo)
                    self.save_state()
                    
                    # Save state before playing (Ko)
                    prev_state = self.board.board.copy()
                    
                    res = self.board.play(r, c, ai_color)
                    if res != -1:
                        self.ko_check_state = prev_state
                        self.last_move = (r, c)
                        played = True
                    else:
                        self.history.pop()
                        
            self.ai_info_label.config(text=ai_stats)
            
            if played:
                self.status_label.config(text="AI Played")
            else:
                # AI Pass or Resign
                self.status_label.config(text="AI Passed (No valid moves)")
                self.board.next_player = self.human_color
                
        except Exception as e:
            print(f"AI Error: {e}")
            self.status_label.config(text="AI Error")
            
        self.is_ai_thinking = False
        self.update_ui()

if __name__ == "__main__":
    root = tk.Tk()
    # Check for best_model_large.pth
    model_path = "best_model_large.pth"
    if not os.path.exists(model_path):
        # Check if fallback exists
        if os.path.exists("best_model.pth"):
             print("Warning: Large model not found, falling back to small model for testing GUI logic.")
             # model_path = "best_model.pth" # Optional: fallback
        pass
        
    gui = GoGui(root, model_path)
    root.mainloop()
