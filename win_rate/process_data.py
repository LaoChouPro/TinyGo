import os
import re
import glob
import numpy as np
import gc

# ---------------------------------------------------------
# 1. NPZ Structure Design
# ---------------------------------------------------------
# Each NPZ chunk will contain:
#   'features': int8 array of shape [N, 19, 19]
#       1  = Current Player's Stone (Next to move)
#       -1 = Opponent's Stone
#       0  = Empty
#   'winrates': float32 array of shape [N]
#       Value in [0, 1] representing Win Rate for Current Player.
# 
# Design Considerations:
#   - Data Bias: Using per-move evaluations (distillation) rather than game outcome reduces variance.
#   - Input: Relative to "Next Player". If next is Black, Black stones=1.
#   - Captures: Must simulate board state correctly using a Go Engine logic.
#   - Memory: Process in chunks to avoid OOM.

# ---------------------------------------------------------
# 2. Go Board Logic (Lightweight)
# ---------------------------------------------------------
class GoBoard:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8) # 0=Empty, 1=Black, -1=White
        self.ko_point = None

    def play(self, row, col, color):
        """
        color: 1 (Black) or -1 (White)
        Returns: True if move valid (always true for SGF replay usually), False otherwise
        """
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            return False # Pass or invalid
        
        # Place stone
        self.board[row, col] = color
        
        # Check captures of opponent
        opponent = -color
        captured_stones = []
        
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                if self.board[nr, nc] == opponent:
                    group, liberties = self.get_group_liberties(nr, nc)
                    if liberties == 0:
                        captured_stones.extend(group)
        
        # Remove captured
        for r, c in captured_stones:
            self.board[r, c] = 0
            
        # Check self suicide (should not happen in proper SGF, but standard rule)
        # In Chinese rules (KataGo), suicide is sometimes allowed or handled, 
        # but usually valid moves don't suicide.
        # We skip checking self-liberties for speed as SGFs are trusted.
        
        return True

    def get_group_liberties(self, r, c):
        color = self.board[r, c]
        if color == 0: return [], 0
        
        stack = [(r, c)]
        visited = {(r, c)}
        group = [(r, c)]
        liberties = set()
        
        while stack:
            curr_r, curr_c = stack.pop()
            
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    n_color = self.board[nr, nc]
                    if n_color == 0:
                        liberties.add((nr, nc))
                    elif n_color == color and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        stack.append((nr, nc))
                        group.append((nr, nc))
                        
        return group, len(liberties)

    def get_features(self, next_player_color):
        # next_player_color: 1 (Black) or -1 (White)
        # We want: 1 for Own stones, -1 for Opponent stones
        # Board is: 1 for Black, -1 for White
        
        # If Next=Black (1):
        #   Feature = Board * 1 -> Black=1, White=-1. Correct.
        # If Next=White (-1):
        #   Feature = Board * -1 -> Black=-1, White=1. Correct.
        
        return self.board * next_player_color

# ---------------------------------------------------------
# 3. Processing Script
# ---------------------------------------------------------

SGF_COORD = "abcdefghijklmnopqrs"
COORD_MAP = {c: i for i, c in enumerate(SGF_COORD)}

def parse_move(move_str):
    if len(move_str) != 2: return None # Pass or invalid
    r = COORD_MAP.get(move_str[1]) # SGF is col,row usually? 
    # SGF standard: "aa" is top-left (0,0). First char col, second char row.
    # Wait, usually SGF is [column][row].
    # But for a symmetric board it doesn't matter as long as consistent.
    # Let's stick to: First char = Col (x), Second char = Row (y).
    # In numpy: board[row, col] -> board[y, x].
    c = COORD_MAP.get(move_str[0])
    return (r, c) # Return (row, col)

def process_sgf(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        return []

    # Regex to find nodes
    # We look for ;B[..]C[..] or ;W[..]C[..]
    # Also Root node might have C[...] but usually no move.
    
    # Let's iterate linearly finding ";"
    # This is a simple parser.
    
    nodes = re.split(r';', content)
    
    game_data = []
    board = GoBoard(19)
    
    # Skip preamble (nodes[0] is usually empty or file header)
    
    for node in nodes:
        if not node.strip(): continue
        
        # Check for Move
        move_match = re.search(r'([BW])\[([a-z]*)\]', node)
        
        # Check for Comment (Winrate)
        comment_match = re.search(r'C\[\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)', node)
        
        current_move_color = 0 # 0=None, 1=B, -1=W
        is_pass = False
        
        if move_match:
            color_str = move_match.group(1)
            coord_str = move_match.group(2)
            current_move_color = 1 if color_str == 'B' else -1
            
            if len(coord_str) == 2:
                r, c = parse_move(coord_str)
                if r is not None and c is not None:
                    board.play(r, c, current_move_color)
            else:
                is_pass = True
        
        # Extract Data for Training
        # We need data *after* this move is made.
        # The comment contains evaluation of the board state *after* the move.
        # Who is the Next Player?
        # If current move was Black, next is White.
        # If current move was White, next is Black.
        # If no move (root node setup), usually check setup properties, but let's skip root for simplicity unless it has clear turn info.
        
        if move_match and comment_match:
            # Comment stats: [WhiteWin, BlackWin, Draw]
            stats = [float(x) for x in comment_match.groups()]
            w_win, b_win, _ = stats
            
            next_player = -current_move_color # The player who will play NEXT
            
            # Label: Winrate of Next Player
            label = 0.0
            if next_player == 1: # Next is Black
                label = b_win
            else: # Next is White
                label = w_win
            
            # Feature: Board from perspective of Next Player
            feature = board.get_features(next_player)
            
            game_data.append({
                'feature': feature.copy(),
                'label': label
            })
            
    return game_data

def main():
    data_dir = '/Users/laochou/Desktop/编程/项目/TinyGo-data/data'
    output_dir = '/Users/laochou/Desktop/编程/项目/TinyGo-data/processed_data'
    os.makedirs(output_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(data_dir, '*.sgf'))
    print(f"Found {len(files)} SGF files.")
    
    CHUNK_SIZE = 5000 # Games per chunk
    current_chunk_data = {'features': [], 'labels': []}
    chunk_idx = 0
    
    total_samples = 0
    
    for i, fpath in enumerate(files):
        samples = process_sgf(fpath)
        if not samples: continue
        
        for s in samples:
            current_chunk_data['features'].append(s['feature'])
            current_chunk_data['labels'].append(s['label'])
            
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1} files... (Current chunk: {len(current_chunk_data['labels'])} samples)")
            
        # Save Chunk
        if (i + 1) % CHUNK_SIZE == 0 or (i + 1) == len(files):
            if current_chunk_data['labels']:
                # Convert to numpy
                feats = np.array(current_chunk_data['features'], dtype=np.int8)
                lbls = np.array(current_chunk_data['labels'], dtype=np.float32)
                
                save_path = os.path.join(output_dir, f'data_chunk_{chunk_idx:03d}.npz')
                print(f"Saving chunk {chunk_idx} to {save_path}...")
                print(f"Shape: {feats.shape}, {lbls.shape}")
                
                np.savez_compressed(save_path, features=feats, winrates=lbls)
                
                total_samples += len(lbls)
                chunk_idx += 1
                
                # Reset
                del feats, lbls
                current_chunk_data = {'features': [], 'labels': []}
                gc.collect()

    print(f"Done. Total samples generated: {total_samples}")

if __name__ == "__main__":
    main()
