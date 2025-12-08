import os
import json
import numpy as np
import glob
import sys
from tqdm import tqdm

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src_extra_large.board import GoBoard, BLACK, WHITE, SIZE

def process_game(file_path):
    """
    Process a single game file.
    Returns a list of (board_state, player, target) tuples.
    """
    try:
        with open(file_path, 'r') as f:
            moves = json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
            
    board = GoBoard()
    game_samples = []
    
    for move in moves:
        # move format: {"B": [x, y]} or {"W": [x, y]}
        if not isinstance(move, dict) or len(move) != 1:
            continue
            
        color_str = list(move.keys())[0]
        coords = move[color_str]
        
        # Check for pass or invalid coords
        # Assuming coords are [1-19, 1-19]
        # If pass, coords might be special? Let's assume valid moves for now.
        # If coords is empty or invalid, skip?
        if not isinstance(coords, list) or len(coords) != 2:
            continue

        x, y = coords[0] - 1, coords[1] - 1 # Convert 1-19 to 0-18
        
        # Validate coordinates
        if not (0 <= x < SIZE and 0 <= y < SIZE):
            continue
            
        color = BLACK if color_str == 'B' else WHITE
        
        # Force correct turn if data is out of sync (rare in good datasets)
        if color != board.next_player:
            board.next_player = color
            
        # Record sample: (Board State, Player to move, Target Action)
        # We store compact data to save RAM/Disk
        game_samples.append((
            board.board.copy(), # (19, 19) int8
            np.int8(color),     # int8
            np.int16(x * SIZE + y) # int16 flat index
        ))
        
        # Apply move
        res = board.play(x, y, color)
        if res == -1:
            # Invalid move in dataset? Stop processing this game or skip.
            # Usually stop to avoid divergence.
            break
            
    return game_samples

def preprocess_data(data_dir, output_file, limit=None):
    """
    Process all .data files in data_dir and save to a single .npz file (or multiple).
    For this example, we save to one .npz.
    """
    files = glob.glob(os.path.join(data_dir, "*.data"))
    if limit:
        files = files[:limit]
    
    all_boards = []
    all_players = []
    all_targets = []
    
    print(f"Processing {len(files)} files...")
    
    for f in tqdm(files):
        samples = process_game(f)
        for b, p, t in samples:
            all_boards.append(b)
            all_players.append(p)
            all_targets.append(t)
            
    # Stack and save
    print("Stacking data...")
    boards = np.stack(all_boards)
    players = np.array(all_players)
    targets = np.array(all_targets)
    
    print(f"Saving {len(boards)} samples to {output_file}...")
    np.savez_compressed(output_file, boards=boards, players=players, targets=targets)
    print("Done.")

if __name__ == "__main__":
    # Example usage: Process local example_data.data
    # In real scenario, point to Training_data folder
    
    # Check if Training_data exists, otherwise use current dir for example
    if os.path.exists("Training_data"):
        data_dir = "Training_data"
        out_path = "processed_data.npz"
        limit = None # Process all files
    else:
        # For the user's specific request, they mentioned Training_data exists.
        # But in this env, I only saw example_data.data in root.
        # Let's just process the current directory's .data files.
        data_dir = "." 
        out_path = "example_processed.npz"
        limit = 100 # Default limit for example data
        
    preprocess_data(data_dir, out_path, limit=limit)
