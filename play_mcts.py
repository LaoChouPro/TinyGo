import sys
import os
import torch
import numpy as np

# Setup paths
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'win_rate', 'src_small'))

# Imports
from src.board import GoBoard, BLACK, WHITE, SIZE
from src_large.model import TinyGoNet
from win_rate.src_small.model import ValueNetSmall
from src_mcts.mcts import MCTS

def load_policy_model(path, device):
    model = TinyGoNet(num_blocks=10, channels=128)
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def load_value_model(path, device):
    model = ValueNetSmall(num_blocks=6, channels=64, input_planes=2)
    checkpoint = torch.load(path, map_location=device)
    # Checkpoint for win_rate might contain 'model_state_dict'
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # It might be the direct state dict or nested
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def main():
    # Device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
        
    print(f"Using device: {device}")
    
    # Paths
    policy_path = os.path.join(project_root, 'best_model_large.pth')
    value_path = os.path.join(project_root, 'win_rate', 'src_small', 'best_model.pth')
    
    if not os.path.exists(policy_path):
        print(f"Policy model not found at {policy_path}")
        return
    if not os.path.exists(value_path):
        print(f"Value model not found at {value_path}")
        return
        
    # Load Models
    print("Loading Policy Model...")
    policy_model = load_policy_model(policy_path, device)
    
    print("Loading Value Model...")
    value_model = load_value_model(value_path, device)
    
    # Init MCTS
    print("Initializing MCTS...")
    mcts = MCTS(policy_model, value_model, device=device, num_simulations=50, c_puct=1.0)
    
    # Test Run
    print("\n=== MCTS Test Run (Empty Board) ===")
    board = GoBoard()
    
    # Run Search
    actions, probs = mcts.run(board, BLACK, temperature=1.0)
    
    # Show Top 5
    print("\nTop 5 Moves:")
    top_indices = np.argsort(probs)[::-1][:5]
    for idx in top_indices:
        r, c = actions[idx]
        p = probs[idx]
        print(f"Move ({r}, {c}): Prob {p:.4f}")
        
    # Pick move
    best_idx = np.argmax(probs)
    r, c = actions[best_idx]
    print(f"\nSelected Move: ({r}, {c})")
    
    # Play a few moves
    print("\n=== Self-Play Simulation (5 moves) ===")
    board.reset()
    curr_color = BLACK
    
    for i in range(5):
        print(f"\nMove {i+1} ({'Black' if curr_color == BLACK else 'White'}):")
        actions, probs = mcts.run(board, curr_color, temperature=1.0)
        
        # Greedy choice for demo
        best_idx = np.argmax(probs)
        r, c = actions[best_idx]
        print(f"MCTS chose: ({r}, {c}) with prob {probs[best_idx]:.4f}")
        
        board.play(r, c, curr_color)
        curr_color = 3 - curr_color # Toggle

if __name__ == "__main__":
    main()
