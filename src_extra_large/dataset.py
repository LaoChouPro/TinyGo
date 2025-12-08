import numpy as np
import torch
from torch.utils.data import Dataset

SIZE = 19

class GoDataset(Dataset):
    def __init__(self, data_path, augment=False, limit=0):
        # Load data fully into memory to avoid Pickle errors with mmap on Windows
        print(f"Loading data from {data_path} (Memory mode)...")
        # Use a context manager to ensure the file handle is closed immediately
        with np.load(data_path) as data:
            total_samples = len(data['boards'])
            
            if limit > 0 and limit < total_samples:
                print(f"Randomly sampling {limit} samples from {total_samples} total samples...")
                indices = np.sort(np.random.choice(total_samples, limit, replace=False))
                
                # Copy only the sampled data to memory
                self.boards = data['boards'][indices]
                self.players = data['players'][indices]
                self.targets = data['targets'][indices]
            else:
                # Copy all data to memory
                self.boards = data['boards'][:]
                self.players = data['players'][:]
                self.targets = data['targets'][:]
        
        self.augment = augment # Kept for compatibility, but logic moved to GPU
        print(f"Dataset ready with {len(self.boards)} samples (RAM).")
        
    def __len__(self):
        return len(self.boards)
    
    def __getitem__(self, idx):
        # Direct access (no index mapping needed anymore as we pre-filtered)
        board = self.boards[idx] # (19, 19)
        player = self.players[idx] # scalar
        target = self.targets[idx] # scalar
        
        # NOTE: Augmentation moved to GPU in train.py for performance
        # No more CPU-side rotation/flip here!
        
        # Construct features
        # Channel 0: Player stones
        # Channel 1: Opponent stones
        # Channel 2: Bias (ones)
        
        feat = np.zeros((3, SIZE, SIZE), dtype=np.float32)
        
        # player: 1(B) or 2(W)
        # board: 0, 1, 2
        
        my_stones = (board == player)
        opp_stones = (board == (3 - player)) # If player=1, opp=2. If player=2, opp=1.
        
        feat[0] = my_stones.astype(np.float32)
        feat[1] = opp_stones.astype(np.float32)
        feat[2] = 1.0
        
        # Flatten target again (it was scalar)
        # We return it as is, GPU will handle coordinate transform
        
        return torch.from_numpy(feat), torch.tensor(target, dtype=torch.long)
