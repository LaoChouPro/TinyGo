import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class GoDataset(Dataset):
    def __init__(self, data_dir, augment=True, max_samples=None):
        self.files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
        self.augment = augment
        
        # We load one chunk at a time ideally, but PyTorch Dataset usually expects random access to all.
        # For huge datasets, we usually map indices to files.
        # But here we have chunks of ~5000 games.
        # Let's verify size.
        # If dataset is HUGE, we should use IterableDataset or a sophisticated indexer.
        # For simplicity in this script, we will load ALL chunks into memory if RAM allows,
        # OR implement a lazy loading mechanism.
        
        # Given 150k games * 200 moves = 30M positions.
        # 30M * 19*19 bytes ~ 10GB. Might be tight for 16GB RAM.
        # Let's use a memory-mapped approach or index-based lazy loading.
        
        # Let's scan files to build an index: [ (file_idx, inner_idx), ... ]
        # This is slow to build on startup if we open every npz.
        # Alternative: Just load a subset of chunks for training per epoch?
        # Or standard approach: Just list files and let DataLoader load on fly? No, too slow.
        
        # "Memory Mapped" NPZ is tricky because compressed.
        # Best approach for this scale on a single machine:
        # Load all data into RAM if possible (32GB+ RAM).
        # If not, use a subset.
        
        # Let's assume the user wants to train on whatever is available in 'processed_data'.
        # We will implement a list of loaded arrays.
        
        print(f"Found {len(self.files)} chunk files.")
        self.data_chunks = []
        total_samples = 0
        
        # Limit to first N chunks to avoid OOM for now?
        # Or try to load all and catch OOM.
        # If max_samples is None, use a safe default or load all (careful).
        # Let's set a default safe limit if not provided, or respect user input.
        limit = max_samples if max_samples is not None else 5000000
        
        for f in self.files:
            try:
                # Load with mmap_mode is not supported for compressed npz.
                # Just load.
                with np.load(f) as data:
                    feats = data['features'] # [N, 19, 19] int8
                    lbls = data['winrates']  # [N] float32
                    
                    # Check if adding this chunk exceeds limit
                    if total_samples + len(lbls) > limit:
                        remaining = limit - total_samples
                        if remaining > 0:
                            self.data_chunks.append((feats[:remaining], lbls[:remaining]))
                            total_samples += remaining
                        print(f"Reached sample limit: {limit} samples.")
                        break
                    
                    self.data_chunks.append((feats, lbls))
                    total_samples += len(lbls)
                    
                    if total_samples >= limit:
                        print(f"Reached sample limit: {limit} samples.")
                        break
            except Exception as e:
                print(f"Error loading {f}: {e}")
                
        self.total_samples = total_samples
        print(f"Loaded {total_samples} samples into RAM.")
        
        # Build cumulative index
        self.chunk_indices = []
        curr = 0
        for feats, _ in self.data_chunks:
            self.chunk_indices.append(curr)
            curr += len(feats)
            
    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Binary search or simple iteration to find chunk
        # Since we have few chunks, iteration is fast enough.
        
        chunk_idx = 0
        for i, start_idx in enumerate(self.chunk_indices):
            if idx >= start_idx:
                chunk_idx = i
            else:
                break
                
        local_idx = idx - self.chunk_indices[chunk_idx]
        feats, lbls = self.data_chunks[chunk_idx]
        
        board = feats[local_idx] # [19, 19] int8
        winrate = lbls[local_idx]
        
        # Data Augmentation
        if self.augment:
            # Random Rotate 0, 90, 180, 270
            rot = np.random.randint(4)
            board = np.rot90(board, k=rot)
            
            # Random Flip
            if np.random.random() > 0.5:
                board = np.flipud(board)
                
        # Convert to Tensor
        # Input to model: [2, 19, 19]
        # Plane 0: My stones (1 -> 1, -1 -> 0)
        # Plane 1: Opponent stones (-1 -> 1, 1 -> 0)
        
        input_tensor = np.zeros((2, 19, 19), dtype=np.float32)
        input_tensor[0] = (board == 1).astype(np.float32)
        input_tensor[1] = (board == -1).astype(np.float32)
        
        return torch.from_numpy(input_tensor), torch.tensor([winrate], dtype=torch.float32)
