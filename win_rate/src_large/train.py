import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Add src_small to path to reuse dataset
sys.path.append(os.path.join(os.path.dirname(__file__), '../src_small'))
from dataset import GoDataset

from model import ValueNetLarge

def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss
    }, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None):
    if not os.path.exists(path):
        return None
    
    print(f"Loading checkpoint from {path}...")
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    return checkpoint['epoch']

def main():
    parser = argparse.ArgumentParser(description='Train TinyGo Large Model')
    parser.add_argument('--data_dir', type=str, default='../processed_data', help='Path to NPZ data')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto, cpu, cuda, mps)')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--max_samples', type=int, default=5000000, help='Max number of samples to load (default: 5M)')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps') # Mac Metal
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
        
    print(f"Using device: {device}")
    
    # Dataset
    full_dataset = GoDataset(args.data_dir, augment=True, max_samples=args.max_samples)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # Model - Using Large Model
    model = ValueNetLarge(num_blocks=10, channels=128).to(device)
    
    # Optimization
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # verbose argument is deprecated/removed in newer PyTorch versions for ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    criterion = nn.MSELoss() # Predicting Win Rate (Regression)
    
    # Checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    
    best_path = 'best_model_large.pth'
    last_path = 'last_checkpoint_large.pth'
    
    # Resume
    if os.path.exists(last_path):
        ep = load_checkpoint(last_path, model, optimizer, scheduler)
        if ep is not None:
            start_epoch = ep + 1
            print(f"Resumed from epoch {start_epoch}")
            
    if os.path.exists(best_path):
        # Just to get best loss
        checkpoint = torch.load(best_path, map_location='cpu')
        best_val_loss = checkpoint.get('loss', float('inf'))
        print(f"Current Best Val Loss: {best_val_loss:.6f}")

    # Training Loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for boards, targets in loop:
            boards, targets = boards.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(boards)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            mae = torch.mean(torch.abs(outputs - targets)).item()
            train_loss += loss.item()
            train_mae += mae
            loop.set_postfix(loss=loss.item(), mae=mae)
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_mae = train_mae / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for boards, targets in loop:
                boards, targets = boards.to(device), targets.to(device)
                outputs = model(boards)
                loss = criterion(outputs, targets)
                mae = torch.mean(torch.abs(outputs - targets)).item()
                val_loss += loss.item()
                val_mae += mae
                loop.set_postfix(loss=loss.item(), mae=mae)
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_mae = val_mae / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.6f}, Train MAE={avg_train_mae:.4f} | Val Loss={avg_val_loss:.6f}, Val MAE={avg_val_mae:.4f}")
        
        # Scheduler
        scheduler.step(avg_val_loss)
        
        # Save Last
        save_checkpoint(model, optimizer, scheduler, epoch, avg_val_loss, last_path)
        
        # Save Best
        if avg_val_loss < best_val_loss:
            print(f"New Best Model! Loss: {avg_val_loss:.6f}")
            best_val_loss = avg_val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, best_path)

if __name__ == "__main__":
    main()
