import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PairedNpyDataset(Dataset):
    def __init__(self, lr_dir, gt_dir, filenames, augment=False):
        self.lr_dir = lr_dir
        self.gt_dir = gt_dir
        self.filenames = filenames
        self.augment = augment

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        
        lr_path = os.path.join(self.lr_dir, fname)
        gt_path = os.path.join(self.gt_dir, fname)
        lr_array = np.load(lr_path).astype(np.float32)
        gt_array = np.load(gt_path).astype(np.float32)
        
        # Convert to Tensor (C, H, W)
        lr_tensor = torch.from_numpy(lr_array).unsqueeze(0) if lr_array.ndim == 2 else torch.from_numpy(lr_array)
        gt_tensor = torch.from_numpy(gt_array).unsqueeze(0) if gt_array.ndim == 2 else torch.from_numpy(gt_array)
        if lr_tensor.max() > 1.0:
            lr_tensor = lr_tensor / 255.0
        if gt_tensor.max() > 1.0:
            gt_tensor = gt_tensor / 255.0

        # Augmentations (Rotation & Flips)
        if self.augment:
            if random.random() > 0.5:
                lr_tensor = torch.flip(lr_tensor, dims=[2])
                gt_tensor = torch.flip(gt_tensor, dims=[2])
            if random.random() > 0.5:
                lr_tensor = torch.flip(lr_tensor, dims=[1])
                gt_tensor = torch.flip(gt_tensor, dims=[1])
            if random.random() > 0.5:
                k = random.randint(1, 3)
                lr_tensor = torch.rot90(lr_tensor, k, [1, 2])
                gt_tensor = torch.rot90(gt_tensor, k, [1, 2])

        return lr_tensor, gt_tensor

def get_dataloaders(lr_dir="LRnoise", gt_dir="Ground_Truth", batch_size=8, split_ratio=0.85):
    all_files = sorted([f for f in os.listdir(lr_dir) if f.endswith(".npy")])
    valid_files = [f for f in all_files if os.path.exists(os.path.join(gt_dir, f))]
    
    random.seed(42)
    random.shuffle(valid_files)
    
    split_idx = int(len(valid_files) * split_ratio)
    train_files = valid_files[:split_idx]
    val_files = valid_files[split_idx:]
    
    train_dataset = PairedNpyDataset(lr_dir, gt_dir, train_files, augment=True)
    val_dataset = PairedNpyDataset(lr_dir, gt_dir, val_files, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    return train_loader, val_loader