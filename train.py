import os
import torch
import torch.optim as optim
from pytorch_msssim import ms_ssim
from dataset import get_dataloaders
from model import NAFNetRCANPipeline

class CharbonnierLoss(torch.nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Load paired dataset from folders
    train_loader, val_loader = get_dataloaders(
        lr_dir="LRnoise", 
        gt_dir="Ground_Truth", 
        batch_size=8, 
        split_ratio=0.85
    )

    model = NAFNetRCANPipeline(in_channels=1, num_features=64, scale_factor=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    charbonnier = CharbonnierLoss().to(device)
    os.makedirs("saved_models", exist_ok=True)
    best_loss = float('inf')

    epochs = 50
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for lr, gt in train_loader:
            lr, gt = lr.to(device), gt.to(device)
            optimizer.zero_grad()
            
            output = model(lr)
            
            # Loss: 80% Charbonnier (pixel fidelity) + 20% Structural Dissimilarity
            loss_pixel = charbonnier(output, gt)
            loss_ssim = 1 - ms_ssim(output, gt, data_range=1.0, size_average=True)
            loss = loss_pixel + (0.2 * loss_ssim)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)

        # Validation Check
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lr_val, gt_val in val_loader:
                lr_val, gt_val = lr_val.to(device), gt_val.to(device)
                pred_val = model(lr_val)
                val_loss += charbonnier(pred_val, gt_val).item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "saved_models/best_nafnet_rcan.pt")

    print("Training finished. Best model saved to saved_models/best_nafnet_rcan.pt")

if __name__ == "__main__":
    train()