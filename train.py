import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class EdgeLoss(nn.Module):
    """Penalizes differences in high-frequency (edge) content using a fixed Laplacian kernel.
    Pushes the model to preserve sharp edges without inventing new texture (no learned weights, no hallucination risk)."""
    def __init__(self):
        super().__init__()
        kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
        self.register_buffer("kernel", kernel.view(1, 1, 3, 3))

    def forward(self, pred, target):
        edge_pred = F.conv2d(pred, self.kernel, padding=1)
        edge_target = F.conv2d(target, self.kernel, padding=1)
        return F.l1_loss(edge_pred, edge_target)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

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
    edge_loss = EdgeLoss().to(device)
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

            loss_pixel = charbonnier(output, gt)
            loss_ssim = 1 - ms_ssim(output, gt, data_range=1.0, size_average=True)
            loss_edge = edge_loss(output, gt)
            loss = loss_pixel + (0.5 * loss_ssim) + (0.3 * loss_edge)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)

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