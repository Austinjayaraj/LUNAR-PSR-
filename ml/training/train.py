import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import vgg16, VGG16_Weights
import os

from src.dataset import LunarDataset 
from src.model import LunarUNet 

# --- THE SECRET WEAPON: VGG-16 TEXTURE CRITIC ---
class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Load a pre-trained VGG16 brain
        print("🧠 Downloading VGG-16 Texture Critic...")
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(device)
        
        # We extract the first 16 layers (this is where edge and rock textures are recognized)
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:16]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False # Freeze the critic so it doesn't learn, only judges
            
        self.mse = nn.MSELoss()

    def forward(self, generated, target):
        # VGG expects 3 color channels (RGB), but our moon is 1 channel (Grayscale).
        # We duplicate the grayscale channel 3 times to trick VGG.
        gen_3c = generated.repeat(1, 3, 1, 1)
        target_3c = target.repeat(1, 3, 1, 1)

        # Extract the deep textures!
        gen_features = self.feature_extractor(gen_3c)
        target_features = self.feature_extractor(target_3c)

        # Force the AI to match the TEXTURES, not just the pixels
        return self.mse(gen_features, target_features)

def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Hardware Acceleration: {device}")

    # Load your extracted real-world OHRC/TMC-2 patches
    dataset = LunarDataset()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = LunarUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    # --- THE DUAL-LOSS SYSTEM ---
    mse_loss = nn.MSELoss()
    perceptual_loss = PerceptualLoss(device)

    epochs = 30
    print("\n--- Starting Deep Perceptual Training ---")
    for epoch in range(epochs):
        total_loss = 0
        for lr_imgs, hr_imgs in dataloader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            optimizer.zero_grad()
            outputs = model(lr_imgs)

            # 1. Pixel Loss (Keeps the lighting and shadows accurate)
            loss_pixel = mse_loss(outputs, hr_imgs)
            
            # 2. Texture Loss (Forces the AI to draw sharp, gritty rocks)
            loss_texture = perceptual_loss(outputs, hr_imgs)
            
            # Combine them: 30% Lighting, 70% Sharp Rocks
            loss = (0.3 * loss_pixel) + (0.7 * loss_texture)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] | Texture Loss: {total_loss/len(dataloader):.4f}")

    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/lunar_unet_best.pth")
    print("✅ Training Complete! Photorealistic AI brain successfully saved.")

if __name__ == "__main__":
    train()