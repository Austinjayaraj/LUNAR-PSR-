import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class LunarDataset(Dataset):
    def __init__(self, num_samples=300, patch_size=256):
        """Generates highly textured synthetic lunar craters."""
        self.num_samples = num_samples
        self.patch_size = patch_size

    def __len__(self):
        return self.num_samples

    def _generate_synthetic_crater(self):
        size = self.patch_size
        
        # 1. Base Regolith (Moon Dust) instead of flat gray
        # We use Gaussian noise to simulate a rocky, uneven surface
        base_noise = np.random.normal(120, 20, (size, size))
        img = cv2.GaussianBlur(base_noise, (5, 5), 0).astype(np.float32)
        
        # 2. Procedurally generate rugged impact craters
        for _ in range(np.random.randint(4, 10)):
            cx, cy = np.random.randint(0, size, 2)
            radius = np.random.randint(15, 50)
            
            # Create mathematical masks for the crater parts
            y, x = np.ogrid[-cy:size-cy, -cx:size-cx]
            crater_mask = x**2 + y**2 <= radius**2
            rim_mask = (x**2 + y**2 <= (radius * 1.1)**2) & (x**2 + y**2 > (radius * 0.8)**2)
            
            # Apply shadows and highlights while PRESERVING the rocky texture underneath
            img[crater_mask] -= 35  # Darker floor
            img[rim_mask] += 45     # Bright rim (sunlight)
            
        # 3. Add fine, sharp high-frequency regolith texture on top
        # This teaches the AI not to over-smooth the final image!
        fine_dust = np.random.normal(0, 8, (size, size))
        img = np.clip(img + fine_dust, 0, 255).astype(np.uint8)
        
        return img

    def _degrade_image(self, image):
        # Downscale severely to simulate bad optics
        low_res = cv2.resize(image, (self.patch_size // 4, self.patch_size // 4), interpolation=cv2.INTER_LINEAR)
        # Inject chunky digital sensor noise
        noise = np.random.normal(0, 20, low_res.shape).astype(np.int16)
        low_res = np.clip(low_res + noise, 0, 255).astype(np.uint8)
        # Upscale back to original tensor size (creates the blurry/blocky look)
        low_res = cv2.resize(low_res, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)
        return low_res

    def __getitem__(self, idx):
        # The Ground Truth now has sharp rocks and dust
        gt_image = self._generate_synthetic_crater()
        # The Input is blurry and artificially noisy
        lr_image = self._degrade_image(gt_image)

        # Convert to Tensors
        lr_tensor = torch.from_numpy(lr_image).unsqueeze(0).float() / 255.0
        gt_tensor = torch.from_numpy(gt_image).unsqueeze(0).float() / 255.0

        return lr_tensor, gt_tensor