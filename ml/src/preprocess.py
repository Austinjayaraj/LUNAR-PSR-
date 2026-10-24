import os
import rasterio
import numpy as np
import cv2

def extract_from_ohrc(ohrc_path, output_dir, patch_size=256, max_patches=1500):
    """
    Slices the high-res OHRC GeoTIFF and mathematically degrades it to simulate TMC-2.
    """
    hr_dir = os.path.join(output_dir, "hr")
    lr_dir = os.path.join(output_dir, "lr")
    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(lr_dir, exist_ok=True)
    
    print(f"🛰️ Opening OHRC Telemetry: {ohrc_path}...")
    
    with rasterio.open(ohrc_path) as src:
        # We don't want to load the whole 1.2GB image into RAM at once.
        # We will read chunks (windows) of it.
        height = src.height
        width = src.width
        
        print(f"📏 Image Dimensions: {width} x {height} pixels.")
        print("⚙️ Extracting and simulating telemetry pairs...")
        
        patch_count = 0
        
        # We step through the massive image, avoiding the very edges
        step = patch_size * 2 
        
        for y in range(0, height - patch_size, step):
            for x in range(0, width - patch_size, step):
                if patch_count >= max_patches:
                    break
                    
                # Read a 256x256 window directly from the file on disk
                window = rasterio.windows.Window(x, y, patch_size, patch_size)
                hr_patch = src.read(1, window=window)
                
                # Skip invalid, empty, or pure black patches (space/deep shadows)
                if hr_patch is None or np.isnan(hr_patch).any() or np.mean(hr_patch) < 15:
                    continue
                    
                # 1. Prepare Ground Truth (HR)
                # Normalize the 16-bit data to 8-bit [0-255]
                hr_patch = cv2.normalize(hr_patch, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                
                # 2. Simulate TMC-2 (LR)
                # TMC-2 has roughly 20x lower resolution (5m vs 0.25m). 
                # We shrink the image down to 1/4th size, add noise, and stretch it back.
                small_size = patch_size // 4
                lr_patch = cv2.resize(hr_patch, (small_size, small_size), interpolation=cv2.INTER_LINEAR)
                
                # Add simulated sensor noise
                noise = np.random.normal(0, 15, lr_patch.shape).astype(np.int16)
                lr_patch = np.clip(lr_patch + noise, 0, 255).astype(np.uint8)
                
                # Upscale back to 256x256 (it will now look blurry and blocky)
                lr_patch = cv2.resize(lr_patch, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
                
                # Save the pair
                cv2.imwrite(os.path.join(hr_dir, f"{patch_count}.png"), hr_patch)
                cv2.imwrite(os.path.join(lr_dir, f"{patch_count}.png"), lr_patch)
                patch_count += 1
                
                if patch_count % 100 == 0:
                    print(f"   ... Extracted {patch_count}/{max_patches} patches")
                    
            if patch_count >= max_patches:
                break
                
    print(f"✅ Extraction Complete! Saved {patch_count} real lunar patches to {output_dir}")

if __name__ == "__main__":
    # Point directly to your clean XML file
    OHRC_FILE = "data/ch2_ohr_ncp_20260102T1224107393_d_img_d18.xml"
    
    if not os.path.exists(OHRC_FILE):
        print(f"❌ Error: Could not find {OHRC_FILE}")
    else:
        extract_from_ohrc(OHRC_FILE, output_dir="data/processed_pairs")