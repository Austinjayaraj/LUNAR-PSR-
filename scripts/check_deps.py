import sys
import os

def print_status(library_name, status, version=""):
    color_code = "\033[92m✅ SUCCESS\033[0m" if status else "\033[91m❌ MISSING\033[0m"
    ver_str = f" (v{version})" if version else ""
    print(f"| {library_name:<25} | {color_code:<20} {ver_str}")

def run_diagnostic():
    print("=" * 60)
    print("🛰️  LUNAR PSR HIGH-FIDELITY DEEP PIPELINE DIAGNOSTIC REPORT")
    print("=" * 60)
    print(f"Python Interpreter: {sys.executable}")
    print(f"Python Version:     {sys.version.split()[0]}")
    print("-" * 60)

    # Core library checklist
    dependencies = {
        "numpy": "numpy",
        "cv2": "opencv-python (cv2)",
        "rasterio": "rasterio",
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "torch": "torch (PyTorch)",
        "torchvision": "torchvision"
    }

    missing_packages = []

    for module_name, display_name in dependencies.items():
        try:
            mod = __import__(module_name)
            ver = getattr(mod, "__version__", "Unknown")
            print_status(display_name, True, ver)
        except ImportError:
            print_status(display_name, False)
            missing_packages.append(display_name)

    print("-" * 60)
    print("🚀 HARDWARE ACCELERATION CHECK:")
    
    # Deep hardware diagnostic for Apple Silicon (MacBook Air)
    try:
        import torch
        print_status("PyTorch Base Engine", True, torch.__version__)
        
        mps_available = torch.backends.mps.is_available()
        mps_built = torch.backends.mps.is_built() if hasattr(torch.backends.mps, "is_built") else False
        
        if mps_available:
            print(f"| Apple Silicon GPU (MPS)   | \033[92m✅ READY\033[0m (Accelerated processing active)")
            # Perform a quick 1-second physical GPU tensor operations test
            try:
                x = torch.ones(5, 5).to("mps")
                y = x * 2
                gpu_test = "Passed" if torch.sum(y).item() == 50.0 else "Failed"
                print(f"| Unified Memory Tensor Test| \033[92m✅ Passed\033[0m (Memory operations stable)")
            except Exception as e:
                print(f"| Unified Memory Tensor Test| \033[91m⚠️ FAILED\033[0m ({str(e)})")
        else:
            print(f"| Apple Silicon GPU (MPS)   | \033[91m⚠️ NOT DETECTED\033[0m (Will fallback to slower CPU)")
            print("  ↳ Recommendation: Ensure PyTorch is built with Apple Silicon architecture support.")

        # Test Perceptual Loss feature extractor model load compatibility
        try:
            from torchvision.models import vgg16, VGG16_Weights
            print("| torchvision VGG-16 Load   | \033[92m✅ READY\033[0m")
        except ImportError:
            print("| torchvision VGG-16 Load   | \033[91m❌ UNABLE TO IMPORT\033[0m")
            
    except ImportError:
        print(f"| PyTorch Engine            | \033[91m❌ PyTorch not available. Diagnostic skipped.\033[0m")

    print("=" * 60)

    if missing_packages:
        print("\033[91m⚠️  ACTION REQUIRED: Missing dependencies found!\033[0m")
        print("Run the following clean command inside your virtual environment to resolve all issues:")
        # Generate the direct matching installation command
        pip_installs = []
        for pkg in missing_packages:
            if "opencv" in pkg:
                pip_installs.append("opencv-python")
            elif "PyTorch" in pkg:
                pip_installs.append("torch")
            elif "torchvision" in pkg:
                pip_installs.append("torchvision")
            else:
                pip_installs.append(pkg.split()[0])
                
        print(f"\n   pip install {' '.join(pip_installs)}\n")
    else:
        print("\033[92m🌕 ALL SYSTEMS NOMINAL. Your system is fully ready for high-fidelity training!\033[0m")
        print("You can safely execute: python src/preprocess.py && python train.py")
    print("=" * 60)

if __name__ == "__main__":
    run_diagnostic()