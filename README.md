# 🛰️ Lunar PSR Terminal
**Deep Learning & 3D Telemetry Pipeline for Permanently Shadowed Regions (PSRs)**

## 📖 Overview
The Lunar South Pole is the primary target for modern space exploration due to its Permanently Shadowed Regions (PSRs), which contain vast reservoirs of water ice. However, operating in these areas presents severe optical and spatial challenges. Raw optical telemetry from lunar satellites relies on faint secondary scattered sunlight, resulting in extreme low-light conditions, near-zero signal-to-noise ratios (SNR), and heavy sensor noise. 

**Lunar PSR Terminal** is an integrated, full-stack software pipeline that unifies artificial intelligence image restoration, automated topographical hazard analysis, and interactive 3D touchdown verification into a single, continuous workflow. 

## ✨ Key Features
* **AI-Driven Image Restoration:** Utilizes a custom PyTorch U-Net architecture to denoise and upscale underexposed orbital telemetry. It uses an overlapping patch-based inference engine (256x256, stride 128) with 2D linear blending to eliminate spatial seam artifacts.
* **Automated Hazard Detection:** Employs OpenCV algorithms (Canny edge detection and mathematical danger dilation) to map steep slopes and jagged boulders, isolating the optimal, continuous flat plain as a safe Landing Zone (LZ).
* **Live Telemetry Streaming:** The Python backend calculates the exact Landing Zone (X,Y) coordinates and injects this data directly into custom HTTP response headers (`X-Landing-Telemetry`).
* **3D Touchdown Simulation:** A React and Three.js frontend maps the 2D pixel coordinates into a dynamic 3D elevation displacement mesh, animating a procedural 3D lander module descending precisely onto the calculated LZ.
* **Mission Video Export:** An integrated `MediaRecorder` pipeline captures the WebGL simulation directly within the browser, exporting a high-quality `.webm` video for mission analysis.

## 🛠️ Technology Stack
**Frontend & 3D Visualization:**
* React.js / Next.js
* Tailwind CSS & Framer Motion
* Three.js, `@react-three/fiber`, `@react-three/drei`

**Backend & Telemetry:**
* Python & FastAPI
* Uvicorn (ASGI server)

**Artificial Intelligence & Computer Vision:**
* PyTorch & Torchvision (U-Net Architecture)
* OpenCV (cv2)
* NumPy

## 📊 Datasets (ISRO & NASA)
This project is designed to process high-resolution lunar orbital data. You can download raw and calibrated satellite imagery from the following official portals:

1. **ISRO Chandrayaan-2 Datasets:**
   * **Source:** Indian Space Science Data Centre (ISSDC) PRADAN Portal
   * **Instruments:** Terrain Mapping Camera-2 (TMC-2) and Orbiter High-Resolution Camera (OHRC)
   * **Link:** [ISSDC PRADAN Portal](https://pradan.issdc.gov.in/) *(Requires free registration to download payload data)*
2. **NASA Lunar Reconnaissance Orbiter (LRO):**
   * **Source:** Planetary Data System (PDS) Cartography and Imaging Sciences Node
   * **Instruments:** Lunar Reconnaissance Orbiter Camera (LROC) Narrow Angle Camera (NAC)
   * **Link:** [LROC Image Search](http://wms.lroc.asu.edu/lroc/search)

## 🚀 Installation & Setup

### 1. Backend (FastAPI & PyTorch)
Navigate to the backend directory and set up the Python environment:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
