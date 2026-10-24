# Lunar PSR Datasets

This directory contains lunar surface imagery used for model training, validation, and testing.

## Directory Structure
- `raw/`: Calibrated raw lunar imagery from ISRO Chandrayaan-2 Orbiter High Resolution Camera (OHRC):
  - `ch2_ohr_ncp_20260102T1224107393_d_img_d18.img`: 12,000 x 50,538 16-bit PDS4 raster image (~1.21 GB).
  - `ch2_ohr_ncp_20260102T1224107393_d_img_d18.xml`: PDS4 XML label and calibration metadata.
- `processed_pairs/`: Preprocessed high-resolution and synthetic low-resolution pairs (256x256):
  - `hr/`: 1,500 ground-truth normalized high-resolution PNG patches.
  - `lr/`: 1,500 degraded low-resolution input PNG patches (noise, blur, low light).
  - `split.json`: Dataset partition mapping:
    - **Train**: 900 pairs
    - **Validation**: 300 pairs
    - **Test**: 300 pairs (held-out)

## Preprocessing
Processed pairs are generated using `ml/src/preprocess.py` and partitioned with `ml/src/split_dataset.py`.
