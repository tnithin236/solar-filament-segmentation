# Solar Filament Instance Segmentation

Entry for the **Solar Filament Segmentation Challenge 2026** (Kaggle, IEEE BigData Cup).
The model finds individual filaments in 2048×2048 GONG H-alpha images and outputs one run-length-encoded mask per filament.

## Approach

1. **U-Net** (ResNet-34 encoder, ImageNet-pretrained, 1-channel input) predicts a binary filament mask.
2. **Post-processing** thresholds the probability map, splits it into instances with 8-connected components, and removes small blobs.
3. Each instance is **RLE-encoded** (pycocotools) into the competition's `submission.csv` format.

| Item | Setting |
|---|---|
| Data | MAGFiLO annotations, grayscale 2048×2048 JPEGs; annotations from multiple annotators merged by union |
| Split | Time-ordered 85% train / 15% validation (limits leakage between neighbouring observations) |
| Training crops | 512×512; 70% centred on a filament pixel, 30% random |
| Augmentation | Horizontal/vertical flip, 90° rotation, brightness/contrast, Gaussian noise |
| Loss | 0.5 × BCE + 0.5 × Dice |
| Optimiser | AdamW (lr 3e-4, weight decay 1e-4), OneCycleLR, mixed precision, seed 42 |
| Hardware | Google Colab, NVIDIA A100 |
| Inference | Full-resolution image, flip test-time augmentation |

## Results

| Metric | Value |
|---|---|
| Validation PQ | **0.3562** |
| Threshold | 0.7 |
| Minimum instance area | 400 px |

Validation PQ treats connected components of the merged ground-truth mask as ground-truth instances. It approximates the official metric, so the leaderboard score may differ.

## Repository

| File | Purpose |
|---|---|
| `train.py` | Mask preparation, training, validation PQ, post-processing tuning |
| `predict.py` | Test images → `submission.csv` |
| `filament_seg.py` | Model, inference, instance extraction, RLE and PQ helpers |
| `visualize.py` | Ground-truth vs prediction overlays |
| `notebook.ipynb` | Colab notebook with the full run |
| `weights/` | `best.pt` (trained model) and `config.json` (threshold, min area, encoder) |
| `requirements.txt` | Pinned package versions |

## Quick start (inference only)

```bash
pip install -r requirements.txt
python predict.py --test_dir <path>/test/test_images --out submission.csv
```

`predict.py` reads `weights/best.pt` and `weights/config.json` by default. A CUDA GPU is required.

## Retrain

```bash
python train.py --epochs 30
```

The script downloads the data with `kagglehub`, so you need a Kaggle login and accepted competition rules.

## Limitations

- A semantic U-Net plus connected components cannot separate filaments that touch each other, and may split one filament into pieces where the mask has gaps.
- Annotations from several annotators are merged by union, which can fuse nearby filaments.
- The validation metric is an approximation of the official PQ.

## Data and citation

Data: GONG H-alpha observations with MAGFiLO annotations, provided by the competition organisers (ESAIR Lab).
Competition: https://www.kaggle.com/competitions/filament-segmentation-2026