# Solar Filament Instance Segmentation (Kaggle 2026)

U-Net (ResNet-34, ImageNet-pretrained, 1-channel input) trained on 512x512 crops of 2048x2048
GONG H-alpha images (MAGFiLO annotations). The binary mask is split into filament instances
with connected components, then encoded to RLE for submission.

## Files
| File | Purpose |
|---|---|
| `train.py` | data prep, training, validation PQ, post-processing tuning -> `weights/` |
| `predict.py` | test images -> `submission.csv` (RLE via pycocotools) |
| `filament_seg.py` | model, inference, instance extraction, RLE, PQ helpers |
| `visualize.py` | ground-truth vs prediction overlays |
| `notebook.ipynb` | Colab notebook that runs the whole pipeline |
| `requirements.txt` | pinned package versions |

## Method
- Model: U-Net with `resnet34` encoder, 1-channel input
- Training: 512x512 crops (70% centred on filament pixels), flips / rot90 / brightness / noise,
  loss = 0.5 BCE + 0.5 Dice, AdamW, OneCycleLR, mixed precision, seed 42
- Split: time-ordered 85% train / 15% validation
- Inference: flip test-time augmentation, threshold, 8-connected components, minimum-area filter

## Reproduce
```bash
pip install -r requirements.txt
python train.py --epochs 30
python predict.py --test_dir <data>/test/test_images --out submission.csv
```
Kaggle login and accepted competition rules are required for the data download.

## Results
| Metric | Value |
|---|---|
| Validation PQ (approximate) | **0.3562** |
| Threshold | 0.7 |
| Minimum instance area | 400 px |

Validation PQ uses connected components of the ground-truth mask as ground-truth instances,
so it is an approximation of the official metric.
