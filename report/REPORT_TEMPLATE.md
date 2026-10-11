# Solar Filament Instance Segmentation - Technical Report (4 pages)

## 1. Problem and data
Task, GONG H-alpha 2048x2048 grayscale, MAGFiLO COCO-style polygons, multiple annotators, PQ metric.

## 2. Method
- Model: U-Net, ResNet-34 encoder (ImageNet), 1-channel input
- Training: 512x512 crops (70% centred on filament pixels), flips/rot90/brightness/noise, 0.5 BCE + 0.5 Dice, AdamW, OneCycleLR, AMP
- Instances: threshold -> 8-connected components -> min-area filter; flip TTA at inference
- Split: time-ordered 85/15 to limit leakage

## 3. Experiments
- Per-epoch val PQ curve (insert plot)
- Threshold / min-area grid (insert table)
- Ablations you actually ran (encoder, crop size, loss, post-processing)

## 4. Error analysis
Overlays (overlays.png): merged filaments, fragmented filaments, false-positive speckle.

## 5. Reproducibility
Repo URL, requirements.txt, seed 42, commands: `python train.py`, `python predict.py ...`.

## 6. Limitations and future work
Report only what you measured. Note that validation PQ uses connected components as GT instances (approximation) and that official PQ may differ.
