"""Save GT-vs-prediction overlays for validation images (for the report)."""
import json, sys, cv2, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from filament_seg import load_model, predict_prob, to_instances

img_dir, mask_dir, names = sys.argv[1], sys.argv[2], sys.argv[3:]
cfg = json.load(open("weights/config.json")); model = load_model("weights/best.pt", cfg["encoder"])
fig, axs = plt.subplots(len(names), 3, figsize=(15, 5 * len(names)), squeeze=False)
for r, f in enumerate(names):
    img = cv2.imread(f"{img_dir}/{f}", 0); gt = cv2.imread(f"{mask_dir}/{f}.png", 0) > 127
    pr = np.zeros_like(gt)
    for m in to_instances(predict_prob(model, img), cfg["thr"], cfg["min_area"]): pr |= m
    for c, (t, mk, cm) in enumerate([("Image", None, None), ("Ground truth", gt, "autumn"), ("Prediction", pr, "cool")]):
        axs[r, c].imshow(img, cmap="gray")
        if mk is not None: axs[r, c].imshow(np.ma.masked_where(~mk, mk), cmap=cm, alpha=.7)
        axs[r, c].set_title(f"{t} - {f}" if c == 0 else t); axs[r, c].axis("off")
plt.tight_layout(); plt.savefig("overlays.png", dpi=120)
