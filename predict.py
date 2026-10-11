"""Generate the Kaggle submission CSV.
Usage: python predict.py --test_dir <.../test/test_images> --out submission.csv
"""
import argparse, os, json, cv2, pandas as pd
from tqdm import tqdm
from filament_seg import load_model, predict_prob, to_instances, encode_rle

ap = argparse.ArgumentParser()
ap.add_argument("--test_dir", required=True)
ap.add_argument("--weights", default="weights/best.pt")
ap.add_argument("--config", default="weights/config.json")
ap.add_argument("--out", default="submission.csv")
a = ap.parse_args()

cfg = json.load(open(a.config))
model = load_model(a.weights, cfg["encoder"])
rows = []
for f in tqdm(sorted(os.listdir(a.test_dir))):
    img = cv2.imread(os.path.join(a.test_dir, f), cv2.IMREAD_GRAYSCALE)
    stem = os.path.splitext(f)[0]
    for k, m in enumerate(to_instances(predict_prob(model, img), cfg["thr"], cfg["min_area"]), 1):
        rows.append((f"{stem}_{k}", encode_rle(m)))
pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(a.out, index=False)
print("saved", a.out, "-", len(rows), "filaments")
