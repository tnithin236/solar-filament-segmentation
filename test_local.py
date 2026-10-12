import argparse, json, os, glob
import cv2, numpy as np, pandas as pd, torch
import segmentation_models_pytorch as smp
from pycocotools import mask as mask_utils
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--test_dir", required=True, help="folder with test images")
ap.add_argument("--weights", default="weights/best_r50.pt")
ap.add_argument("--config", default="weights/config_r50.json")
ap.add_argument("--out", default="submission_local.csv")
ap.add_argument("--no_tta", action="store_true", help="faster, slightly less accurate")
ap.add_argument("--viz", type=int, default=0, help="save overlay PNGs for the first N images")
ap.add_argument("--limit", type=int, default=0, help="only process the first N images")
a = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

cfg = json.load(open(a.config))
thr, min_area, close_k = cfg["thr"], cfg["min_area"], cfg.get("close_k", 0)
model = smp.Unet(cfg["encoder"], encoder_weights=None, in_channels=1, classes=1).to(device)
model.load_state_dict(torch.load(a.weights, map_location=device))
model.eval()
print(f"loaded {a.weights} | encoder {cfg['encoder']} | thr {thr} | min_area {min_area} | close_k {close_k}")

def norm(img):
    return ((img.astype(np.float32) / 255.0) - 0.5) / 0.5

@torch.no_grad()
def predict_prob(img):
    x = torch.from_numpy(norm(img))[None, None].to(device)
    amp = device == "cuda"
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
        p = torch.sigmoid(model(x))
        if not a.no_tta:
            p = (p + torch.sigmoid(model(x.flip(-1))).flip(-1)
                   + torch.sigmoid(model(x.flip(-2))).flip(-2)) / 3
    return p[0, 0].float().cpu().numpy()

def to_instances(prob):
    b = (prob > thr).astype(np.uint8)
    if close_k > 0:
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    return [lab == i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]

def encode_rle(m):
    return mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))["counts"].decode()

exts = ("*.jpg", "*.jpeg", "*.png")
files = sorted(sum([glob.glob(os.path.join(a.test_dir, e)) for e in exts], []))
if a.limit: files = files[:a.limit]
print(len(files), "images")
if a.viz: os.makedirs("viz_out", exist_ok=True)

rows = []
for idx, path in enumerate(tqdm(files)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    stem = os.path.splitext(os.path.basename(path))[0]
    insts = to_instances(predict_prob(img))
    for k, m in enumerate(insts, 1):
        rows.append((f"{stem}_{k}", encode_rle(m)))
    if idx < a.viz:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for m in insts:
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cs, -1, (0, 0, 255), 3)
        cv2.imwrite(f"viz_out/{stem}_pred.png", cv2.resize(vis, (1024, 1024)))

pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(a.out, index=False)
print(f"saved {a.out} | {len(rows)} filaments in {len(files)} images")