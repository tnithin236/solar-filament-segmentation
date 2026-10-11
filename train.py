"""Train U-Net on the Solar Filament Segmentation Challenge 2026 data.
Usage: python train.py --epochs 30
Outputs: weights/best.pt, weights/config.json
"""
import os, glob, json, random, time, argparse
import numpy as np, cv2, torch, torch.nn as nn
import albumentations as A
import segmentation_models_pytorch as smp
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from filament_seg import build_model, norm, predict_prob, to_instances, panoptic_quality

ap = argparse.ArgumentParser()
ap.add_argument("--data_root", default=None, help="competition folder; downloaded with kagglehub if omitted")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--encoder", default="resnet34")
ap.add_argument("--crop", type=int, default=512)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
device = "cuda"
H = W = 2048
CROP = args.crop

# ---------- data ----------
if args.data_root is None:
    import kagglehub
    args.data_root = kagglehub.competition_download("filament-segmentation-2026")
ANN = glob.glob(f"{args.data_root}/**/*train*.json", recursive=True)[0]
TRAIN_DIR = os.path.dirname(glob.glob(f"{args.data_root}/**/train_images/*.jp*g", recursive=True)[0])
MASK_DIR = "masks"
os.makedirs(MASK_DIR, exist_ok=True)

coco = COCO(ANN)
files = []
for img_id, info in tqdm(coco.imgs.items(), desc="masks"):
    f = os.path.basename(info["file_name"])
    m = np.zeros((H, W), np.uint8)
    for ann in coco.imgToAnns[img_id]:          # union of all annotators
        m |= coco.annToMask(ann).astype(np.uint8)
    cv2.imwrite(f"{MASK_DIR}/{f}.png", m * 255)
    files.append(f)
files.sort()                                     # timestamp names -> temporal split
n_val = max(1, int(0.15 * len(files)))
train_files, val_files = files[:-n_val], files[-n_val:]
print(len(train_files), "train /", len(val_files), "val")

aug = A.Compose([A.HorizontalFlip(), A.VerticalFlip(), A.RandomRotate90(),
                 A.RandomBrightnessContrast(0.2, 0.2, p=0.5), A.GaussNoise(p=0.2)])


class FilDS(Dataset):
    def __init__(self, files, n):
        self.files, self.n = files, n
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        f = self.files[i % len(self.files)]
        img = cv2.imread(f"{TRAIN_DIR}/{f}", cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(f"{MASK_DIR}/{f}.png", cv2.IMREAD_GRAYSCALE) // 255
        ys, xs = np.nonzero(msk)
        if len(ys) and random.random() < 0.7:   # 70% crops centred on a filament pixel
            j = random.randrange(len(ys)); cy, cx = ys[j], xs[j]
        else:
            cy, cx = random.randrange(H), random.randrange(W)
        y0 = int(np.clip(cy - CROP // 2, 0, H - CROP)); x0 = int(np.clip(cx - CROP // 2, 0, W - CROP))
        img, msk = img[y0:y0 + CROP, x0:x0 + CROP], msk[y0:y0 + CROP, x0:x0 + CROP]
        out = aug(image=img, mask=msk)
        return (torch.from_numpy(norm(out["image"]))[None],
                torch.from_numpy(out["mask"].astype(np.float32))[None])


props = torch.cuda.get_device_properties(0)
BS = 16 if props.total_memory > 30e9 else 8
train_dl = DataLoader(FilDS(train_files, len(train_files) * 8), batch_size=BS, shuffle=True,
                      num_workers=args.workers, persistent_workers=args.workers > 0,
                      drop_last=True, pin_memory=True)

# ---------- model ----------
model = build_model(args.encoder, True, device).to(memory_format=torch.channels_last)
dice, bce = smp.losses.DiceLoss("binary"), nn.BCEWithLogitsLoss()
loss_fn = lambda p, t: 0.5 * bce(p, t) + 0.5 * dice(p, t)
amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
scaler = torch.amp.GradScaler(enabled=(amp_dtype == torch.float16))
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl))


def evaluate(thr=0.5, min_area=50, tta=False):
    S = TP = FP = FN = 0
    for f in val_files:
        img = cv2.imread(f"{TRAIN_DIR}/{f}", cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(f"{MASK_DIR}/{f}.png", cv2.IMREAD_GRAYSCALE) > 127
        n, lab = cv2.connectedComponents(gt.astype(np.uint8), connectivity=8)
        gts = [lab == k for k in range(1, n)]    # approximation of GT instances
        s, tp, fp, fn = panoptic_quality(to_instances(predict_prob(model, img, device, tta), thr, min_area), gts)
        S += s; TP += tp; FP += fp; FN += fn
    return S / (TP + 0.5 * FP + 0.5 * FN + 1e-9), TP, FP, FN


# ---------- train ----------
os.makedirs("weights", exist_ok=True)
best = -1
for ep in range(args.epochs):
    model.train(); t0 = time.time(); run = 0
    for x, y in tqdm(train_dl, leave=False):
        x = x.to(device).contiguous(memory_format=torch.channels_last); y = y.to(device)
        with torch.autocast("cuda", dtype=amp_dtype):
            loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        run += loss.item()
    pq, tp, fp, fn = evaluate()
    print(f"ep {ep+1:02d} loss {run/len(train_dl):.4f} | val PQ {pq:.4f} (TP {tp} FP {fp} FN {fn}) | {time.time()-t0:.0f}s", flush=True)
    if pq > best:
        best = pq; torch.save(model.state_dict(), "weights/best.pt")

# ---------- tune post-processing on validation ----------
model.load_state_dict(torch.load("weights/best.pt"))
res = sorted(((evaluate(t, m)[0], t, m) for t in [0.4, 0.5, 0.6, 0.7] for m in [50, 100, 200, 400, 800]), reverse=True)
print("top settings (PQ, thr, min_area):", res[:5])
json.dump({"thr": res[0][1], "min_area": res[0][2], "encoder": args.encoder, "val_pq": res[0][0]},
          open("weights/config.json", "w"), indent=2)
