"""Shared helpers: model loading, inference, instance extraction, RLE, PQ."""
import numpy as np, cv2, torch
import segmentation_models_pytorch as smp
from pycocotools import mask as mask_utils


def build_model(encoder="resnet34", pretrained=True, device="cuda"):
    return smp.Unet(encoder, encoder_weights="imagenet" if pretrained else None,
                    in_channels=1, classes=1).to(device)


def load_model(weights, encoder="resnet34", device="cuda"):
    m = build_model(encoder, pretrained=False, device=device)
    m.load_state_dict(torch.load(weights, map_location=device))
    m.eval()
    return m


def norm(img_u8):
    return ((img_u8.astype(np.float32) / 255.0) - 0.5) / 0.5


@torch.no_grad()
def predict_prob(model, img_u8, device="cuda", tta=True):
    model.eval()
    x = torch.from_numpy(norm(img_u8))[None, None].to(device)
    dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.autocast("cuda", dtype=dt):
        p = torch.sigmoid(model(x))
        if tta:
            p = (p + torch.sigmoid(model(x.flip(-1))).flip(-1)
                   + torch.sigmoid(model(x.flip(-2))).flip(-2)) / 3
    return p[0, 0].float().cpu().numpy()


def to_instances(prob, thr=0.5, min_area=50):
    n, lab = cv2.connectedComponents((prob > thr).astype(np.uint8), connectivity=8)
    out = []
    for k in range(1, n):
        m = lab == k
        if m.sum() >= min_area:
            out.append(m)
    return out


def encode_rle(m):
    return mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))["counts"].decode()


def panoptic_quality(preds, gts, iou_thr=0.5):
    """Returns (sum_iou_of_TP, TP, FP, FN) for one image."""
    matched, iou_sum = set(), 0.0
    for g in gts:
        for j, p in enumerate(preds):
            if j in matched:
                continue
            iou = (g & p).sum() / ((g | p).sum() + 1e-9)
            if iou > iou_thr:
                matched.add(j)
                iou_sum += iou
                break
    tp = len(matched)
    return iou_sum, tp, len(preds) - tp, len(gts) - tp
