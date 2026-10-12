import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import segmentation_models_pytorch as smp

try:
    from pycocotools import mask as mask_utils
except Exception:
    mask_utils = None

st.set_page_config(page_title="Solar Filament Segmentation", page_icon="☀️",
                   layout="wide", initial_sidebar_state="collapsed")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TTA = DEVICE == "cuda"          # flip TTA only when a GPU is available
HERE = Path(__file__).resolve().parent

# ---------------- model path: repo first, local Windows path as fallback ----------------
CANDIDATES = [
    HERE / "weights" / "best.pt",
    HERE.parent / "weights" / "best.pt",
    Path(r"C:\Users\Nithin T\Desktop\CV_Projects\solar-filament-segmentation\weights\best.pt"),
]
MODEL_PATH = next((p for p in CANDIDATES if p.exists()), None)
CFG_PATH = MODEL_PATH.with_name("config.json") if MODEL_PATH else None
CFG = json.load(open(CFG_PATH)) if CFG_PATH and CFG_PATH.exists() else {}
ENCODER = CFG.get("encoder", "resnet34")
DEF_THR = float(CFG.get("thr", 0.7))
DEF_MIN_AREA = int(CFG.get("min_area", 400))
CLOSE_K = int(CFG.get("close_k", 0))


# ---------------- inference ----------------
@st.cache_resource(show_spinner="Loading model...")
def load_model(path: str, encoder: str):
    m = smp.Unet(encoder, encoder_weights=None, in_channels=1, classes=1)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.to(DEVICE).eval()


def norm(img):
    return ((img.astype(np.float32) / 255.0) - 0.5) / 0.5


@torch.no_grad()
def run_once(model, img_u8, tta):
    h, w = img_u8.shape
    ph, pw = (-h) % 32, (-w) % 32
    x = np.pad(img_u8, ((0, ph), (0, pw)), mode="reflect") if (ph or pw) else img_u8
    t = torch.from_numpy(norm(x))[None, None].to(DEVICE)
    with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=(DEVICE == "cuda")):
        p = torch.sigmoid(model(t))
        if tta:
            p = (p + torch.sigmoid(model(t.flip(-1))).flip(-1)
                   + torch.sigmoid(model(t.flip(-2))).flip(-2)) / 3
    return p[0, 0].float().cpu().numpy()[:h, :w]


def run_tiled(model, img_u8, tta, tile=1024, pad=64):
    h, w = img_u8.shape
    out = np.zeros((h, w), np.float32)
    step = tile - 2 * pad
    for y in range(0, h, step):
        for x in range(0, w, step):
            y0, x0 = max(y - pad, 0), max(x - pad, 0)
            y1, x1 = min(y + step + pad, h), min(x + step + pad, w)
            p = run_once(model, img_u8[y0:y1, x0:x1], tta)
            ye, xe = min(y + step, h), min(x + step, w)
            out[y:ye, x:xe] = p[y - y0:ye - y0, x - x0:xe - x0]
    return out


@st.cache_data(show_spinner="Running the model...", max_entries=4)
def compute_prob(file_bytes: bytes, tiled: bool):
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    model = load_model(str(MODEL_PATH), ENCODER)
    prob = run_tiled(model, img, TTA) if tiled else run_once(model, img, TTA)
    return img, prob


def to_instances(prob, thr, min_area):
    b = (prob > thr).astype(np.uint8)
    if CLOSE_K > 0:
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K)))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(b, connectivity=8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return lab, keep, stats, cents


def encode_rle(m):
    return mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))["counts"].decode()


def make_overlay(img, lab, keep, cents, alpha):
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.float32)
    rng = np.random.RandomState(3)
    for i in keep:
        m = lab == i
        vis[m] = (1 - alpha) * vis[m] + alpha * rng.randint(60, 255, 3).astype(np.float32)
    vis = vis.astype(np.uint8)
    scale = max(img.shape) / 1024
    for r, i in enumerate(keep, 1):
        cx, cy = cents[i]
        cv2.putText(vis, str(r), (int(cx) + 8, int(cy) - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8 * scale, (255, 215, 0), max(1, int(2 * scale)), cv2.LINE_AA)
    return vis


# ---------------- styling ----------------
CSS = """
<style>
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.stApp { background: radial-gradient(1200px 650px at 18% -10%, #14284f 0%, #0a1424 58%); }
.block-container { padding-top: 1.3rem; padding-bottom: 2rem; max-width: 1500px; }

.sf-header { display:flex; align-items:center; justify-content:space-between;
             padding-bottom:1rem; margin-bottom:1.3rem; border-bottom:1px solid #1c2f4d; }
.sf-brand { display:flex; align-items:center; gap:16px; }
.sf-title { font-size:2rem; font-weight:700; color:#fff; line-height:1.1; }
.sf-sub { color:#9db0cf; font-size:1rem; margin-top:4px; }
.sf-pill { display:flex; align-items:center; gap:8px; background:#0f2a2a; border:1px solid #145046;
           color:#23d18b; padding:8px 16px; border-radius:999px; font-weight:600; font-size:.95rem; }
.sf-dot { width:10px; height:10px; border-radius:50%; background:#23d18b; }
.sf-pill.bad { background:#2a1313; border-color:#5a2020; color:#ff6b6b; }
.sf-pill.bad .sf-dot { background:#ff6b6b; }

.st-key-settings, .st-key-uploadbar, .st-key-card_in, .st-key-card_out, .st-key-card_dl {
  background:#0f1d33; border:1px solid #1c2f4d; border-radius:16px; padding:18px 20px; }
.st-key-card_in img, .st-key-card_out img { border-radius:10px; }

.sf-card-title { display:flex; align-items:center; gap:12px; font-size:1.25rem;
                 font-weight:700; color:#fff; margin-bottom:.8rem; }
.sf-ico { display:inline-flex; width:38px; height:38px; border-radius:12px; background:#15294a;
          color:#7fb0ff; align-items:center; justify-content:center; font-size:1.1rem; }
.sf-count { color:#3b82f6; text-decoration:underline; }
.sf-row { display:flex; justify-content:space-between; font-weight:600; color:#e6edf8; margin-top:1.1rem; }
.sf-val { color:#3b82f6; text-decoration:underline; }
.sf-empty { display:flex; align-items:center; justify-content:center; min-height:380px;
            border:1px dashed #27406a; border-radius:12px; color:#7f93b5; }

[data-testid="stSliderThumbValue"] { display:none; }

.st-key-uploadbar [data-testid="stFileUploader"] { display:flex; flex-direction:row-reverse;
    justify-content:space-between; align-items:center; gap:16px; }
.st-key-uploadbar [data-testid="stFileUploaderDropzone"] { background:transparent; border:none; padding:0; }
.st-key-uploadbar [data-testid="stFileUploaderDropzoneInstructions"] { display:none; }
.st-key-uploadbar [data-testid="stFileUploaderDropzone"] button { background:#2563eb; color:#fff; border:none;
    border-radius:10px; padding:.75rem 1.5rem; font-size:0; }
.st-key-uploadbar [data-testid="stFileUploaderDropzone"] button::after { content:"⬆  Upload Image";
    font-size:1rem; font-weight:600; }
.st-key-uploadbar [data-testid="stFileUploaderFile"] { background:#15294a; border-radius:12px; padding:6px 10px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def sun_svg():
    rays = "".join(
        f'<line x1="{24 + 15 * math.cos(a):.1f}" y1="{24 + 15 * math.sin(a):.1f}" '
        f'x2="{24 + 21 * math.cos(a):.1f}" y2="{24 + 21 * math.sin(a):.1f}"/>'
        for a in [i * math.pi / 6 for i in range(12)])
    return (f'<svg width="48" height="48" viewBox="0 0 48 48"><circle cx="24" cy="24" r="9" fill="#4d8dff"/>'
            f'<g stroke="#4d8dff" stroke-width="2.6" stroke-linecap="round">{rays}</g></svg>')


def show_image(x, **kw):
    try:
        st.image(x, width="stretch", **kw)
    except Exception:
        st.image(x, use_container_width=True, **kw)


def show_df(df):
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True)


def row(label, value):
    return f'<div class="sf-row"><span>{label}</span><span class="sf-val">{value}</span></div>'


# ---------------- header ----------------
if MODEL_PATH is None:
    pill = '<div class="sf-pill bad"><span class="sf-dot"></span>Model missing</div>'
else:
    load_model(str(MODEL_PATH), ENCODER)
    pill = '<div class="sf-pill"><span class="sf-dot"></span>Model Ready</div>'

st.markdown(
    f'<div class="sf-header"><div class="sf-brand">{sun_svg()}<div>'
    f'<div class="sf-title">Solar Filament Segmentation</div>'
    f'<div class="sf-sub">Detect and segment solar filaments from solar images</div></div></div>{pill}</div>',
    unsafe_allow_html=True)
if MODEL_PATH is None:
    st.error("best.pt not found. Put it in the repo's `weights/` folder.")
    st.stop()

# ---------------- layout ----------------
left, right = st.columns([1, 3.2], gap="large")

with left:
    with st.container(key="settings"):
        st.markdown('<div class="sf-card-title"><span class="sf-ico">⚙</span>Settings</div>',
                    unsafe_allow_html=True)
        ph1 = st.empty()
        thr = st.slider("Probability threshold", 0.0, 1.0, round(DEF_THR / 0.05) * 0.05, 0.05,
                        label_visibility="collapsed")
        ph1.markdown(row("Probability threshold", f"{thr:.2f}"), unsafe_allow_html=True)

        ph2 = st.empty()
        min_area = st.slider("Minimum filament area (px)", 100, 2000,
                             int(min(max(round(DEF_MIN_AREA / 25) * 25, 100), 2000)), 25,
                             label_visibility="collapsed")
        ph2.markdown(row("Minimum filament area (px)", min_area), unsafe_allow_html=True)

        tiled = st.checkbox("Low-memory tiled inference", value=True)

        ph3 = st.empty()
        alpha = st.slider("Overlay opacity", 0.0, 1.0, 0.6, 0.05, label_visibility="collapsed")
        ph3.markdown(row("Overlay opacity", f"{alpha:.2f}"), unsafe_allow_html=True)

with right:
    with st.container(key="uploadbar"):
        up = st.file_uploader("Upload a solar image", type=["jpg", "jpeg", "png"],
                              label_visibility="collapsed")

    c1, c2 = st.columns(2, gap="medium")

    if up is None:
        with c1:
            with st.container(key="card_in"):
                st.markdown('<div class="sf-card-title"><span class="sf-ico">🖼</span>Input</div>',
                            unsafe_allow_html=True)
                st.markdown('<div class="sf-empty">Upload an image to begin</div>', unsafe_allow_html=True)
        with c2:
            with st.container(key="card_out"):
                st.markdown('<div class="sf-card-title"><span class="sf-ico">⛶</span>'
                            'Detected filaments</div>', unsafe_allow_html=True)
                st.markdown('<div class="sf-empty">Results will appear here</div>', unsafe_allow_html=True)
        st.stop()

    img, prob = compute_prob(up.getvalue(), tiled)
    lab, keep, stats, cents = to_instances(prob, thr, min_area)
    overlay = make_overlay(img, lab, keep, cents, alpha)

    with c1:
        with st.container(key="card_in"):
            st.markdown('<div class="sf-card-title"><span class="sf-ico">🖼</span>Input</div>',
                        unsafe_allow_html=True)
            show_image(img, clamp=True)
    with c2:
        with st.container(key="card_out"):
            st.markdown(f'<div class="sf-card-title"><span class="sf-ico">⛶</span>Detected filaments: '
                        f'<span class="sf-count">{len(keep)}</span></div>', unsafe_allow_html=True)
            show_image(overlay)

    if img.shape != (2048, 2048):
        st.warning(f"Image is {img.shape[1]} x {img.shape[0]}; the model was trained on 2048 x 2048 "
                   f"images, so results may be worse.")

    with st.container(key="card_dl"):
        st.markdown('<div class="sf-card-title"><span class="sf-ico">⬇</span>Details &amp; downloads</div>',
                    unsafe_allow_html=True)
        rows = [{"id": r, "area_px": int(stats[i, cv2.CC_STAT_AREA]),
                 "x": int(stats[i, cv2.CC_STAT_LEFT]), "y": int(stats[i, cv2.CC_STAT_TOP]),
                 "width": int(stats[i, cv2.CC_STAT_WIDTH]), "height": int(stats[i, cv2.CC_STAT_HEIGHT])}
                for r, i in enumerate(keep, 1)]
        if rows:
            show_df(pd.DataFrame(rows))

        stem = Path(up.name).stem
        d1, d2, d3 = st.columns(3)
        if mask_utils is not None and keep:
            sub = pd.DataFrame([(f"{stem}_{r}", encode_rle(lab == i)) for r, i in enumerate(keep, 1)],
                               columns=["filament_id", "segmentation_rle"])
            d1.download_button("Download RLE CSV", sub.to_csv(index=False).encode(),
                               f"{stem}_filaments.csv", "text/csv")
        inst_map = np.zeros(lab.shape, np.uint16)
        for r, i in enumerate(keep, 1):
            inst_map[lab == i] = r
        d2.download_button("Download instance mask (PNG)", cv2.imencode(".png", inst_map)[1].tobytes(),
                           f"{stem}_instances.png", "image/png")
        d3.download_button("Download overlay (PNG)",
                           cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))[1].tobytes(),
                           f"{stem}_overlay.png", "image/png")