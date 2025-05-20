
# %%
from pathlib import Path
import numpy as np
import cv2
import math
import pandas as pd
import warnings

# ──────────────────────── Configuration ────────────────────────
PRED_DIR = Path(r"src\Evaluation\preds\DA2_preds\da2_pipe_preds_npy")    # your model outputs
GT_DIR   = Path(r"src\Evaluation\preds\Midas_preds\preds_npy")  # your ground-truth maps
PRED_EXT = ".npy"   # extension of prediction files (e.g. .npy or .png)
GT_EXT   = ".npy"   # extension of GT files
# ────────────────────────────────────────────────────────────────

# ──────────────────────── Metric functions ────────────────────────
def _mask(gt, pr):
    return (gt > 0) & np.isfinite(gt) & np.isfinite(pr)

def si_rmse(gt, pr):
    m = _mask(gt, pr)
    a = (gt[m]*pr[m]).sum() / (pr[m]**2).sum()
    return math.sqrt(((gt[m] - a*pr[m])**2).mean())

def abs_rel(gt, pr):
    m = _mask(gt, pr)
    return np.mean(np.abs(gt[m] - pr[m]) / gt[m])

def rmse(gt, pr):
    m = _mask(gt, pr)
    return math.sqrt(((gt[m] - pr[m])**2).mean())

def log_rmse(gt, pr):
    m = _mask(gt, pr)
    return math.sqrt(((np.log(gt[m]) - np.log(pr[m]))**2).mean())

def delta(gt, pr, thr):
    m = _mask(gt, pr)
    r = np.maximum(gt[m]/pr[m], pr[m]/gt[m])
    return np.mean(r < thr)

METRIC_FNS = {
    "si-RMSE":  si_rmse,
    "AbsRel":   abs_rel,
    "RMSE":     rmse,
    "logRMSE":  log_rmse,
    "δ1":       lambda g,p: delta(g,p,1.25),
    "δ2":       lambda g,p: delta(g,p,1.25**2),
    "δ3":       lambda g,p: delta(g,p,1.25**3),
}

# ───────────────────────── File loading ─────────────────────────
def load_depth(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext == ".npy":
        return np.load(path).astype(np.float32)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    # if >255 assume millimeters
    return img / (1000.0 if img.max()>255 else 1.0)

# ─────────────────────────── Main eval ────────────────────────────
records = []
for pred_path in sorted(PRED_DIR.glob(f"*{PRED_EXT}")):
    stem = pred_path.stem
    gt_path = GT_DIR / f"{stem}{GT_EXT}"
    if not gt_path.exists():
        warnings.warn(f"GT missing for {stem}, skipping")
        continue

    pred = load_depth(pred_path)
    gt   = load_depth(gt_path)

    rec = {"sample": stem}
    for name, fn in METRIC_FNS.items():
        rec[name] = fn(gt, pred)
    records.append(rec)

df = pd.DataFrame(records)
if df.empty:
    raise RuntimeError("No valid prediction/GT pairs found!")

# ────────────────────── Summarize & display ──────────────────────
summary = df.mean(numeric_only=True).to_frame("mean").T
print("\n=== Mean Depth-Estimation Metrics ===")
print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
