import math
import os
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
import torch
from tqdm import tqdm

from MSPN_SDR.lib.model.MSPN import MSPN
from run_refinement import load_inputs, load_model, args, prepare_inputs, RGB_DIR, DEPTH_MD_DIR, TOTAL_VARIANCE_DIR


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


def run_evaluation(output_triples, title=''):
    records = []
    for pred, gt, sample_num in output_triples:
        rec = {"sample": sample_num}
        for name, fn in METRIC_FNS.items():
            rec[name] = fn(gt, pred)
        records.append(rec)

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No valid prediction/GT pairs found!")

    summary = df.mean(numeric_only=True).to_frame("mean").T
    print(f"\n=== Mean Depth-Estimation Metrics {'('+title+')' if title else ''} ===")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))


def load_depth(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext == ".npy":
        return np.load(path).astype(np.float32)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    # if >255 assume millimeters
    return img / (1000.0 if img.max()>255 else 1.0)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MSPN(args).to(device)
    model.eval()
    load_model(model, args)

    guidance_net = torch.load(os.path.join('.', 'MSPN_SDR', 'test_models', 'guidance_net.pt'))
    guidance_net = guidance_net.to(device).eval()

    with open(os.path.join('..', 'data', 'train_list.txt'), 'r') as f:
        all_samples = [s.strip() for s in sorted(list(f))]
    mid = int(len(all_samples)*0.9)
    tail_samples = all_samples[mid:]

    output_triples = []
    for sample_num in tqdm(tail_samples):

        rgb_file = os.path.join(RGB_DIR, f'sample_{sample_num}_rgb.png')
        var_file = os.path.join(TOTAL_VARIANCE_DIR, f'sample_{sample_num}_depth_var_total.npy')
        depth_md_file = os.path.join(DEPTH_MD_DIR, f'sample_{sample_num}_depth_mean.npy')

        rgb, s_depth, depth_md, variance = load_inputs(rgb_file, var_file, depth_md_file, threshold=0.05)
        rgb = rgb.to(device)

        # "Batch Size 1"
        rgb = rgb.unsqueeze(0)
        s_depth = s_depth.unsqueeze(0)
        depth_md = depth_md.unsqueeze(0)
        variance = variance.unsqueeze(0)

        pred_init, guide, s_depth, var_init = prepare_inputs(rgb, s_depth, depth_md, variance, guidance_net, device)
        
        with torch.no_grad():
            pred_inter, var_inter, _gain = model(
                pred_init,
                var_init,
                guide
            )
        
        gt = np.load(os.path.join(RGB_DIR, f'sample_{sample_num}_depth.npy'))
        gt = gt[np.newaxis, np.newaxis, ...]

        output_triples.append((pred_inter[-1], gt, sample_num))

    run_evaluation(output_triples)


if __name__ == '__main__':
    main()