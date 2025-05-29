import argparse
import math
import os
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm


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


def run_evaluation(output_triples, title='', print_summary=True):
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
    if print_summary:
        print(f"\n=== Mean Depth-Estimation Metrics {'('+title+')' if title else ''} ===")
        print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    return summary


def load_depth(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext == ".npy":
        return np.load(path).astype(np.float32)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    # if >255 assume millimeters
    return img / (1000.0 if img.max()>255 else 1.0)


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--predictions_dir', type=str, default='../data/output/predictions')
    parser.add_argument('--gt_dir', type=str, default='../data/train/train')
    parser.add_argument('--train_list', type=str, default='../data/train_list.txt')

    args = parser.parse_args()

    with open(args.train_list, 'r') as f:
        all_samples = [s.strip() for s in sorted(list(f))]
    mid = int(len(all_samples)*0.9)
    tail_samples = all_samples[mid:]

    predictions_dir = args.predictions_dir
    gt_dir = args.gt_dir

    output_triples = []

    for filenames in tqdm(tail_samples):
        sample_num = filenames.strip()[7:13]
        pred_path = Path(os.path.join(predictions_dir, f'sample_{sample_num}_depth.npy'))
        gt_path = Path(os.path.join(gt_dir, f'sample_{sample_num}_depth.npy'))
        pred = load_depth(pred_path)
        gt = load_depth(gt_path)
        output_triples.append((pred, gt, sample_num))

    run_evaluation(output_triples)


if __name__ == '__main__':
    main()