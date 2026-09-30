"""
Select the fingerprint decision threshold on the VALIDATION split, then report the model on the test split.

Every fingerprint predictor in this repository turns a per-bit score into a binary fingerprint with a fixed
threshold of 0.5 (`predict.py`). This script sweeps the threshold on the validation predictions, keeps the
argmax, and applies that single value to the test predictions. The test split is never used for selection.

Inputs, per run folder (the layout written by `predict.py` / `run_nn.py`):
    val_results.pkl     {id: {"pred": [n_bits floats], "GT": [n_bits floats]}}   <- required
    test_results.pkl    same, on the test split                                  <- required

Output, per run folder:
    test_performance_val_threshold.json
        {"threshold", "val_jaccard", "test_jaccard", "test_jaccard_at_0.5", "n_val", "n_test", "scale", "grid"}

`pred` may hold either probabilities (MIST, nearest neighbour) or raw logits (`other_baselines`, whose
`predict.py` stores the pre-sigmoid output); `--scale auto` tells them apart and applies a sigmoid to logits.

Usage:
    python tune_threshold.py --runs ../results/FP_prediction/mist/*/*
    python tune_threshold.py --runs ../results/FP_prediction/*/*/* --scale auto
Generate the validation predictions first, e.g.
    cd mist            && python predict.py --checkpoint <run folder> --split val
    cd other_baselines && python predict.py --checkpoint <run folder> --split val
"""

import re
import json
import pickle
import argparse
import numpy as np
from pathlib import Path

# Threshold grid.
DEFAULT_GRID = np.round(np.arange(0.05, 0.9751, 0.025), 4)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def norm_id(key):
    """Result files key spectra inconsistently: str ids, and `torch.Tensor` objects in the NPLIB1
    `other_baselines` runs (which hash by identity, so they cannot be looked up). Normalise to str."""
    s = str(key)
    m = re.fullmatch(r"tensor\((\d+)\)", s)
    return m.group(1) if m else s


def stack(results):
    """(ids, pred, GT) as arrays, in a fixed order."""
    ids = [norm_id(k) for k in results]
    pred = np.asarray([np.asarray(v["pred"], dtype=np.float32) for v in results.values()])
    gt = np.asarray([np.asarray(v["GT"]) for v in results.values()], dtype=np.uint8)
    return ids, pred, gt


def to_probability(pred, scale):
    """`other_baselines/predict.py` stores pre-sigmoid logits, MIST and the retrieval baselines store
    probabilities. Detect it from the range unless told explicitly."""
    if scale == "auto":
        scale = "logit" if (pred.min() < 0.0 or pred.max() > 1.0) else "prob"
    if scale == "logit":
        return 1.0 / (1.0 + np.exp(-pred.astype(np.float64))), "logit"
    return pred, "prob"


def jaccard(pred_bits, gt_bits):
    """Row-wise Tanimoto, matching `nearest_neighbour/utils/nn_lib.jaccard` (empty union -> 0)."""
    pred_bits = pred_bits.astype(bool)
    gt_bits = gt_bits.astype(bool)
    inter = np.logical_and(pred_bits, gt_bits).sum(axis=1)
    union = np.logical_or(pred_bits, gt_bits).sum(axis=1)
    return inter / np.maximum(union, 1)


def sweep(prob, gt, grid, batch=512):
    """Mean Tanimoto at every threshold in `grid`, in row batches so a large split stays in memory."""
    total = np.zeros(len(grid))
    for s in range(0, prob.shape[0], batch):
        p, g = prob[s:s + batch], gt[s:s + batch]
        for i, t in enumerate(grid):
            total[i] += jaccard(p > t, g).sum()
    return total / prob.shape[0]


def tune(run, grid, scale, overwrite):
    val_path, test_path = run / "val_results.pkl", run / "test_results.pkl"
    out = run / "test_performance_val_threshold.json"
    if out.exists() and not overwrite:
        return "already done"
    if not val_path.exists():
        return "SKIP: no val_results.pkl (run predict.py --split val first)"
    if not test_path.exists():
        return "SKIP: no test_results.pkl"

    _, val_pred, val_gt = stack(load_pickle(val_path))
    val_prob, detected = to_probability(val_pred, scale)
    val_curve = sweep(val_prob, val_gt, grid)

    # Argmax on validation; ties -> the threshold closest to 0.5. The test split plays no part here.
    best = int(np.lexsort((np.abs(grid - 0.5), -val_curve))[0])
    threshold = float(grid[best])

    test_ids, test_pred, test_gt = stack(load_pickle(test_path))
    test_prob, _ = to_probability(test_pred, scale)
    test_j = float(jaccard(test_prob > threshold, test_gt).mean())
    test_j_half = float(jaccard(test_prob > 0.5, test_gt).mean())

    with open(out, "w") as f:
        json.dump({"threshold": threshold,
                   "val_jaccard": float(val_curve[best]),
                   "test_jaccard": test_j,
                   "test_jaccard_at_0.5": test_j_half,
                   "n_val": int(val_prob.shape[0]),
                   "n_test": int(test_prob.shape[0]),
                   "scale": detected,
                   "selected_on": "validation",
                   "grid": {str(t): float(j) for t, j in zip(grid, val_curve)}}, f, indent=2)
    return f"threshold {threshold:.3f} (val {val_curve[best]:.4f}) -> test {test_j:.4f} (was {test_j_half:.4f} at 0.5, {test_j - test_j_half:+.4f})"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run folders holding val_results.pkl and test_results.pkl")
    ap.add_argument("--scale", choices=("auto", "logit", "prob"), default="auto",
                    help="how to read `pred`: raw logits (sigmoid applied) or probabilities")
    ap.add_argument("--grid", nargs="+", type=float, default=None, help="thresholds to try (default 0.05..0.975 step 0.025)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    grid = np.asarray(a.grid if a.grid else DEFAULT_GRID, dtype=float)
    for r in sorted(a.runs):
        run = Path(r)
        if not run.is_dir():
            continue
        print(f"{run}: {tune(run, grid, a.scale, a.overwrite)}", flush=True)
