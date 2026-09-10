"""Nearest-neighbour sensitivity to the number of neighbours (k).

The corrected nearest-neighbour baseline uses the single most similar training
spectrum. This script sweeps k over the *whole* training split (no formula
filtering) and aggregates the neighbours' fingerprints two ways:

* ``vote``    -- majority vote on each bit,
* ``average`` -- mean of each bit, thresholded at 0.5 for the binary score.

For odd k the two agree on Jaccard by construction; they differ on fingerprint
cosine, which is scored against the continuous aggregate.

Reads the same MGF/spectra inputs as ``02_compute_nn.py`` and writes one JSON
summary per dataset/split plus a combined ``summary.json``.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarked_models.common.benchmark_utils import (  # noqa: E402
    load_split_file,
    write_json,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402

_nn = import_module("02_compute_nn")


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def jaccard_rows(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    inter = np.logical_and(pred, target).sum(axis=1)
    union = np.logical_or(pred, target).sum(axis=1)
    return inter / (union + 1e-9)


def cosine_rows(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    num = (pred * target).sum(axis=1)
    den = np.linalg.norm(pred, axis=1) * np.linalg.norm(target, axis=1)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def topk_indices(test_normed: np.ndarray, train_normed: np.ndarray, k: int, chunk: int = 256) -> np.ndarray:
    """Indices of the k most cosine-similar training spectra, best first."""
    out = np.empty((len(test_normed), k), dtype=np.int64)
    for start in tqdm(range(0, len(test_normed), chunk), desc=f"top-{k}"):
        end = min(start + chunk, len(test_normed))
        sims = test_normed[start:end] @ train_normed.T
        part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sims, part, axis=1), axis=1)
        out[start:end] = np.take_along_axis(part, order, axis=1)
    return out


def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for dataset in args.datasets:
        cached = None
        for split in args.splits:
            split_file = Path(args.splits_folder) / dataset / f"{split}.json"
            if args.input_source != "mgf" and not split_file.exists():
                print(f"Skipping missing split file: {split_file}")
                continue
            if args.input_source == "mgf":
                # Same spectrum representation as 02_compute_nn.py in MGF mode.
                # The two sources are not interchangeable: on an identical split
                # they bin spectra differently and give different neighbours.
                train_ids, test_ids, info = _nn.load_split_mgf_info(args, dataset, split)
                ms_info, fp_info = info[0], info[1]
            else:
                ids = load_split_file(split_file)
                train_ids, test_ids = ids["train"], ids["test"]
                if cached is None:
                    cached = _nn.load_dataset_info(dataset, args.data_folder)
                ms_info, fp_info = cached[0], cached[1]

            train_ms = normalize_rows([ms_info[i] for i in train_ids])
            test_ms = normalize_rows([ms_info[i] for i in test_ids])
            train_fp = np.asarray([fp_info[i] for i in train_ids], dtype=np.float32)
            test_fp = np.asarray([fp_info[i] for i in test_ids], dtype=np.float32)

            neighbours = topk_indices(test_ms, train_ms, max(args.k_values))
            cell = {}
            for k in args.k_values:
                fps = train_fp[neighbours[:, :k]]          # (n_test, k, n_bits)
                mean_fp = fps.mean(axis=1)
                vote_fp = (fps.sum(axis=1) * 2 > k).astype(np.float32)   # strict majority

                cell[f"k={k}"] = {
                    "jaccard_vote": float(jaccard_rows(vote_fp, test_fp).mean()),
                    "jaccard_average": float(jaccard_rows((mean_fp > 0.5).astype(np.float32), test_fp).mean()),
                    "fp_cosine_vote": float(cosine_rows(vote_fp, test_fp).mean()),
                    "fp_cosine_average": float(cosine_rows(mean_fp, test_fp).mean()),
                }
            cell["n_test"] = len(test_ids)
            cell["n_train"] = len(train_ids)
            summary[f"{dataset}/{split}"] = cell
            write_json(cell, out_dir / f"{dataset}_{split}_knn.json")
            print(f"{dataset}/{split} (n={len(test_ids)}):")
            for k in args.k_values:
                c = cell[f"k={k}"]
                print(f"   k={k}: jaccard vote={c['jaccard_vote']:.4f} avg={c['jaccard_average']:.4f}"
                      f" | fp cosine vote={c['fp_cosine_vote']:.4f} avg={c['fp_cosine_average']:.4f}")

    write_json(summary, out_dir / "summary.json")
    print(f"\nwrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", default="data/processed_data")
    parser.add_argument("--splits-folder", default="data/splits")
    parser.add_argument("--output-dir", default="results/nearest_neighbour/knn_sensitivity")
    parser.add_argument("--datasets", nargs="+", default=["NPLIB1", "massspecgym"])
    parser.add_argument("--splits", nargs="+", default=["random", "scaffold"])
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--input-source", choices=["mgf", "processed_pickle"], default="mgf")
    parser.add_argument("--mgf-folder", default="data/MGF_files")
    parser.add_argument("--metadata-file", type=Path, default=None)
    parser.add_argument("--bin-resolution", type=float, default=0.25)
    parser.add_argument("--max-da", type=float, default=2000.0)
    main(parser.parse_args())
