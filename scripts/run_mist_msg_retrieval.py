#!/usr/bin/env python3
"""Run MassSpecGym retrieval metrics for MIST, NN, and DreaMS predictions."""

import argparse
import subprocess
import sys
from pathlib import Path


CANDIDATE_FILES = {
    "official_formula": "cands_df_test_formula_256.tsv",
    "official_mass": "cands_df_test_mass_256.tsv",
}


def find_candidate(raw_root, filename):
    direct = raw_root / filename
    if direct.exists():
        return direct
    matches = sorted(raw_root.rglob(filename))
    return matches[0] if matches else None


def main(args):
    repo = args.repo_root.resolve()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for candidate_kind, filename in CANDIDATE_FILES.items():
        candidate_file = find_candidate(args.candidates_root, filename)
        if candidate_file is None:
            print(f"Missing candidate table: {filename}")
            continue
        for split in args.splits:
            method_predictions = {
                "tuned_mist": args.mist_results_root
                / f"MSG_MIST_4096_{split}"
                / "test_results.pkl",
                "nearest_neighbour": args.nn_dir / f"massspecgym_{split}.pkl",
                "dreams_nn": args.dreams_dir / f"massspecgym_{split}_dreaMS.pkl",
            }
            for method, predictions in method_predictions.items():
                if not predictions.exists():
                    print(f"Missing predictions for {method}/{split}: {predictions}")
                    continue
                output = out_dir / f"massspecgym_{split}_{method}_{candidate_kind}.csv"
                cmd = [
                    args.python,
                    str(repo / "benchmarked_models" / "evaluation" / "evaluate_mist_retrieval.py"),
                    "--predictions",
                    str(predictions),
                    "--labels-file",
                    str(args.labels_file),
                    "--candidates-file",
                    str(candidate_file),
                    "--output",
                    str(output),
                ]
                print(" ".join(cmd))
                if not args.dry_run:
                    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--labels-file",
        type=Path,
        default=Path("data/metadata/massspecgym_msg_all_metadata.tsv"),
    )
    parser.add_argument(
        "--candidates-root",
        type=Path,
        default=Path("data/massspecgym"),
        help="Directory containing MassSpecGym official candidate TSV files.",
    )
    parser.add_argument(
        "--mist-results-root",
        type=Path,
        default=Path("benchmarked_models/mist/results/mist"),
    )
    parser.add_argument(
        "--nn-dir",
        type=Path,
        default=Path("results/nearest_neighbour/nn_sim/all_train_candidates"),
    )
    parser.add_argument(
        "--dreams-dir",
        type=Path,
        default=Path("results/nearest_neighbour/nn_sim_dreaMS/all_train_candidates"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/comparison/massspecgym_retrieval"),
    )
    parser.add_argument("--splits", nargs="+", default=["random", "scaffold"])
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args())
