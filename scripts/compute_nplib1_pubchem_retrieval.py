#!/usr/bin/env python3
"""Score NPLIB1 retrieval against PubChem candidates grouped by formula."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import multiprocessing as mp
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem


METHODS = ("Tuned MIST", "Nearest neighbour", "DreaMS NN")
SPLITS = ("random", "scaffold")
WORKER_QUERIES_BY_FORMULA: dict[str, list[dict[str, Any]]] = {}


def two_d_inchikey(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value.split("-")[0]


def fp_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    if isinstance(value, str):
        text = value.strip()
        return np.fromiter(
            (1.0 if char == "1" else 0.0 for char in text),
            dtype=np.float32,
            count=len(text),
        )
    return np.asarray(value, dtype=np.float32)


def candidate_fp(smiles: str) -> tuple[int, ...] | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=4096)
    bits = tuple(bitvect.GetOnBits())
    if not bits:
        return None
    return bits


def load_metadata(path: Path) -> dict[str, dict[str, str]]:
    df = pd.read_csv(path, sep="\t", dtype={"id": str})
    needed = ["id", "formula", "inchikey"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df.set_index("id")[["formula", "inchikey"]].to_dict("index")


def add_query(
    queries_by_formula: dict[str, list[dict[str, Any]]],
    metadata: dict[str, dict[str, str]],
    dataset: str,
    split: str,
    method: str,
    spec_id: Any,
    pred_fp: Any,
) -> None:
    spec_id = str(spec_id)
    meta = metadata.get(spec_id)
    if not meta:
        return
    formula = meta.get("formula")
    target = two_d_inchikey(meta.get("inchikey"))
    if not formula or not target:
        return
    pred = fp_array(pred_fp)
    norm = float(np.linalg.norm(pred))
    queries_by_formula[formula].append(
        {
            "dataset": dataset,
            "split": split,
            "method": method,
            "spec_id": spec_id,
            "pred": pred,
            "pred_norm": norm,
            "target_2d": target,
        }
    )


def load_mist_queries(
    queries_by_formula: dict[str, list[dict[str, Any]]],
    metadata: dict[str, dict[str, str]],
    split: str,
    pred_dir: Path,
    results_root: Path,
    run_template: str,
) -> None:
    npz_file = pred_dir / f"nplib1_{split}_preds_only.npz"
    if npz_file.exists():
        data = np.load(npz_file)
        for spec_id, pred in zip(data["ids"], data["preds"]):
            add_query(
                queries_by_formula,
                metadata,
                "NPLIB1",
                split,
                "Tuned MIST",
                spec_id,
                pred,
            )
        return

    pkl_file = results_root / run_template.format(split=split) / "test_results.pkl"
    with pkl_file.open("rb") as handle:
        predictions = pickle.load(handle)
    for spec_id, record in predictions.items():
        add_query(
            queries_by_formula,
            metadata,
            "NPLIB1",
            split,
            "Tuned MIST",
            spec_id,
            record["pred"],
        )


def load_nn_queries(
    queries_by_formula: dict[str, list[dict[str, Any]]],
    metadata: dict[str, dict[str, str]],
    split: str,
    nn_dir: Path,
) -> None:
    with (nn_dir / f"NPLIB1_{split}.pkl").open("rb") as handle:
        records = pickle.load(handle)
    for record in records:
        add_query(
            queries_by_formula,
            metadata,
            "NPLIB1",
            split,
            "Nearest neighbour",
            record["spec_id"],
            record["pred_fp"],
        )


def load_dreams_queries(
    queries_by_formula: dict[str, list[dict[str, Any]]],
    metadata: dict[str, dict[str, str]],
    split: str,
    dreams_dir: Path,
) -> None:
    path = dreams_dir / f"NPLIB1_{split}.pkl"
    if not path.exists():
        path = dreams_dir / f"NPLIB1_{split}_dreaMS.pkl"
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if isinstance(data, dict):
        for spec_id, pred in zip(data["test_ids"], data["pred_fps"]):
            add_query(queries_by_formula, metadata, "NPLIB1", split, "DreaMS NN", spec_id, pred)
    else:
        for record in data:
            add_query(
                queries_by_formula,
                metadata,
                "NPLIB1",
                split,
                "DreaMS NN",
                record["spec_id"],
                record["pred_fp"],
            )


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for split in SPLITS:
        for method in METHODS:
            selected = [
                rec
                for rec in records
                if rec["split"] == split and rec["method"] == method
            ]
            n_total = len(selected)
            covered = [rec for rec in selected if rec["has_candidates"]]
            n_covered = len(covered)
            target_present = [
                rec for rec in covered if rec["target_in_candidates"]
            ]
            row = {
                "dataset": "NPLIB1",
                "split": split,
                "method": method,
                "candidate_set": "pubchem_formula",
                "n_total": n_total,
                "n_covered": n_covered,
                "n_no_candidates": n_total - n_covered,
                "n_target_in_candidates": len(target_present),
                "target_candidate_coverage": (
                    len(target_present) / n_covered if n_covered else float("nan")
                ),
            }
            for k in (1, 5, 10):
                hits = sum(1 for rec in covered if rec[f"top{k}"])
                all_hits = sum(1 for rec in selected if rec[f"top{k}"])
                row[f"top{k}_2d_inchikey"] = hits / n_covered if n_covered else float("nan")
                row[f"top{k}_2d_inchikey_all_predictions"] = (
                    all_hits / n_total if n_total else float("nan")
                )
            rows.append(row)
    return rows


def init_worker(queries_by_formula: dict[str, list[dict[str, Any]]]) -> None:
    global WORKER_QUERIES_BY_FORMULA
    WORKER_QUERIES_BY_FORMULA = queries_by_formula
    RDLogger.DisableLog("rdApp.*")


def score_candidate_line(task: tuple[int, str]) -> tuple[int, str, list[dict[str, Any]]]:
    line_no, line = task
    formula, raw_json = line.rstrip("\n").split("\t", 1)
    queries = WORKER_QUERIES_BY_FORMULA.get(formula)
    if not queries:
        return line_no, formula, []
    candidates = json.loads(raw_json)
    return line_no, formula, score_formula(formula, candidates, queries)


def score_formula(
    formula: str,
    candidates: list[list[str]],
    queries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cand_bits: list[np.ndarray] = []
    cand_norms: list[float] = []
    cand_2d: list[str] = []
    for item in candidates:
        if len(item) < 2:
            continue
        smiles, inchikey = item[0], item[1]
        target_2d = two_d_inchikey(inchikey)
        if not smiles or not target_2d:
            continue
        bits = candidate_fp(smiles)
        if bits is None:
            continue
        cand_bits.append(np.asarray(bits, dtype=np.int16))
        cand_norms.append(math.sqrt(len(bits)))
        cand_2d.append(target_2d)

    outputs = []
    if not cand_bits:
        for query in queries:
            outputs.append(
                {
                    **{k: v for k, v in query.items() if k != "pred"},
                    "formula": formula,
                    "has_candidates": False,
                    "n_candidates": 0,
                    "target_in_candidates": False,
                    "top1": False,
                    "top5": False,
                    "top10": False,
                }
            )
        return outputs

    target_set = set(cand_2d)
    cand_norms_arr = np.asarray(cand_norms, dtype=np.float32)
    for query in queries:
        pred = query["pred"]
        pred_norm = query["pred_norm"]
        if pred_norm <= 0:
            scores = np.full(len(cand_bits), -np.inf, dtype=np.float32)
        else:
            scores = np.fromiter(
                (
                    float(pred[bits].sum()) / (pred_norm * cand_norm)
                    for bits, cand_norm in zip(cand_bits, cand_norms_arr)
                ),
                dtype=np.float32,
                count=len(cand_bits),
            )
        order = np.argsort(scores)[::-1][:10]
        ranked = [cand_2d[idx] for idx in order]
        target = query["target_2d"]
        outputs.append(
            {
                **{k: v for k, v in query.items() if k != "pred"},
                "formula": formula,
                "has_candidates": True,
                "n_candidates": len(cand_bits),
                "target_in_candidates": target in target_set,
                "top1": target in ranked[:1],
                "top5": target in ranked[:5],
                "top10": target in ranked[:10],
            }
        )
    return outputs


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata/NPLIB1_metadata.tsv"))
    parser.add_argument(
        "--candidates",
        type=Path,
        nargs="+",
        default=Path("results/comparison/nplib1_pubchem_hdf5_candidates.tsv.gz"),
        help="Gzipped TSV: formula<TAB>JSON list of [smiles, inchikey] PubChem candidates.",
    )
    parser.add_argument("--candidate-set", default="pubchem_formula")
    parser.add_argument(
        "--mist-pred-dir",
        type=Path,
        default=Path("results/comparison/compact_mist_predictions"),
        help="Optional directory containing nplib1_{split}_preds_only.npz files.",
    )
    parser.add_argument(
        "--mist-results-root",
        type=Path,
        default=Path("benchmarked_models/mist/results/mist"),
        help="Root containing standard MIST run directories with test_results.pkl.",
    )
    parser.add_argument(
        "--mist-run-template",
        default="NPLIB1_MIST_4096_{split}",
        help="Format string under --mist-results-root for NPLIB1 split runs.",
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
        "--output-prefix",
        type=Path,
        default=Path("results/comparison/nplib1_pubchem_formula_retrieval_full_test_methods"),
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    metadata = load_metadata(args.metadata)
    queries_by_formula: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split in SPLITS:
        load_mist_queries(
            queries_by_formula,
            metadata,
            split,
            args.mist_pred_dir,
            args.mist_results_root,
            args.mist_run_template,
        )
        load_nn_queries(queries_by_formula, metadata, split, args.nn_dir)
        load_dreams_queries(queries_by_formula, metadata, split, args.dreams_dir)

    records: list[dict[str, Any]] = []
    seen_formulas: set[str] = set()
    candidate_paths = args.candidates
    if isinstance(candidate_paths, Path):
        candidate_paths = [candidate_paths]
    for candidate_path in candidate_paths:
        print(f"scoring_candidates={candidate_path}", flush=True)
        with gzip.open(candidate_path, "rt") as handle:
            if args.workers > 1:
                initargs = (queries_by_formula,)
                with mp.Pool(args.workers, initializer=init_worker, initargs=initargs) as pool:
                    tasks = enumerate(handle, start=1)
                    for completed, (line_no, formula, scored) in enumerate(
                        pool.imap_unordered(score_candidate_line, tasks, chunksize=1),
                        start=1,
                    ):
                        if scored:
                            seen_formulas.add(formula)
                            records.extend(scored)
                        if completed % 100 == 0:
                            print(
                                f"completed_lines={completed} last_line={line_no} "
                                f"scored_records={len(records)}",
                                flush=True,
                            )
            else:
                init_worker(queries_by_formula)
                for line_no, line in enumerate(handle, start=1):
                    _, formula, scored = score_candidate_line((line_no, line))
                    if scored:
                        seen_formulas.add(formula)
                        records.extend(scored)
                    if line_no % 100 == 0:
                        print(
                            f"processed_lines={line_no} scored_records={len(records)}",
                            flush=True,
                        )

    for formula, queries in queries_by_formula.items():
        if formula in seen_formulas:
            continue
        records.extend(score_formula(formula, [], queries))

    rows = summarize_records(records)
    for row in rows:
        row["candidate_set"] = args.candidate_set
    out_csv = args.output_prefix.with_suffix(".csv")
    out_json = args.output_prefix.with_suffix(".json")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {out_csv}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
