#!/usr/bin/env python3
"""Build CSV tables used by the public README from generated artifacts."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATASETS = ("NPLIB1", "massspecgym")
SPLITS = ("scaffold", "random")
METADATA_ID_COLUMNS = ("spec", "name", "id", "id_", "ID_")
METADATA_INCHIKEY_COLUMNS = (
    "inchikey",
    "inchikey_2d",
    "inchikey_full",
    "inchikey_original",
    "InChIKey",
)


def normalize_spec_id(value: Any) -> str:
    text = str(value).strip()
    tensor_match = re.match(r"^tensor\(([^,\)]+)", text)
    if tensor_match:
        text = tensor_match.group(1).strip()
    if text.endswith(".pkl"):
        text = text[:-4]
    return text.strip("'\"")


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


def jaccard(pred: Any, target: Any) -> float | None:
    if pred is None or target is None:
        return None
    pred_arr = fp_array(pred).astype(bool)
    target_arr = fp_array(target).astype(bool)
    union = np.logical_or(pred_arr, target_arr).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(pred_arr, target_arr).sum() / union)


def cosine(pred: Any, target: Any) -> float | None:
    if pred is None or target is None:
        return None
    pred_arr = fp_array(pred)
    target_arr = fp_array(target)
    denom = float(np.linalg.norm(pred_arr) * np.linalg.norm(target_arr))
    if denom <= 0:
        return 0.0
    return float(pred_arr.dot(target_arr) / denom)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def iter_mgf_ids(path: Path):
    with path.open(errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            upper = line.upper()
            if upper.startswith("ID_="):
                yield normalize_spec_id(line.split("=", 1)[1])


def metadata_path_for(dataset: str, args: argparse.Namespace) -> Path:
    if dataset == "NPLIB1":
        return args.nplib_metadata
    return args.msg_metadata


def load_metadata_inchikeys(path: Path) -> dict[str, str]:
    df = pd.read_csv(path, sep="\t", dtype=str)
    id_cols = [col for col in METADATA_ID_COLUMNS if col in df.columns]
    inchikey_col = next((col for col in METADATA_INCHIKEY_COLUMNS if col in df.columns), None)
    if not id_cols or inchikey_col is None:
        raise ValueError(
            f"{path} must contain at least one id column from {METADATA_ID_COLUMNS} "
            f"and one inchikey column from {METADATA_INCHIKEY_COLUMNS}; "
            f"columns={list(df.columns)}"
        )

    out: dict[str, str] = {}
    for _, row in df.iterrows():
        inchikey = row.get(inchikey_col)
        if not isinstance(inchikey, str) or not inchikey:
            continue
        for id_col in id_cols:
            spec_id = row.get(id_col)
            if isinstance(spec_id, str) and spec_id:
                out[normalize_spec_id(spec_id)] = inchikey
    return out


def get_formula_subset_records(
    args: argparse.Namespace,
    dataset: str,
    split: str,
) -> tuple[list[dict[str, Any]], Path | None]:
    candidate_paths = [
        args.formula_nn_dir / f"{dataset}_{split}.pkl",
        args.formula_dreams_dir / f"{dataset}_{split}_dreaMS.pkl",
        args.formula_oracle_dir / f"{dataset}_{split}_fp_oracle.pkl",
    ]
    for path in candidate_paths:
        if path.exists():
            return list(iter_nn_predictions(path)), path
    return [], None


def leakage_for_subset(
    args: argparse.Namespace,
    dataset: str,
    split: str,
    subset_ids: list[str],
) -> dict[str, Any]:
    if not subset_ids:
        return {
            "leakage_fraction": None,
            "leakage_denominator": 0,
            "leakage_source": None,
        }

    metadata_file = metadata_path_for(dataset, args)
    train_mgf = args.mgf_root / dataset / split / "train.mgf"
    if not metadata_file.exists() or not train_mgf.exists():
        return {
            "leakage_fraction": None,
            "leakage_denominator": 0,
            "leakage_source": f"missing metadata or train MGF: {metadata_file}; {train_mgf}",
        }

    id_to_inchikey = load_metadata_inchikeys(metadata_file)
    train_inchikeys = {
        two_d_inchikey(id_to_inchikey[spec_id])
        for spec_id in iter_mgf_ids(train_mgf)
        if spec_id in id_to_inchikey and two_d_inchikey(id_to_inchikey[spec_id])
    }
    exact_overlap = []
    for spec_id in subset_ids:
        target = two_d_inchikey(id_to_inchikey.get(normalize_spec_id(spec_id)))
        if target is not None:
            exact_overlap.append(target in train_inchikeys)

    return {
        "leakage_fraction": float(np.mean(exact_overlap)) if exact_overlap else None,
        "leakage_denominator": len(exact_overlap),
        "leakage_source": f"{metadata_file}; {train_mgf}",
    }


def iter_mist_predictions(path: Path):
    data = load_pickle(path)
    for spec_id, record in data.items():
        yield {
            "spec_id": normalize_spec_id(spec_id),
            "pred": record.get("pred"),
            "target": record.get("GT"),
            "jaccard": record.get("jaccard"),
            "inchikey": None,
            "top_train_inchikey": None,
        }


def iter_nn_predictions(path: Path):
    data = load_pickle(path)
    if isinstance(data, dict):
        for spec_id, pred, target in zip(data["test_ids"], data["pred_fps"], data["test_fps"]):
            yield {
                "spec_id": normalize_spec_id(spec_id),
                "pred": pred,
                "target": target,
                "jaccard": jaccard(pred, target),
                "inchikey": None,
                "top_train_inchikey": None,
            }
        return
    for record in data:
        if not record.get("has_candidate", True) or record.get("pred_fp") is None:
            continue
        yield {
            "spec_id": normalize_spec_id(record["spec_id"]),
            "pred": record.get("pred_fp"),
            "target": record.get("target_fp"),
            "jaccard": record.get("jaccard"),
            "inchikey": record.get("inchikey"),
            "top_train_inchikey": record.get("top_train_inchikey"),
        }


def summarize_prediction_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    jaccards = [r["jaccard"] for r in records if r.get("jaccard") is not None]
    cosines = [cosine(r["pred"], r["target"]) for r in records]
    cosines = [value for value in cosines if value is not None]
    return {
        "n": len(records),
        "mean_jaccard": float(np.mean(jaccards)) if jaccards else None,
        "median_jaccard": float(np.median(jaccards)) if jaccards else None,
        "mean_fp_cosine": float(np.mean(cosines)) if cosines else None,
        "median_fp_cosine": float(np.median(cosines)) if cosines else None,
    }


def build_full_test_table(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        dataset_label = "MassSpecGym" if dataset == "massspecgym" else dataset
        mist_prefix = "MSG" if dataset == "massspecgym" else "NPLIB1"
        for split in SPLITS:
            sources = [
                (
                    "MIST in Comment",
                    args.mist_results_root / f"{mist_prefix}_ORIGINAL_MIST_4096_{split}" / "test_results.pkl",
                    iter_mist_predictions,
                ),
                (
                    "Tuned MIST",
                    args.mist_results_root / f"{mist_prefix}_MIST_4096_{split}" / "test_results.pkl",
                    iter_mist_predictions,
                ),
                (
                    "Nearest neighbour",
                    args.nn_dir / f"{dataset}_{split}.pkl",
                    iter_nn_predictions,
                ),
                (
                    "DreaMS",
                    args.dreams_dir / f"{dataset}_{split}_dreaMS.pkl",
                    iter_nn_predictions,
                ),
            ]
            for method, path, loader in sources:
                if not path.exists():
                    rows.append(
                        {
                            "dataset": dataset_label,
                            "split": split,
                            "method": method,
                            "source": str(path),
                            "status": "missing",
                        }
                    )
                    continue
                summary = summarize_prediction_records(list(loader(path)))
                rows.append(
                    {
                        "dataset": dataset_label,
                        "split": split,
                        "method": method,
                        "source": str(path),
                        "status": "ok",
                        **summary,
                    }
                )
    return pd.DataFrame(rows)


def build_formula_filtered_table(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        dataset_label = "MassSpecGym" if dataset == "massspecgym" else dataset
        mist_prefix = "MSG" if dataset == "massspecgym" else "NPLIB1"
        for split in SPLITS:
            subset_records, subset_source = get_formula_subset_records(args, dataset, split)
            formula_subset_ids = [r["spec_id"] for r in subset_records]
            mist_path = (
                args.mist_results_root
                / f"{mist_prefix}_MIST_4096_{split}"
                / "test_results.pkl"
            )
            mist_records = list(iter_mist_predictions(mist_path)) if mist_path.exists() else None
            if mist_records is not None:
                mist_ids = {r["spec_id"] for r in mist_records}
                subset_ids = [spec_id for spec_id in formula_subset_ids if spec_id in mist_ids]
            else:
                subset_ids = formula_subset_ids
            subset_id_set = set(subset_ids)
            leakage = leakage_for_subset(args, dataset, split, subset_ids)
            sources = [
                (
                    "Tuned MIST",
                    mist_path,
                    iter_mist_predictions,
                    "model_prediction_on_same_formula_subset",
                ),
                (
                    "Formula NN",
                    args.formula_nn_dir / f"{dataset}_{split}.pkl",
                    iter_nn_predictions,
                    "same_formula_candidates_skip_missing",
                ),
                (
                    "Formula DreaMS NN",
                    args.formula_dreams_dir / f"{dataset}_{split}_dreaMS.pkl",
                    iter_nn_predictions,
                    "same_formula_candidates_skip_missing",
                ),
                (
                    "FP oracle (NN upper bound)",
                    args.formula_oracle_dir / f"{dataset}_{split}_fp_oracle.pkl",
                    iter_nn_predictions,
                    "same_formula_candidates_skip_missing",
                ),
            ]
            for method, path, loader, candidate_policy in sources:
                if method == "Tuned MIST" and subset_source is None:
                    rows.append(
                        {
                            "dataset": dataset_label,
                            "split": split,
                            "method": method,
                            "source": str(path),
                            "subset_source": None,
                            "subset_n": 0,
                            "formula_subset_n": len(formula_subset_ids),
                            "candidate_policy": candidate_policy,
                            "status": "missing_formula_subset",
                            **leakage,
                        }
                    )
                    continue
                if not path.exists():
                    rows.append(
                        {
                            "dataset": dataset_label,
                            "split": split,
                            "method": method,
                            "source": str(path),
                            "subset_source": str(subset_source) if subset_source else None,
                            "subset_n": len(subset_ids),
                            "formula_subset_n": len(formula_subset_ids),
                            "candidate_policy": candidate_policy,
                            "status": "missing",
                            **leakage,
                        }
                    )
                    continue
                records = mist_records if method == "Tuned MIST" and mist_records is not None else list(loader(path))
                if subset_source is not None:
                    records = [r for r in records if r["spec_id"] in subset_id_set]
                summary = summarize_prediction_records(records)
                subset_missing_from_source = max(len(subset_ids) - summary["n"], 0)
                rows.append(
                    {
                        "dataset": dataset_label,
                        "split": split,
                        "method": method,
                        "source": str(path),
                        "subset_source": str(subset_source) if subset_source else None,
                        "subset_n": len(subset_ids),
                        "formula_subset_n": len(formula_subset_ids),
                        "subset_missing_from_source": subset_missing_from_source,
                        "candidate_policy": candidate_policy,
                        "status": "ok",
                        **summary,
                        **leakage,
                    }
                )
    return pd.DataFrame(rows)


def read_retrieval_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "source": str(path)}
    return {"status": "ok", "source": str(path), **json.loads(path.read_text())}


def build_retrieval_table(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    nplib_path = args.nplib_retrieval.with_suffix(".csv")
    if nplib_path.exists():
        rows.extend(pd.read_csv(nplib_path).to_dict(orient="records"))
    else:
        rows.append({"dataset": "NPLIB1", "status": "missing", "source": str(nplib_path)})

    for split in SPLITS:
        for method_key, method_label in [
            ("tuned_mist", "Tuned MIST"),
            ("nearest_neighbour", "Nearest neighbour"),
            ("dreams_nn", "DreaMS"),
        ]:
            for candidate_set in ("official_formula", "official_mass"):
                path = (
                    args.msg_retrieval_dir
                    / f"massspecgym_{split}_{method_key}_{candidate_set}.json"
                )
                data = read_retrieval_json(path)
                rows.append(
                    {
                        "dataset": "MassSpecGym",
                        "split": split,
                        "method": method_label,
                        "candidate_set": candidate_set,
                        "n_covered": data.get("n_covered"),
                        "top1_2d_inchikey": data.get("top_1_exact_2d_inchikey"),
                        "top5_2d_inchikey": data.get("top_5_exact_2d_inchikey"),
                        "top10_2d_inchikey": data.get("top_10_exact_2d_inchikey"),
                        "status": data.get("status"),
                        "source": data.get("source"),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mist-results-root", type=Path, default=Path("benchmarked_models/mist/results/mist"))
    parser.add_argument("--nn-dir", type=Path, default=Path("results/nearest_neighbour/nn_sim/all_train_candidates"))
    parser.add_argument("--dreams-dir", type=Path, default=Path("results/nearest_neighbour/nn_sim_dreaMS/all_train_candidates"))
    parser.add_argument("--formula-nn-dir", type=Path, default=Path("results/nearest_neighbour/nn_sim/same_formula_candidates_skip_missing"))
    parser.add_argument("--formula-dreams-dir", type=Path, default=Path("results/nearest_neighbour/nn_sim_dreaMS/same_formula_candidates_skip_missing"))
    parser.add_argument("--formula-oracle-dir", type=Path, default=Path("results/nearest_neighbour/fp_oracle_upper_bound/same_formula_candidates_skip_missing"))
    parser.add_argument("--mgf-root", type=Path, default=Path("data/MGF_files"))
    parser.add_argument("--nplib-metadata", type=Path, default=Path("data/metadata/NPLIB1_metadata.tsv"))
    parser.add_argument("--msg-metadata", type=Path, default=Path("data/metadata/massspecgym_msg_all_metadata.tsv"))
    parser.add_argument("--nplib-retrieval", type=Path, default=Path("results/comparison/nplib1_pubchem_formula_retrieval_full_test_methods"))
    parser.add_argument("--msg-retrieval-dir", type=Path, default=Path("results/comparison/massspecgym_retrieval"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/comparison/readme_tables"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "full_test_fingerprint_metrics.csv": build_full_test_table(args),
        "formula_filtered_metrics.csv": build_formula_filtered_table(args),
        "retrieval_metrics.csv": build_retrieval_table(args),
    }
    for filename, df in outputs.items():
        out = args.output_dir / filename
        df.to_csv(out, index=False)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
