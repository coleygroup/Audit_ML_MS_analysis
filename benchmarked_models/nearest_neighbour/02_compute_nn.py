import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarked_models.common.benchmark_utils import (  # noqa: E402
    SKIP_MISSING_FORMULA_POLICIES,
    VALID_CANDIDATE_POLICIES,
    cosine_top1,
    fingerprint_to_bits,
    jaccard_score,
    load_pickle,
    load_split_file,
    summarize_records,
    write_json,
    write_pickle,
)
from benchmarked_models.common.mgf_utils import load_metadata_sidecar, load_mgf_records  # noqa: E402


def bin_MS(spec, bin_resolution=0.25, max_da=2000.0):
    peaks = spec["peaks"]
    precursor_mz = spec["precursor_MZ_final"]
    peaks = peaks + [{"mz": precursor_mz, "intensity_norm": 100}]

    n_bins = int(math.ceil(max_da / bin_resolution))
    out = [0.0] * n_bins
    inv = 1.0 / bin_resolution
    for peak in peaks:
        bin_idx = math.floor(peak["mz"] * inv)
        if 0 <= bin_idx < n_bins:
            out[bin_idx] += peak["intensity_norm"]
    return np.asarray(out, dtype=np.float32)


def get_info(data):
    ms_info, fp_info, formula_info, inchikey_info, smiles_info = {}, {}, {}, {}, {}
    for row in tqdm(data, desc="index records"):
        spec_id = str(row["id_"])
        ms_info[spec_id] = bin_MS(row)
        fp_info[spec_id] = fingerprint_to_bits(row["FPs"]["morgan4_4096"])
        formula_info[spec_id] = row.get("formula")
        inchikey_info[spec_id] = row.get("inchikey")
        smiles_info[spec_id] = row.get("smiles")
    return ms_info, fp_info, formula_info, inchikey_info, smiles_info


def get_info_nist2023(folder, ids):
    ms_info, fp_info, formula_info, inchikey_info, smiles_info = {}, {}, {}, {}, {}
    for spec_id in tqdm(ids, desc="index nist2023 records"):
        row = load_pickle(folder / f"{spec_id}.pkl")
        ms_info[spec_id] = bin_MS(row)
        fp_info[spec_id] = fingerprint_to_bits(row["FPs"]["morgan4_4096"])
        formula_info[spec_id] = row.get("formula")
        inchikey_info[spec_id] = row.get("inchikey")
        smiles_info[spec_id] = row.get("smiles")
    return ms_info, fp_info, formula_info, inchikey_info, smiles_info


def candidate_indices(policy, train_formula, test_formula):
    if policy == "all_train_candidates":
        return np.arange(len(train_formula), dtype=int)
    return np.asarray(
        [idx for idx, formula in enumerate(train_formula) if formula == test_formula],
        dtype=int,
    )


def formula_candidate_index(train_formula):
    index = {}
    for idx, formula in enumerate(train_formula):
        index.setdefault(formula, []).append(idx)
    return {
        formula: np.asarray(indices, dtype=int)
        for formula, indices in index.items()
    }


def compute_nn_records(
    dataset,
    split,
    train_ids,
    test_ids,
    ms_info,
    fp_info,
    formula_info,
    inchikey_info,
    smiles_info,
    candidate_policy,
    batch_size=64,
):
    train_ms = np.asarray([ms_info[spec_id] for spec_id in train_ids], dtype=np.float32)
    train_fp = [fp_info[spec_id] for spec_id in train_ids]
    train_formula = np.asarray([formula_info.get(spec_id) for spec_id in train_ids])
    formula_to_candidates = (
        None
        if candidate_policy == "all_train_candidates"
        else formula_candidate_index(train_formula)
    )

    records = []
    if candidate_policy == "all_train_candidates":
        train_norm = np.linalg.norm(train_ms, axis=1)
        train_normed = np.divide(
            train_ms,
            train_norm[:, None],
            out=np.zeros_like(train_ms, dtype=np.float32),
            where=train_norm[:, None] > 0,
        )
        test_ms = np.asarray([ms_info[spec_id] for spec_id in test_ids], dtype=np.float32)
        for start in tqdm(range(0, len(test_ids), batch_size), desc=f"{dataset}/{split}"):
            end = min(start + batch_size, len(test_ids))
            batch = test_ms[start:end]
            batch_norm = np.linalg.norm(batch, axis=1)
            batch_normed = np.divide(
                batch,
                batch_norm[:, None],
                out=np.zeros_like(batch, dtype=np.float32),
                where=batch_norm[:, None] > 0,
            )
            sims = batch_normed @ train_normed.T
            top_indices = np.argmax(sims, axis=1)
            for offset, train_idx in enumerate(top_indices):
                spec_id = test_ids[start + offset]
                top_train_id = train_ids[int(train_idx)]
                pred_fp = train_fp[int(train_idx)]
                test_fp = fp_info[spec_id]
                records.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "method": "binned_spectrum_nn",
                        "candidate_policy": candidate_policy,
                        "spec_id": spec_id,
                        "top_train_id": top_train_id,
                        "has_candidate": True,
                        "similarity": float(sims[offset, train_idx]),
                        "formula": formula_info.get(spec_id),
                        "target_fp": test_fp.tolist(),
                        "pred_fp": pred_fp.tolist(),
                        "jaccard": jaccard_score(pred_fp, test_fp),
                        "inchikey": inchikey_info.get(spec_id),
                        "smiles": smiles_info.get(spec_id),
                        "top_train_inchikey": inchikey_info.get(top_train_id),
                        "top_train_smiles": smiles_info.get(top_train_id),
                    }
                )
        return records

    for spec_id in tqdm(test_ids, desc=f"{dataset}/{split}"):
        test_formula = formula_info.get(spec_id)
        test_fp = fp_info.get(spec_id)
        cand_idx = formula_to_candidates.get(test_formula, np.asarray([], dtype=int))
        if len(cand_idx) == 0:
            if candidate_policy in SKIP_MISSING_FORMULA_POLICIES:
                continue
            records.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "method": "binned_spectrum_nn",
                    "candidate_policy": candidate_policy,
                    "spec_id": spec_id,
                    "top_train_id": None,
                    "has_candidate": False,
                    "similarity": None,
                    "formula": test_formula,
                    "target_fp": test_fp.tolist() if test_fp is not None else None,
                    "pred_fp": None,
                    "jaccard": None,
                    "inchikey": inchikey_info.get(spec_id),
                    "smiles": smiles_info.get(spec_id),
                }
            )
            continue

        local_idx, similarity = cosine_top1(ms_info[spec_id], train_ms[cand_idx])
        train_idx = int(cand_idx[local_idx])
        top_train_id = train_ids[train_idx]
        pred_fp = train_fp[train_idx]
        records.append(
            {
                "dataset": dataset,
                "split": split,
                "method": "binned_spectrum_nn",
                "candidate_policy": candidate_policy,
                "spec_id": spec_id,
                "top_train_id": top_train_id,
                "has_candidate": True,
                "similarity": similarity,
                "formula": test_formula,
                "target_fp": test_fp.tolist(),
                "pred_fp": pred_fp.tolist(),
                "jaccard": jaccard_score(pred_fp, test_fp),
                "inchikey": inchikey_info.get(spec_id),
                "smiles": smiles_info.get(spec_id),
                "top_train_inchikey": inchikey_info.get(top_train_id),
                "top_train_smiles": smiles_info.get(top_train_id),
            }
        )
    return records


def load_dataset_info(dataset, data_folder, ids_needed=None):
    if dataset == "nist2023":
        if ids_needed is None:
            raise ValueError("NIST2023 pickle loading requires split ids")
        return get_info_nist2023(Path(data_folder) / "nist2023", ids_needed)
    data = load_pickle(Path(data_folder) / f"{dataset}.pkl")
    return get_info(data)


def load_split_mgf_info(args, dataset, split):
    split_dir = Path(args.mgf_folder) / dataset / split
    metadata = load_metadata_sidecar(args.metadata_file)
    train = load_mgf_records(
        split_dir / "train.mgf",
        bin_resolution=args.bin_resolution,
        max_da=args.max_da,
        metadata_by_id=metadata,
    )
    test = load_mgf_records(
        split_dir / "test.mgf",
        bin_resolution=args.bin_resolution,
        max_da=args.max_da,
        metadata_by_id=metadata,
    )
    train_ids = list(train[0].keys())
    test_ids = list(test[0].keys())
    info = []
    for train_part, test_part in zip(train, test):
        merged = dict(train_part)
        merged.update(test_part)
        info.append(merged)
    missing_fp = [spec_id for spec_id in train_ids + test_ids if spec_id not in info[1]]
    if missing_fp:
        raise ValueError(
            f"{dataset}/{split} MGF records missing FP values for {len(missing_fp)} spectra"
        )
    return train_ids, test_ids, tuple(info)


def main(args):
    if args.candidate_policy not in VALID_CANDIDATE_POLICIES:
        raise ValueError(f"Unknown candidate policy: {args.candidate_policy}")

    output_dir = Path(args.output_dir) / args.candidate_policy
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = {}

    for dataset in args.datasets:
        cached_info = None
        for split in args.splits:
            split_file = Path(args.splits_folder) / dataset / f"{split}.json"
            use_mgf = args.input_source == "mgf" or (
                args.input_source == "auto" and not (Path(args.data_folder) / f"{dataset}.pkl").exists()
            )
            if use_mgf:
                train_ids, test_ids, info = load_split_mgf_info(args, dataset, split)
                split_file_label = str(Path(args.mgf_folder) / dataset / split)
                if args.candidate_policy != "all_train_candidates" and not any(info[2].values()):
                    print(
                        f"{dataset}/{split}: formula metadata absent in MGF; "
                        "falling back to all_train_candidates"
                    )
                    candidate_policy = "all_train_candidates"
                else:
                    candidate_policy = args.candidate_policy
            else:
                if not split_file.exists():
                    print(f"Skipping missing split file: {split_file}")
                    continue
                split_ids = load_split_file(split_file)
                train_ids = split_ids["train"]
                test_ids = split_ids["test"]
                candidate_policy = args.candidate_policy
                split_file_label = str(split_file)
            if not train_ids or not test_ids:
                print(f"Skipping {dataset}/{split}: empty train or test split")
                continue

            if use_mgf:
                pass
            elif dataset == "nist2023":
                ids_needed = sorted(set(train_ids + test_ids))
                info = load_dataset_info(dataset, args.data_folder, ids_needed)
            else:
                if cached_info is None:
                    cached_info = load_dataset_info(dataset, args.data_folder)
                info = cached_info

            records = compute_nn_records(
                dataset=dataset,
                split=split,
                train_ids=train_ids,
                test_ids=test_ids,
                ms_info=info[0],
                fp_info=info[1],
                formula_info=info[2],
                inchikey_info=info[3],
                smiles_info=info[4],
                candidate_policy=candidate_policy,
                batch_size=args.batch_size,
            )
            metrics = summarize_records(records)
            original_test_count = len(test_ids)
            n_records_written = len(records)
            n_skipped = original_test_count - n_records_written
            predicted_mean = metrics["mean_jaccard_predicted"]
            mean_jaccard_zero_skipped = (
                float(predicted_mean * metrics["n_evaluated"] / original_test_count)
                if predicted_mean is not None and original_test_count
                else 0.0
            )
            metrics.update(
                {
                    "dataset": dataset,
                    "split": split,
                    "method": "binned_spectrum_nn",
                    "candidate_policy": candidate_policy,
                    "split_file": split_file_label,
                    "input_source": "mgf" if use_mgf else "processed_pickle",
                    "n_test_original": original_test_count,
                    "n_records_written": n_records_written,
                    "n_skipped_no_formula_candidate": n_skipped,
                    "coverage_of_original_test": (
                        float(metrics["n_evaluated"] / original_test_count)
                        if original_test_count
                        else 0.0
                    ),
                    "mean_jaccard_zero_skipped": mean_jaccard_zero_skipped,
                }
            )

            stem = f"{dataset}_{split}"
            write_pickle(records, output_dir / f"{stem}.pkl")
            write_json(metrics, output_dir / f"{stem}_metrics.json")
            all_metrics[stem] = metrics

    write_json(all_metrics, output_dir / "summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-folder",
        type=Path,
        default=REPO_ROOT / "data" / "processed_data",
    )
    parser.add_argument(
        "--splits-folder",
        type=Path,
        default=REPO_ROOT / "data" / "splits",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "nearest_neighbour" / "nn_sim",
    )
    parser.add_argument(
        "--mgf-folder",
        type=Path,
        default=REPO_ROOT / "data" / "MGF_files",
    )
    parser.add_argument(
        "--input-source",
        choices=["auto", "processed_pickle", "mgf"],
        default="auto",
    )
    parser.add_argument("--metadata-file", type=Path, default=None)
    parser.add_argument("--bin-resolution", type=float, default=0.25)
    parser.add_argument("--max-da", type=float, default=2000.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--datasets", nargs="+", default=["NPLIB1", "massspecgym"])
    parser.add_argument("--splits", nargs="+", default=["scaffold", "random"])
    parser.add_argument(
        "--candidate-policy",
        choices=sorted(VALID_CANDIDATE_POLICIES),
        default="all_train_candidates",
    )
    main(parser.parse_args())
