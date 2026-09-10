import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarked_models.common.benchmark_utils import (  # noqa: E402
    FALLBACK_TO_ALL_TRAIN_POLICIES,
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


def get_info(data):
    fp_info, formula_info, inchikey_info, smiles_info = {}, {}, {}, {}
    for row in tqdm(data, desc="index records"):
        spec_id = str(row["id_"])
        fp_info[spec_id] = fingerprint_to_bits(row["FPs"]["morgan4_4096"])
        formula_info[spec_id] = row.get("formula")
        inchikey_info[spec_id] = row.get("inchikey")
        smiles_info[spec_id] = row.get("smiles")
    return fp_info, formula_info, inchikey_info, smiles_info


def get_info_nist2023(folder, ids):
    fp_info, formula_info, inchikey_info, smiles_info = {}, {}, {}, {}
    for spec_id in tqdm(ids, desc="index nist2023 records"):
        row = load_pickle(folder / f"{spec_id}.pkl")
        fp_info[spec_id] = fingerprint_to_bits(row["FPs"]["morgan4_4096"])
        formula_info[spec_id] = row.get("formula")
        inchikey_info[spec_id] = row.get("inchikey")
        smiles_info[spec_id] = row.get("smiles")
    return fp_info, formula_info, inchikey_info, smiles_info


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
    train_emb,
    test_emb,
    fp_info,
    formula_info,
    inchikey_info,
    smiles_info,
    candidate_policy,
    batch_size=256,
):
    train_fp = [fp_info[spec_id] for spec_id in train_ids]
    train_formula = np.asarray([formula_info.get(spec_id) for spec_id in train_ids])
    formula_to_candidates = (
        None
        if candidate_policy == "all_train_candidates"
        else formula_candidate_index(train_formula)
    )

    records = []
    if candidate_policy == "all_train_candidates":
        train_norm = np.linalg.norm(train_emb, axis=1)
        train_normed = np.divide(
            train_emb,
            train_norm[:, None],
            out=np.zeros_like(train_emb, dtype=np.float32),
            where=train_norm[:, None] > 0,
        )
        for start in tqdm(range(0, len(test_ids), batch_size), desc=f"{dataset}/{split}"):
            end = min(start + batch_size, len(test_ids))
            batch = test_emb[start:end]
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
                        "method": "dreams_embedding_nn",
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

    for test_idx, spec_id in tqdm(list(enumerate(test_ids)), desc=f"{dataset}/{split}"):
        test_formula = formula_info.get(spec_id)
        test_fp = fp_info.get(spec_id)
        cand_idx = formula_to_candidates.get(test_formula, np.asarray([], dtype=int))
        used_fallback = False
        if len(cand_idx) == 0:
            if candidate_policy in SKIP_MISSING_FORMULA_POLICIES:
                continue
            if candidate_policy in FALLBACK_TO_ALL_TRAIN_POLICIES:
                # Formula-first, but never drop the query: fall back to the
                # whole training set so this spectrum still gets a prediction.
                cand_idx = np.arange(len(train_ids), dtype=int)
                used_fallback = True
            else:
                records.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "method": "dreams_embedding_nn",
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

        local_idx, similarity = cosine_top1(test_emb[test_idx], train_emb[cand_idx])
        train_idx = int(cand_idx[local_idx])
        top_train_id = train_ids[train_idx]
        pred_fp = train_fp[train_idx]
        records.append(
            {
                "dataset": dataset,
                "split": split,
                "method": "dreams_embedding_nn",
                "candidate_policy": candidate_policy,
                "spec_id": spec_id,
                "top_train_id": top_train_id,
                "has_candidate": True,
                "used_fallback": used_fallback,
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
    train = load_mgf_records(split_dir / "train.mgf", metadata_by_id=metadata)
    test = load_mgf_records(split_dir / "test.mgf", metadata_by_id=metadata)
    train_ids = list(train[0].keys())
    test_ids = list(test[0].keys())
    fp_info = dict(train[1])
    fp_info.update(test[1])
    formula_info = dict(train[2])
    formula_info.update(test[2])
    inchikey_info = dict(train[3])
    inchikey_info.update(test[3])
    smiles_info = dict(train[4])
    smiles_info.update(test[4])
    missing_fp = [spec_id for spec_id in train_ids + test_ids if spec_id not in fp_info]
    if missing_fp:
        raise ValueError(
            f"{dataset}/{split} MGF records missing FP values for {len(missing_fp)} spectra"
        )
    return train_ids, test_ids, (fp_info, formula_info, inchikey_info, smiles_info)


def main(args):
    if args.candidate_policy not in VALID_CANDIDATE_POLICIES:
        raise ValueError(f"Unknown candidate policy: {args.candidate_policy}")

    output_dir = Path(args.output_dir) / args.candidate_policy
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = {}

    for dataset in args.datasets:
        cached_info = None
        for split in args.splits:
            emb_dir = Path(args.embeddings_folder) / dataset / split
            if not (emb_dir / "train.pkl").exists() or not (emb_dir / "test.pkl").exists():
                print(f"Skipping missing DreaMS embeddings: {emb_dir}")
                continue
            split_file = Path(args.splits_folder) / dataset / f"{split}.json"
            use_mgf = args.input_source == "mgf" or (
                args.input_source == "auto" and not (Path(args.data_folder) / f"{dataset}.pkl").exists()
            )
            if use_mgf:
                train_ids, test_ids, info = load_split_mgf_info(args, dataset, split)
                split_file_label = str(Path(args.mgf_folder) / dataset / split)
                if args.candidate_policy != "all_train_candidates" and not any(info[1].values()):
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
                split_file_label = str(split_file)
                candidate_policy = args.candidate_policy

            if dataset == "nist2023" and not use_mgf:
                ids_needed = sorted(set(train_ids + test_ids))
                info = load_dataset_info(dataset, args.data_folder, ids_needed)
            elif not use_mgf:
                if cached_info is None:
                    cached_info = load_dataset_info(dataset, args.data_folder)
                info = cached_info

            train_emb = np.asarray(load_pickle(emb_dir / "train.pkl"), dtype=np.float32)
            test_emb = np.asarray(load_pickle(emb_dir / "test.pkl"), dtype=np.float32)
            records = compute_nn_records(
                dataset=dataset,
                split=split,
                train_ids=train_ids,
                test_ids=test_ids,
                train_emb=train_emb,
                test_emb=test_emb,
                fp_info=info[0],
                formula_info=info[1],
                inchikey_info=info[2],
                smiles_info=info[3],
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
                    "method": "dreams_embedding_nn",
                    "candidate_policy": candidate_policy,
                    "split_file": split_file_label,
                    "embeddings_folder": str(emb_dir),
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
            write_pickle(records, output_dir / f"{stem}_dreaMS.pkl")
            write_json(metrics, output_dir / f"{stem}_dreaMS_metrics.json")
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
        "--embeddings-folder",
        type=Path,
        default=REPO_ROOT / "results" / "nearest_neighbour" / "DreaMS_emb",
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
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "nearest_neighbour" / "nn_sim_dreaMS",
    )
    parser.add_argument("--datasets", nargs="+", default=["NPLIB1", "massspecgym"])
    parser.add_argument("--splits", nargs="+", default=["scaffold", "random"])
    parser.add_argument(
        "--candidate-policy",
        choices=sorted(VALID_CANDIDATE_POLICIES),
        default="same_formula_candidates_fallback",
    )
    main(parser.parse_args())
