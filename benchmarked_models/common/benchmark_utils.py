import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


VALID_CANDIDATE_POLICIES = {
    "all_train_candidates",
    "same_formula_candidates_full",
    "same_formula_candidates_legacy",
    "same_formula_candidates_skip_missing",
}

SKIP_MISSING_FORMULA_POLICIES = {
    "same_formula_candidates_legacy",
    "same_formula_candidates_skip_missing",
}


def ensure_dir(path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as fp:
        return pickle.load(fp)


def write_pickle(data: Any, path: Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "wb") as fp:
        pickle.dump(data, fp)


def load_json(path: Path) -> Any:
    with open(path) as fp:
        return json.load(fp)


def write_json(data: Any, path: Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w") as fp:
        json.dump(data, fp, indent=2)


def fingerprint_to_bits(fp: Any) -> np.ndarray:
    """Convert common fingerprint encodings to a 0/1 numpy vector."""
    if fp is None:
        raise ValueError("Cannot convert a null fingerprint")
    if isinstance(fp, np.ndarray):
        return fp.astype(np.int8)
    if isinstance(fp, (list, tuple)):
        return np.asarray(fp, dtype=np.int8)
    if isinstance(fp, str):
        return np.asarray([int(bit) for bit in fp.strip()], dtype=np.int8)
    raise TypeError(f"Unsupported fingerprint type: {type(fp)!r}")


def jaccard_score(pred_fp: Any, target_fp: Any) -> Optional[float]:
    if pred_fp is None or target_fp is None:
        return None
    pred = fingerprint_to_bits(pred_fp).astype(bool)
    target = fingerprint_to_bits(target_fp).astype(bool)
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(pred, target).sum() / union)


def cosine_top1(query: np.ndarray, candidates: np.ndarray) -> Tuple[int, float]:
    """Return the top cosine candidate index and similarity."""
    query = np.asarray(query, dtype=np.float32)
    candidates = np.asarray(candidates, dtype=np.float32)
    query_norm = np.linalg.norm(query)
    cand_norms = np.linalg.norm(candidates, axis=1)
    denom = cand_norms * query_norm
    sims = np.divide(
        candidates @ query,
        denom,
        out=np.zeros_like(cand_norms, dtype=np.float32),
        where=denom > 0,
    )
    idx = int(np.argmax(sims))
    return idx, float(sims[idx])


def two_d_inchikey(inchikey: Optional[str]) -> Optional[str]:
    if not isinstance(inchikey, str) or not inchikey:
        return None
    return inchikey.split("-")[0]


def load_split_file(
    split_file: Path,
    name_col: Optional[str] = None,
    split_col: Optional[str] = None,
) -> Dict[str, List[str]]:
    """Load JSON or TSV split files and return train/val/test id lists."""
    split_file = Path(split_file)
    if split_file.suffix.lower() == ".json":
        split_data = load_json(split_file)
        return {
            key: [str(item).replace(".pkl", "") for item in split_data.get(key, [])]
            for key in ("train", "val", "test")
        }

    split_df = pd.read_csv(split_file, sep="\t")
    if name_col is None:
        name_col = "name" if "name" in split_df.columns else "spec"
    if split_col is None:
        split_col = "split" if "split" in split_df.columns else "Fold_0"
    if name_col not in split_df.columns or split_col not in split_df.columns:
        raise ValueError(
            f"{split_file} must contain {name_col!r} and {split_col!r}; "
            f"columns={list(split_df.columns)}"
        )

    out = {"train": [], "val": [], "test": []}
    for _, row in split_df.iterrows():
        fold = str(row[split_col])
        if fold in out:
            out[fold].append(str(row[name_col]).replace(".pkl", ""))
    return out


def normalize_split_tsv(
    split_file: Path,
    output_file: Path,
    name_col: Optional[str] = None,
    split_col: Optional[str] = None,
) -> Path:
    split_df = pd.read_csv(split_file, sep="\t")
    if name_col is None:
        name_col = "name" if "name" in split_df.columns else "spec"
    if split_col is None:
        split_col = "split" if "split" in split_df.columns else "Fold_0"
    normalized = split_df[[name_col, split_col]].rename(
        columns={name_col: "name", split_col: "split"}
    )
    ensure_dir(Path(output_file).parent)
    normalized.to_csv(output_file, sep="\t", index=False)
    return Path(output_file)


def summarize_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n_total = len(records)
    predicted = [r for r in records if r.get("has_candidate", True) and r.get("jaccard") is not None]
    no_candidate = [r for r in records if not r.get("has_candidate", True)]
    jaccards = np.asarray([r["jaccard"] for r in predicted], dtype=float)
    similarities = np.asarray(
        [r["similarity"] for r in predicted if r.get("similarity") is not None],
        dtype=float,
    )

    summary: Dict[str, Any] = {
        "n_test": n_total,
        "n_evaluated": len(predicted),
        "n_no_candidate": len(no_candidate),
        "coverage": float(len(predicted) / n_total) if n_total else 0.0,
    }
    if len(jaccards):
        summary.update(
            {
                "mean_jaccard_predicted": float(np.mean(jaccards)),
                "median_jaccard_predicted": float(np.median(jaccards)),
                "mean_jaccard_zero_missing": float(np.sum(jaccards) / n_total) if n_total else 0.0,
            }
        )
    else:
        summary.update(
            {
                "mean_jaccard_predicted": None,
                "median_jaccard_predicted": None,
                "mean_jaccard_zero_missing": 0.0 if n_total else None,
            }
        )
    summary["mean_similarity_predicted"] = (
        float(np.mean(similarities)) if len(similarities) else None
    )
    exact_total = 0
    exact_hits = 0
    for record in records:
        target = two_d_inchikey(record.get("inchikey"))
        pred = two_d_inchikey(record.get("top_train_inchikey"))
        if target is None or pred is None:
            continue
        exact_total += 1
        exact_hits += int(target == pred)
    summary["top_1_exact_2d_inchikey"] = (
        float(exact_hits / exact_total) if exact_total else None
    )
    summary["exact_2d_inchikey_covered"] = exact_total
    return summary


def exact_retrieval_at_k(
    ranked_candidate_inchikeys: Iterable[Sequence[str]],
    target_inchikeys: Iterable[Optional[str]],
    ks: Sequence[int] = (1, 5, 10),
) -> Dict[str, Any]:
    counts = {k: 0 for k in ks}
    total = 0
    covered = 0
    for ranked, target in zip(ranked_candidate_inchikeys, target_inchikeys):
        total += 1
        target_2d = two_d_inchikey(target)
        ranked_2d = [two_d_inchikey(i) for i in ranked if two_d_inchikey(i)]
        if not target_2d or not ranked_2d:
            continue
        covered += 1
        for k in ks:
            if target_2d in ranked_2d[:k]:
                counts[k] += 1
    out = {"retrieval_total": total, "retrieval_covered": covered}
    for k in ks:
        out[f"top_{k}_exact_2d_inchikey"] = (
            float(counts[k] / covered) if covered else None
        )
    return out
