import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


def two_d_inchikey(value):
    if not isinstance(value, str) or not value:
        return None
    return value.split("-")[0]


def morgan_fp(smiles, n_bits=4096, radius=2):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    on_bits = list(fp.GetOnBits())
    arr[on_bits] = 1.0
    return arr


def cosine_scores(query, candidates):
    query = np.asarray(query, dtype=np.float32)
    candidates = np.asarray(candidates, dtype=np.float32)
    denom = np.linalg.norm(query) * np.linalg.norm(candidates, axis=1)
    return np.divide(
        candidates @ query,
        denom,
        out=np.zeros((len(candidates),), dtype=np.float32),
        where=denom > 0,
    )


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


def iter_predictions(predictions):
    """Yield (spec_id, predicted_fp) from MIST or NN prediction artifacts."""
    if isinstance(predictions, dict) and "test_ids" in predictions and "pred_fps" in predictions:
        for spec_id, pred in zip(predictions["test_ids"], predictions["pred_fps"]):
            yield str(spec_id), fp_array(pred)
        return

    if isinstance(predictions, dict):
        for spec_id, record in predictions.items():
            if isinstance(record, dict) and "pred" in record:
                yield str(spec_id), fp_array(record["pred"])
            elif isinstance(record, dict) and "pred_fp" in record:
                yield str(spec_id), fp_array(record["pred_fp"])
        return

    if isinstance(predictions, list):
        for record in predictions:
            if not record.get("has_candidate", True):
                continue
            pred = record.get("pred_fp")
            if pred is None:
                continue
            yield str(record["spec_id"]), fp_array(pred)
        return

    raise TypeError(f"Unsupported predictions object: {type(predictions)!r}")


def main(args):
    try:
        import rdkit  # noqa: F401
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except ImportError as exc:
        raise SystemExit("RDKit is required for MIST retrieval evaluation") from exc

    with open(args.predictions, "rb") as fp:
        predictions = dict(iter_predictions(pickle.load(fp)))
    labels = pd.read_csv(args.labels_file, sep="\t")
    if "spec" not in labels.columns and "name" in labels.columns:
        labels = labels.rename(columns={"name": "spec"})
    labels["inchikey_2d"] = labels["inchikey"].map(two_d_inchikey)
    label_map = labels.set_index("spec")["inchikey_2d"].to_dict()
    prediction_ids = set(predictions.keys())
    candidates = pd.read_csv(args.candidates_file, sep="\t")
    candidates = candidates.loc[candidates["spec"].isin(prediction_ids)]
    candidate_groups = {
        spec_id: group for spec_id, group in candidates.groupby("spec", sort=False)
    }

    fp_cache = {}
    hits = {k: 0 for k in args.k}
    covered = 0
    rows = []
    for spec_id, pred in tqdm(predictions.items(), desc="retrieval"):
        if spec_id not in label_map:
            continue
        spec_candidates = candidate_groups.get(spec_id)
        if spec_candidates is None or spec_candidates.empty:
            continue
        cand_fps = []
        cand_inchikeys = []
        for _, row in spec_candidates.iterrows():
            smiles = row["smiles"]
            if smiles not in fp_cache:
                fp_cache[smiles] = morgan_fp(smiles, n_bits=args.n_bits)
            fp = fp_cache[smiles]
            if fp is None:
                continue
            cand_fps.append(fp)
            cand_inchikeys.append(two_d_inchikey(row.get("inchikey")))
        if not cand_fps:
            continue
        scores = cosine_scores(pred, np.vstack(cand_fps))
        order = np.argsort(-scores)
        ranked = [cand_inchikeys[i] for i in order]
        target = label_map[spec_id]
        covered += 1
        for k in args.k:
            hits[k] += int(target in ranked[:k])
        rows.append(
            {
                "spec": spec_id,
                "target_2d_inchikey": target,
                "top_1_2d_inchikey": ranked[0],
                "top_1_score": float(scores[order[0]]),
                "n_candidates": len(ranked),
            }
        )

    summary = {"n_covered": covered}
    for k in args.k:
        summary[f"top_{k}_exact_2d_inchikey"] = float(hits[k] / covered) if covered else None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    Path(args.output).with_suffix(".json").write_text(pd.Series(summary).to_json(indent=2))
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels-file", type=Path, required=True)
    parser.add_argument("--candidates-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-bits", type=int, default=4096)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    main(parser.parse_args())
