"""
Split-level properties of each dataset / split (run_nn.py --audit).

    coverage          fraction of test spectra whose neutral formula occurs in the training split, i.e. the spectra a
                      formula-restricted search can serve.
    leakage           fraction of test spectra whose structure is already in the training split, by 2D InChIKey
                      (InChIKey14) and by exact SMILES (the two differ wherever stereochemistry
                      or salt form varies).
    fp_oracle         the nearest-neighbour ceiling: per test spectrum, the highest Tanimoto between its true
                      fingerprint and ANY training fingerprint -- the best score reachable by a method that returns a
                      training-set fingerprint. Also on the formula-matched subset and within the same-formula pool.

The fingerprint cosine is reported for every baseline by the baselines themselves (metrics "cosine"), on the raw
prediction.

Outputs: results/nearest_neighbour/<dataset>_<split>/audit_metrics_<fp>.json and, from summarize,
results/nearest_neighbour/audit_metrics_<fp>.md.
"""

import json
import numpy as np
from tqdm import tqdm

from . import nn_lib as nn


def inchikey14(records):
    """2D InChIKey (first block) per record; None when RDKit cannot parse the SMILES."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    cache, out = {}, []
    for r in records:
        s = r["smiles"]
        if s not in cache:
            m = Chem.MolFromSmiles(s)
            cache[s] = Chem.MolToInchiKey(m).split("-")[0] if m is not None else None
        out.append(cache[s])
    return out


def max_tanimoto(test_FP, train_FP, batch=512, rows=None, desc="fp oracle"):
    """Per test row, the highest Tanimoto against any training fingerprint (the retrieval ceiling). `rows` optionally
    restricts the training pool per test row (list of index arrays; NaN for an empty pool). Products in torch (numpy in
    the `mist` env uses reference BLAS)."""
    import torch
    t = torch.from_numpy(np.ascontiguousarray(test_FP, dtype=np.float32))
    tr = torch.from_numpy(np.ascontiguousarray(train_FP, dtype=np.float32))
    tr_pop = tr.sum(dim=1)
    out = np.zeros(t.shape[0], dtype=np.float64)
    if rows is None:
        for s in tqdm(range(0, t.shape[0], batch), desc=desc):
            q = t[s:s + batch]
            inter = q @ tr.T
            union = q.sum(dim=1)[:, None] + tr_pop[None, :] - inter
            out[s:s + batch] = (inter / union.clamp(min=1.0)).max(dim=1).values.numpy()
    else:
        for i in tqdm(range(t.shape[0]), desc=desc):
            pool = rows[i]
            if len(pool) == 0:
                out[i] = np.nan
                continue
            sub = tr[torch.from_numpy(np.asarray(pool, dtype=np.int64))]
            inter = sub @ t[i]
            union = t[i].sum() + sub.sum(dim=1) - inter
            out[i] = float((inter / union.clamp(min=1.0)).max())
    return out


def mean(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    return float(x.mean()) if len(x) else None


def run(dataset, split, fp_def, skip_oracle=False):
    out = nn.RESULTS_FOLDER / f"{dataset}_{split}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n===== audit {dataset} / {split} | fingerprint {fp_def}")
    data = nn.load_dataset(dataset)
    train_ids, val_ids, test_ids = nn.load_split(dataset, split)
    train_recs = [data[i] for i in train_ids]
    test_recs = [data[i] for i in test_ids]

    train_formula = [r["formula"] for r in train_recs]
    train_formula_set = set(train_formula)
    matched = np.array([r["formula"] in train_formula_set for r in test_recs])
    res = {"dataset": dataset, "split": split, "fingerprint": fp_def,
           "n_train": len(train_ids), "n_val": len(val_ids), "n_test": len(test_ids),
           "coverage": float(matched.mean()), "n_matched": int(matched.sum())}

    tr_smiles = set(r["smiles"] for r in train_recs)
    res["leakage_smiles"] = float(np.mean([r["smiles"] in tr_smiles for r in test_recs]))
    tr_key = set(k for k in inchikey14(train_recs) if k)
    te_key = inchikey14(test_recs)
    res["leakage_inchikey14"] = float(np.mean([k in tr_key for k in te_key if k]))
    res["n_unparsed_test"] = int(sum(k is None for k in te_key))
    print(f"coverage {res['coverage']:.3f} | leakage InChIKey14 {res['leakage_inchikey14']:.3f} "
          f"/ exact SMILES {res['leakage_smiles']:.3f}")

    if not skip_oracle:
        fp_cache = {}
        train_FP = nn.fingerprints(train_recs, fp_def, fp_cache)
        test_FP = nn.fingerprints(test_recs, fp_def, fp_cache)
        orc = max_tanimoto(test_FP, train_FP, desc=f"{dataset}/{split} oracle (all train)")
        res["fp_oracle_full"] = mean(orc)
        res["fp_oracle_formula_subset"] = mean(orc[matched])
        groups = {}
        for k, f in enumerate(train_formula):
            groups.setdefault(f, []).append(k)
        pools = [np.array(groups.get(r["formula"], []), dtype=np.int64) for r in test_recs]
        orc_f = max_tanimoto(test_FP, train_FP, rows=pools, desc=f"{dataset}/{split} oracle (same formula)")
        res["fp_oracle_same_formula_pool"] = mean(orc_f[matched])
        print(f"fp oracle: all-train {res['fp_oracle_full']:.3f} | on the formula subset "
              f"{res['fp_oracle_formula_subset']:.3f} | within the same-formula pool "
              f"{res['fp_oracle_same_formula_pool']:.3f}")

    (out / f"audit_metrics_{fp_def}.json").write_text(json.dumps(res, indent=2))
    return res


def summarize(fp_def):
    """Table of every audit_metrics_<fp>.json -> results/nearest_neighbour/audit_metrics_<fp>.md"""
    order = {(d, s): n for n, (d, s) in enumerate((d, s) for d in nn.DATASETS for s in nn.SPLITS)}
    rows = []
    for p in nn.RESULTS_FOLDER.glob(f"*/audit_metrics_{fp_def}.json"):
        m = json.loads(p.read_text())
        d, s = (m["dataset"], m["split"]) if "dataset" in m else p.parent.name.rsplit("_", 1)
        rows.append(((d, s), m))
    rows.sort(key=lambda x: (order.get(x[0], len(order)), x[0]))
    if not rows:
        print("no audit_metrics_*.json found"); return
    f3 = lambda v: "--" if v is None else f"{v:.3f}"
    L = [f"# Split properties, fingerprint = {fp_def}", "",
         "`coverage` = test spectra whose formula occurs in training; `leakage` = test structures already in training; "
         "`FP oracle` = best Tanimoto reachable by returning some training fingerprint (retrieval ceiling).", "",
         "| dataset / split | train | test | coverage | leakage (InChIKey14) | leakage (exact SMILES) | FP oracle, all train | FP oracle, formula subset | FP oracle, same-formula pool |",
         "|---|---|---|---|---|---|---|---|---|"]
    for (d, s), m in rows:
        L.append(f"| {d} / {s} | {m['n_train']} | {m['n_test']} | {m['coverage']:.3f} | {m['leakage_inchikey14']:.3f} | "
                 f"{m['leakage_smiles']:.3f} | {f3(m.get('fp_oracle_full'))} | {f3(m.get('fp_oracle_formula_subset'))} | "
                 f"{f3(m.get('fp_oracle_same_formula_pool'))} |")
    md = "\n".join(L) + "\n"
    (nn.RESULTS_FOLDER / f"audit_metrics_{fp_def}.md").write_text(md)
    print(md)
