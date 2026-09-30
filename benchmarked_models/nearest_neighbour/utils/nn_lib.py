"""
Library for the nearest-neighbour baselines (used by `run_nn.py` through utils/nn_driver.py).

Contents
    data          load_dataset / load_split / load_dreams (order-checked against the MGF ids) / load_mist_jaccard
    fingerprints  dataset_fps (the `morgan4_4096` bits shipped with the data) and mist_fps (RDKit Morgan r=2,
                  4096 bits from SMILES, the definition MIST is trained and scored on)
    spectra       bin_spectrum (0.25 Da bins, precursor added), l2_normalise, cosine_argmax
    sub-formula   annotate_peaks / subformula_tokens / build_token_matrix: every peak is explained by a sub-formula
                  of the precursor ion formula; a spectrum becomes a bag of fragment + neutral-loss formula tokens
    formula       formula_vectors / weighted_formula_sim: heteroatom-weighted Bray-Curtis similarity
    consensus     topk_sorted / softmax_consensus: softmax-weighted average of the top-k fingerprints
    metric        jaccard, cosine
"""

import re
import math
import json
import pickle
import itertools
import numpy as np
from pathlib import Path
from scipy import sparse

ROOT = Path(__file__).resolve().parents[3]
DATA_FOLDER = ROOT / "data" / "processed_data"
SPLITS_FOLDER = ROOT / "data" / "splits"
MGF_FOLDER = ROOT / "data" / "MGF_files"
RESULTS_FOLDER = ROOT / "results" / "nearest_neighbour"
DREAMS_FOLDER = RESULTS_FOLDER / "DreaMS_emb"
MIST_FOLDER = ROOT / "results" / "FP_prediction" / "mist"
MIST_TAG = {"NPLIB1": "NPLIB1", "massspecgym": "MSG"}

DATASETS = ("NPLIB1", "massspecgym")
SPLITS = ("scaffold", "random")

# --------------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------------

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

# Datasets outside data/processed_data + data/splits, registered with register_dataset (run_nn.py --data_file)
CUSTOM_DATASETS = {}

def register_dataset(name, data_file, split_dir):
    """Use `data_file` (a pickled list of records, see load_dataset) and `split_dir`/<split>.json for dataset `name`."""
    CUSTOM_DATASETS[name] = (Path(data_file), Path(split_dir))

def data_path(dataset):
    return CUSTOM_DATASETS[dataset][0] if dataset in CUSTOM_DATASETS else DATA_FOLDER / f"{dataset}.pkl"

def split_path(dataset, split):
    return (CUSTOM_DATASETS[dataset][1] if dataset in CUSTOM_DATASETS else SPLITS_FOLDER / dataset) / f"{split}.json"

_DATA_CACHE = {}

def load_dataset(dataset):
    """id -> record, from a pickled list of records (cached per process: MassSpecGym takes ~1 min to load). Records
    carry `id_`, `peaks` ([{mz, intensity_norm}]), `precursor_MZ_final`, `formula` (neutral; formula filter),
    `formula_corrected` (precursor ion formula; sub-formula rule), `smiles` (fingerprint "mist") and, for fingerprint
    "dataset", `FPs["morgan4_4096"]`."""
    path = data_path(dataset)
    if path not in _DATA_CACHE:
        _DATA_CACHE.clear()
        _DATA_CACHE[path] = {str(r["id_"]): r for r in load_pickle(path)}
    return _DATA_CACHE[path]

def load_split(dataset, split):
    """(train_ids, val_ids, test_ids) as strings, from a json {"train": [...], "val": [...], "test": [...]} of record ids
    (a trailing ".pkl" is stripped); the three lists are disjoint."""
    js = json.load(open(split_path(dataset, split)))
    parts = [[t.replace(".pkl", "") for t in js[k]] for k in ("train", "val", "test")]
    assert not (set(parts[0]) & set(parts[2])) and not (set(parts[1]) & set(parts[2])) and not (set(parts[0]) & set(parts[1]))
    return parts

def mgf_ids(path):
    ids = []
    with open(path) as f:
        for line in f:
            if line.startswith("ID_="):
                ids.append(line[4:].strip())
    return ids

def load_dreams(dataset, split, part, expected_ids):
    """DreaMS embeddings (from utils/cache_dreams_embeddings.py) for `part` in {train, test}, as an (n x d) float32 array.
    The embeddings were computed from data/MGF_files/{dataset}/{split}/{part}.mgf; the spectrum ids in that file are
    checked against `expected_ids` so that row i is guaranteed to be the embedding of expected_ids[i]."""
    emb = np.asarray(load_pickle(DREAMS_FOLDER / dataset / split / f"{part}.pkl"), dtype=np.float32)
    ids = mgf_ids(MGF_FOLDER / dataset / split / f"{part}.mgf")
    assert ids == list(expected_ids), f"MGF order differs from the split for {dataset}/{split}/{part}"
    assert emb.shape[0] == len(ids)
    return emb

def load_mist_jaccard(dataset, split):
    """test id -> Jaccard of MIST's prediction (thresholded at 0.5) against the RDKit Morgan r=2/4096 ground truth,
    as stored by the MIST evaluation. None when the run does not exist."""
    path = MIST_FOLDER / dataset / f"{MIST_TAG.get(dataset, dataset)}_MIST_4096_{split}" / "test_results.pkl"
    if not path.exists():
        return None
    return {str(k): float(v["jaccard"]) for k, v in load_pickle(path).items()}

def load_mist_cosine(dataset, split):
    """test id -> cosine similarity between MIST's raw predicted probabilities (no threshold) and the stored binary
    ground truth. None when the run does not exist."""
    path = MIST_FOLDER / dataset / f"{MIST_TAG.get(dataset, dataset)}_MIST_4096_{split}" / "test_results.pkl"
    if not path.exists():
        return None
    res = load_pickle(path)
    ids = [str(k) for k in res]
    pred = np.array([np.asarray(v["pred"], dtype=np.float64) for v in res.values()])
    true = np.array([np.asarray(v["GT"], dtype=np.float64) for v in res.values()])
    return dict(zip(ids, map(float, cosine(pred, true))))

# --------------------------------------------------------------------------------------------------
# Fingerprints and metric
# --------------------------------------------------------------------------------------------------

def dataset_fps(records):
    """The `morgan4_4096` bit strings shipped with the processed data (Morgan r=2 on the kekulised molecule)."""
    return np.array([[int(c) for c in r["FPs"]["morgan4_4096"]] for r in records], dtype=np.uint8)

def mist_fps(records, cache=None):
    """RDKit Morgan radius 2, 4096 bits, from SMILES: the definition MIST is trained and evaluated on (its stored
    ground truth matches this bit for bit). Cached per SMILES."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    cache = {} if cache is None else cache
    out = np.empty((len(records), 4096), dtype=np.uint8)
    for i, r in enumerate(records):
        s = r["smiles"]
        if s not in cache:
            cache[s] = gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)).astype(np.uint8)
        out[i] = cache[s]
    return out

def fingerprints(records, definition, cache=None):
    if definition == "mist":
        return mist_fps(records, cache)
    if definition == "dataset":
        return dataset_fps(records)
    raise ValueError(definition)

def jaccard(pred, true):
    """Row-wise Tanimoto / Jaccard between binary fingerprint matrices."""
    pred = np.asarray(pred).astype(bool); true = np.asarray(true).astype(bool)
    return np.logical_and(pred, true).sum(axis=1) / np.maximum(np.logical_or(pred, true).sum(axis=1), 1)

def cosine(pred, true):
    """Row-wise cosine similarity <p, t> / (|p| |t|) between (continuous or binary) prediction and target matrices, used
    on the raw prediction, without threshold; for binary vectors it is |A & B| / sqrt(|A| |B|). 0 when either is zero."""
    pred = np.asarray(pred, dtype=np.float64); true = np.asarray(true, dtype=np.float64)
    denom = np.linalg.norm(pred, axis=1) * np.linalg.norm(true, axis=1)
    return (pred * true).sum(axis=1) / np.where(denom > 0, denom, 1.0)

# --------------------------------------------------------------------------------------------------
# Binned spectra and cosine similarity
# --------------------------------------------------------------------------------------------------

def bin_spectrum(rec, bin_resolution=0.25, max_da=2000.0):
    """0.25 Da binned intensity vector (intensity_norm summed per bin), with the precursor m/z added as a peak of
    intensity 100, as in the original nearest-neighbour baseline."""
    n_bins = int(math.ceil(max_da / bin_resolution))
    inv = 1.0 / bin_resolution
    out = np.zeros(n_bins, dtype=np.float32)
    for p in rec["peaks"] + [{"mz": rec["precursor_MZ_final"], "intensity_norm": 100}]:
        b = math.floor(p["mz"] * inv)
        if 0 <= b < n_bins:
            out[b] += p["intensity_norm"]
    return out

def l2_normalise(X):
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def cosine_argmax(Q, X, batch=512):
    """Row-wise argmax of Q @ X.T for L2-normalised Q (n x d) and X (m x d), in row batches."""
    out = np.empty(Q.shape[0], dtype=np.int64)
    for s in range(0, Q.shape[0], batch):
        out[s:s + batch] = (Q[s:s + batch] @ X.T).argmax(axis=1)
    return out

def cosine_topk(Q, X, k, batch=512):
    """Indices and values of the k largest entries of each row of Q @ X.T (L2-normalised rows), best first, in row
    batches. Column 0 is the argmax (first index among exact ties), so top-1 agrees with cosine_argmax.
    The products run in torch on CPU: numpy in the `mist` env is linked against reference BLAS (single-threaded,
    ~1 GFLOP/s), which makes a MassSpecGym search take ~25 min instead of ~1 min."""
    import torch
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    Qt = torch.from_numpy(np.ascontiguousarray(Q, dtype=np.float32))
    idx = np.empty((Q.shape[0], k), dtype=np.int64)
    val = np.empty((Q.shape[0], k), dtype=np.float32)
    for s in range(0, Q.shape[0], batch):
        S = Qt[s:s + batch] @ Xt.T
        v, i = torch.topk(S, k, dim=1)
        v, i = v.numpy(), i.numpy()
        first = S.argmax(dim=1).numpy()   # torch returns the first maximal index
        for r in np.where(i[:, 0] != first)[0]:   # exact ties at the top: put the first maximal index in front
            row = list(i[r])
            if first[r] in row:
                row.remove(first[r])
            else:
                row.pop()
            i[r] = [first[r]] + row
        idx[s:s + batch], val[s:s + batch] = i, v
    return idx, val

# --------------------------------------------------------------------------------------------------
# Sub-formula annotation and bag-of-tokens representation
# --------------------------------------------------------------------------------------------------

MONO_MASS = {
    "H": 1.00782503207, "D": 2.01410177812, "He": 4.00260325415, "Li": 7.01600455, "Be": 9.0121822,
    "B": 11.0093054, "C": 12.0, "N": 14.0030740048, "O": 15.99491461956, "F": 18.99840322,
    "Ne": 19.9924401754, "Na": 22.9897692809, "Mg": 23.985041700, "Al": 26.98153863,
    "Si": 27.9769265325, "P": 30.97376163, "S": 31.97207100, "Cl": 34.96885268, "Ar": 39.9623831225,
    "K": 38.96370668, "Ca": 39.96259098, "Fe": 55.9349375, "Zn": 63.9291422, "As": 74.9215965,
    "Se": 79.9165213, "Br": 78.9183371, "I": 126.904473,
}
ELECTRON_MASS = 0.00054857990946
ELEMENTS = sorted(MONO_MASS)
ELEMENT_IDX = {e: i for i, e in enumerate(ELEMENTS)}
HETEROATOM = np.array([e not in ("C", "H") for e in ELEMENTS])
HALOGENS = ("F", "Cl", "Br", "I")
FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")

def parse_formula(formula):
    """'C6H12O6' -> {'C': 6, 'H': 12, 'O': 6}; charge signs are ignored."""
    counts = {}
    for element, count in FORMULA_RE.findall(formula):
        if element:
            counts[element] = counts.get(element, 0) + (int(count) if count else 1)
    return counts

def check_known_elements(counts, formula=None):
    """Raise a readable error for elements this module has no monoisotopic mass for. `ELEMENTS` (and with it the
    element-count vectors and the sub-formula enumeration) is derived from MONO_MASS, so an unlisted element --
    a metal, say -- would otherwise surface as a bare KeyError deep inside the annotation."""
    unknown = sorted(e for e in counts if e not in MONO_MASS)
    if unknown:
        where = f" in formula {formula!r}" if formula is not None else ""
        raise ValueError(f"nn_lib has no monoisotopic mass for {', '.join(unknown)}{where}. Add the element(s) to "
                         f"nn_lib.MONO_MASS (ELEMENTS and ELEMENT_IDX are derived from it) to use this dataset.")

def counts_to_string(counts):
    """Hill-order string of an element-count dict."""
    parts = []
    for e in ["C", "H"] + [x for x in sorted(counts) if x not in ("C", "H")]:
        c = counts.get(e, 0)
        if c > 0:
            parts.append(e if c == 1 else f"{e}{c}")
    return "".join(parts)

def formula_vectors(formulas):
    """(n x len(ELEMENTS)) int32 element-count matrix."""
    E = np.zeros((len(formulas), len(ELEMENTS)), dtype=np.int32)
    for i, f in enumerate(formulas):
        counts = parse_formula(f)
        check_known_elements(counts, f)
        for e, c in counts.items():
            E[i, ELEMENT_IDX[e]] = c
    return E

def weighted_formula_sim(test_E, train_E, hetero_weight, batch=64):
    """Heteroatom-weighted Bray-Curtis similarity of element-count vectors, in [0, 1]:
        s_F(q, t) = 1 - sum_e w_e |q_e - t_e| / sum_e w_e (q_e + t_e),   w_e = hetero_weight for e not in {C, H}, else 1.
    hetero_weight = 1 is the plain Bray-Curtis similarity; 1 iff the formulas are identical."""
    w = np.ones(len(ELEMENTS), dtype=np.float32)
    w[HETEROATOM] = hetero_weight
    out = np.empty((test_E.shape[0], train_E.shape[0]), dtype=np.float32)
    test_wsum = (test_E * w).sum(axis=1).astype(np.float32)
    train_wsum = (train_E * w).sum(axis=1).astype(np.float32)
    for s in range(0, test_E.shape[0], batch):
        e = min(s + batch, test_E.shape[0])
        d = (np.abs(test_E[s:e, None, :] - train_E[None, :, :]) * w[None, None, :]).sum(axis=2)
        out[s:e] = 1.0 - d / (test_wsum[s:e, None] + train_wsum[None, :])
    return out

_HEAVY_CACHE = {}

def heavy_subformulas(parent_counts):
    """All sub-formulas of the parent's heavy (non-H) atoms: (elements, count matrix, sorted monoisotopic masses)."""
    key = counts_to_string(parent_counts)
    if key in _HEAVY_CACHE:
        return _HEAVY_CACHE[key]
    check_known_elements(parent_counts, key)
    elements = [e for e in parent_counts if e != "H" and parent_counts[e] > 0]
    if elements:
        combos = np.array(list(itertools.product(*[range(parent_counts[e] + 1) for e in elements])), dtype=np.int16)
        masses = combos.astype(np.float64) @ np.array([MONO_MASS[e] for e in elements], dtype=np.float64)
    else:
        combos, masses = np.zeros((1, 0), dtype=np.int16), np.zeros(1)
    order = np.argsort(masses)
    out = (elements, combos[order], masses[order])
    _HEAVY_CACHE[key] = out
    return out

def annotate_peaks(mz_array, parent_counts, ppm=20.0, min_da=0.01):
    """For every peak m/z, the sub-formula of the precursor ion formula whose (singly charged, positive) ion mass is
    closest to it within max(min_da, ppm): heavy atoms are enumerated, the hydrogen count is solved from the mass
    residual, and a loose valence check (2 * RDBE >= -1) is applied. None when no sub-formula explains the peak."""
    elements, combos, heavy_masses = heavy_subformulas(parent_counts)
    H_parent = parent_counts.get("H", 0)
    mH = MONO_MASS["H"]
    c_idx = elements.index("C") if "C" in elements else None
    n_idx = elements.index("N") if "N" in elements else None
    p_idx = elements.index("P") if "P" in elements else None
    x_idx = [elements.index(x) for x in HALOGENS if x in elements]
    out = []
    for mz in mz_array:
        target = mz + ELECTRON_MASS
        tol = max(min_da, ppm * 1e-6 * target)
        lo = np.searchsorted(heavy_masses, target - H_parent * mH - tol)
        hi = np.searchsorted(heavy_masses, target + tol)
        if hi <= lo:
            out.append(None); continue
        hm = heavy_masses[lo:hi]
        H = np.rint((target - hm) / mH).astype(np.int32)
        err = np.abs(target - hm - H * mH)
        ok = (H >= 0) & (H <= H_parent) & (err <= tol)
        if ok.any():
            cand = combos[lo:hi]
            C = cand[:, c_idx].astype(np.int32) if c_idx is not None else 0
            N = cand[:, n_idx].astype(np.int32) if n_idx is not None else 0
            P = cand[:, p_idx].astype(np.int32) if p_idx is not None else 0
            X = sum(cand[:, i].astype(np.int32) for i in x_idx) if x_idx else 0
            ok &= (2 * C + N + P + 2 - (H + X)) >= -1
        if not ok.any():
            out.append(None); continue
        best = np.where(ok)[0][np.argmin(err[ok])]
        counts = {e: int(c) for e, c in zip(elements, combos[lo + best]) if c > 0}
        if H[best] > 0:
            counts["H"] = int(H[best])
        out.append(counts)
    return out

def subformula_tokens(rec, top_k=100, ppm=20.0, min_da=0.01):
    """{token: weight}: a fragment token 'F:<sub-formula>' and a neutral-loss token 'L:<ion formula - sub-formula>'
    per annotated peak (the top_k most intense), weighted by sqrt(intensity_norm) (max over repeats)."""
    parent = parse_formula(rec.get("formula_corrected") or rec["formula"])
    peaks = sorted(rec["peaks"], key=lambda p: -p["intensity_norm"])[:top_k]
    annotations = annotate_peaks(np.array([p["mz"] for p in peaks], dtype=np.float64), parent, ppm=ppm, min_da=min_da)
    tokens = {}
    for p, ann in zip(peaks, annotations):
        if ann is None:
            continue
        w = math.sqrt(max(p["intensity_norm"], 0.0))
        frag = "F:" + counts_to_string(ann)
        loss_counts = {e: parent.get(e, 0) - ann.get(e, 0) for e in parent}
        loss = "L:" + (counts_to_string(loss_counts) if any(v > 0 for v in loss_counts.values()) else "none")
        tokens[frag] = max(tokens.get(frag, 0.0), w)
        tokens[loss] = max(tokens.get(loss, 0.0), w)
    return tokens

_G = {}   # records visible to forked Pool workers without pickling

def _token_worker(id_):
    """(tokens, error): a failure is reported back rather than raised, so one bad spectrum cannot kill the pool."""
    try:
        return subformula_tokens(_G["data"][id_]), None
    except Exception as e:
        return {}, f"{type(e).__name__}: {e}"

def tokenize(data, ids, n_workers):
    """Sub-formula token dicts for `ids` (multiprocessing over forked workers). A spectrum that fails to annotate
    yields an empty bag, which the fallback then scores by the formula term alone -- so failures are counted and
    reported here instead of degrading retrieval silently."""
    from multiprocessing import Pool
    from collections import Counter
    from tqdm import tqdm
    _G["data"] = data
    with Pool(n_workers) as pool:
        out = list(tqdm(pool.imap(_token_worker, ids, chunksize=64), total=len(ids), desc="sub-formula tokens"))
    tokens = [tok for tok, _ in out]
    errors = Counter(err for _, err in out if err is not None)
    if errors:
        print(f"WARNING: sub-formula tokenisation raised for {sum(errors.values())} of {len(ids)} spectra "
              f"(empty token bag, scored by the formula term alone); most common:")
        for msg, c in errors.most_common(3):
            print(f"    {c:6d}  {msg}")
    n_empty = sum(1 for tok in tokens if not tok) - sum(errors.values())
    if n_empty:
        print(f"note: {n_empty} of {len(ids)} spectra have no annotatable peak (empty token bag, no error)")
    return tokens

def build_token_matrix(token_dicts, vocab=None):
    """Sparse CSR matrix of L2-normalised token rows. The vocabulary is built from `token_dicts` when not given
    (training set); tokens outside the vocabulary are dropped."""
    if vocab is None:
        vocab = {}
        for d in token_dicts:
            for t in d:
                if t not in vocab:
                    vocab[t] = len(vocab)
    rows, cols, vals = [], [], []
    for i, d in enumerate(token_dicts):
        for t, w in d.items():
            j = vocab.get(t)
            if j is not None:
                rows.append(i); cols.append(j); vals.append(w)
    M = sparse.csr_matrix((vals, (rows, cols)), shape=(len(token_dicts), len(vocab)), dtype=np.float32)
    norms = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return (sparse.diags(1.0 / norms) @ M).tocsr(), vocab

# --------------------------------------------------------------------------------------------------
# Top-k consensus
# --------------------------------------------------------------------------------------------------

def topk_sorted(score, K, batch=512):
    """Per row of `score`: indices and values of the K largest entries, best first."""
    n = score.shape[0]
    idx = np.empty((n, K), dtype=np.int64)
    sc = np.empty((n, K), dtype=np.float32)
    for s in range(0, n, batch):
        Sb = score[s:s + batch]
        part = np.argpartition(-Sb, K - 1, axis=1)[:, :K]
        vals = np.take_along_axis(Sb, part, axis=1)
        order = np.argsort(-vals, axis=1)
        idx[s:s + batch] = np.take_along_axis(part, order, axis=1)
        sc[s:s + batch] = np.take_along_axis(vals, order, axis=1)
    return idx, sc

def softmax_weights(sc, T, k):
    """w_i proportional to exp((s_i - s_1) / T) over the k best scores (rows of `sc` are sorted best first)."""
    w = np.exp((sc[:, :k] - sc[:, :1]) / np.float32(T))
    return w / w.sum(axis=1, keepdims=True)

def softmax_consensus(idx, sc, train_FP, T, k, batch=256):
    """Prediction = elementwise threshold at 0.5 of the softmax(T)-weighted average of the top-k training fingerprints.
    k = 1 copies the fingerprint of the best match."""
    n = idx.shape[0]
    pred = np.empty((n, train_FP.shape[1]), dtype=np.uint8)
    for s in range(0, n, batch):
        w = softmax_weights(sc[s:s + batch], T, k)
        fp = train_FP[idx[s:s + batch, :k]].astype(np.float32)
        pred[s:s + batch] = (np.einsum("bk,bkc->bc", w, fp) >= 0.5)
    return pred
