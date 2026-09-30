"""
Sub-formula fallback for test spectra without a same-formula training spectrum (settings *_subformula_match and subformula_only of run_nn.py).

Every training / query spectrum is a bag of sub-formula fragment and neutral-loss tokens (nn_lib.subformula_tokens).
Training spectra are scored by

    score(q, t) = s_SF(q, t) + alpha * s_F(q, t; w)

with s_SF the cosine between token vectors and s_F the heteroatom-weighted Bray-Curtis similarity of the element-count
vectors (a heteroatom mismatch counts w times a C/H mismatch). The prediction is the elementwise threshold at 0.5 of the
softmax(T)-weighted average of the fingerprints of the k best-scoring training compounds (k = 1 copies the best match).
(w, alpha, T, k) are selected on the validation spectra that have no same-formula training spectrum (retrieved against
the training split), under the fingerprint definition being evaluated.

The same rule can also be applied to every test spectrum (scope="all", setting subformula_only); (w, alpha, T, k)
are then selected on the whole validation split.

Outputs in results/nearest_neighbour/<dataset>_<split>/ (see PATHS):
    scope "unmatched": subformula_selection_<fp>.json      validation grid and the selected (w, alpha, T, k)
                       subformula_fallback_<fp>.pkl         {"ids", "topk", "weights", "hparams"}: top-k training ids
                                                            and softmax weights
    scope "all":       subformula_all_selection_<fp>.json   likewise, selected on all validation spectra
                       subformula_all_<fp>.pkl              likewise, for every test spectrum
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from . import nn_lib as nn

# Validation grid of the sub-formula fallback
HETW = (1.0, 2.0, 4.0, 8.0, 16.0)
ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
TEMPS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
KS = (1, 3, 5, 10, 20, 50)

def current_grid():
    """The validation grid as it is stored in (and compared against) subformula_selection_<fp>.json."""
    return {"w": list(HETW), "alpha": list(ALPHAS), "T": list(TEMPS), "k": list(KS)}


PATHS = {"unmatched": ("subformula_selection_{fp}.json", "subformula_fallback_{fp}.pkl"),
         "all": ("subformula_all_selection_{fp}.json", "subformula_all_{fp}.pkl")}

def paths(out, fp_def, scope):
    sel, res = PATHS[scope]
    return out / sel.format(fp=fp_def), out / res.format(fp=fp_def)


def load_json(path, default):
    return json.load(open(path)) if Path(path).exists() else default

def dump_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)



def consensus_grid(idx, sc, train_FP, true_FP, batch=256):
    """Mean Jaccard of the softmax(T)-weighted top-k consensus for every (T, k) in TEMPS x KS."""
    n = idx.shape[0]
    sums = {T: {k: 0.0 for k in KS} for T in TEMPS}
    for s in range(0, n, batch):
        fp = train_FP[idx[s:s + batch]].astype(np.float32)
        ss, tf = sc[s:s + batch], true_FP[s:s + batch]
        for T in TEMPS:
            w_all = np.exp((ss - ss[:, :1]) / np.float32(T))
            for k in KS:
                w = w_all[:, :k] / w_all[:, :k].sum(axis=1, keepdims=True)
                sums[T][k] += nn.jaccard(np.einsum("bk,bkc->bc", w, fp[:, :k]) >= 0.5, tf).sum()
    return {T: {k: float(sums[T][k] / n) for k in KS} for T in TEMPS}

def select_hparams(S_sf, val_E, train_E, train_FP, val_FP):
    """Full grid over (w, alpha, T, k) on the validation no-formula-match subset. Returns (best, grid);
    ties -> smaller k, larger T, smaller alpha, smaller w."""
    grid, best, best_key = {}, None, None
    for w in HETW:
        fsim = nn.weighted_formula_sim(val_E, train_E, w)
        grid[w] = {}
        for alpha in tqdm(ALPHAS, desc=f"validation grid (w={w:g})"):
            if alpha == 0.0 and w != HETW[0]:
                grid[w][alpha] = grid[HETW[0]][alpha]      # alpha = 0 does not depend on w
            else:
                idx, sc = nn.topk_sorted(S_sf + np.float32(alpha) * fsim, max(KS))
                grid[w][alpha] = consensus_grid(idx, sc, train_FP, val_FP)
            for T in TEMPS:
                for k in KS:
                    key = (round(grid[w][alpha][T][k], 6), -k, T, -alpha, -w)
                    if best_key is None or key > best_key:
                        best_key, best = key, {"w": w, "alpha": alpha, "T": T, "k": k, "val_jaccard": grid[w][alpha][T][k]}
        del fsim
    return best, grid

def subformula_fallback(data, train_ids, val_ids, test_ids, unmatched_pos, train_FP, fp_def, fp_cache, hparams, n_workers, out,
                        scope="unmatched"):
    """Softmax top-k consensus of the sub-formula rule on test_ids[unmatched_pos]. Returns (ids, predicted fingerprints,
    hparams). Selection on validation unless `hparams` is given: on the validation spectra without a same-formula
    training spectrum (scope "unmatched", the fallback) or on all validation spectra (scope "all")."""
    train_formula = [data[i]["formula"] for i in train_ids]
    train_formula_set = set(train_formula)
    fb_test = [test_ids[p] for p in unmatched_pos]
    train_tokens = nn.tokenize(data, train_ids, n_workers)
    train_M, vocab = nn.build_token_matrix(train_tokens)
    train_E = nn.formula_vectors(train_formula)

    p_sel, p_res = paths(out, fp_def, scope)
    if hparams is None:
        sel = load_json(p_sel, None)
        if sel is not None and sel.get("grid") != current_grid():
            print(f"{p_sel.name} was selected on a different grid ({sel.get('grid')}); re-running the selection "
                  f"on {current_grid()}")
            sel = None
        if sel is None:
            if scope == "all":
                fb_val = list(val_ids)
                print(f"validation: selection on all {len(fb_val)} spectra")
            else:
                fb_val = [i for i in val_ids if data[i]["formula"] not in train_formula_set]
                print(f"validation: {len(fb_val)} of {len(val_ids)} spectra have no same-formula training spectrum")
            val_M, _ = nn.build_token_matrix(nn.tokenize(data, fb_val, n_workers), vocab)
            S_sf = (val_M @ train_M.T).toarray().astype(np.float32)
            val_E = nn.formula_vectors([data[i]["formula"] for i in fb_val])
            val_FP = nn.fingerprints([data[i] for i in fb_val], fp_def, fp_cache)
            best, grid = select_hparams(S_sf, val_E, train_E, train_FP, val_FP)
            del S_sf
            sel = {"fingerprint": fp_def, "scope": scope, "n_val": len(val_ids), "n_val_used": len(fb_val),
                   "grid": current_grid(), "selected": best,
                   "val_grid": {str(w): {str(a): {str(T): {str(k): grid[w][a][T][k] for k in KS} for T in TEMPS} for a in ALPHAS} for w in HETW}}
            dump_json(sel, p_sel)
        hparams = {k: sel["selected"][k] for k in ("w", "alpha", "T", "k")}
        hparams["selection"] = "validation"
    else:
        hparams = dict(hparams, selection="given")
    print("sub-formula fallback hyperparameters:", hparams)

    test_M, _ = nn.build_token_matrix(nn.tokenize(data, fb_test, n_workers), vocab)
    S_sf = (test_M @ train_M.T).toarray().astype(np.float32)
    test_E = nn.formula_vectors([data[i]["formula"] for i in fb_test])
    score = S_sf + np.float32(hparams["alpha"]) * nn.weighted_formula_sim(test_E, train_E, hparams["w"])
    del S_sf
    idx, sc = nn.topk_sorted(score, hparams["k"])
    del score
    pred = nn.softmax_consensus(idx, sc, train_FP, hparams["T"], hparams["k"])
    nn.save_pickle({"ids": fb_test, "topk": [[train_ids[t] for t in row] for row in idx],
                    "weights": nn.softmax_weights(sc, hparams["T"], hparams["k"]), "hparams": hparams},
                   p_res)
    return fb_test, pred, hparams


def load_or_run(data, train_ids, val_ids, test_ids, unmatched_pos, train_FP, fp_def, fp_cache, n_workers, out,
                scope="unmatched"):
    """The validation-selected sub-formula fallback for test_ids[unmatched_pos]: {"ids", "topk", "weights",
    "hparams"} with topk as training ids, best first, and the softmax weights of the consensus.
    out/subformula_fallback_<fp>.pkl (written by run_nn.py or by a previous call) is reused when it covers exactly these
    test spectra and its hyperparameters are the current validation selection; otherwise the fallback is recomputed
    (tokenisation of the training split takes a few minutes on MassSpecGym)."""
    fb_test = [test_ids[p] for p in unmatched_pos]
    p_sel, path = paths(out, fp_def, scope)
    sel = load_json(p_sel, None)
    if path.exists() and sel is not None and sel.get("grid") == current_grid():
        fb = nn.load_pickle(path)
        selected = {k: sel["selected"][k] for k in ("w", "alpha", "T", "k")}
        if fb["ids"] == fb_test and fb["hparams"].get("selection") == "validation" and \
                all(fb["hparams"][k] == v for k, v in selected.items()):
            print(f"reusing {path} ({fb['hparams']})")
            return fb
    subformula_fallback(data, train_ids, val_ids, test_ids, unmatched_pos, train_FP, fp_def, fp_cache, None, n_workers, out,
                        scope)
    return nn.load_pickle(path)
