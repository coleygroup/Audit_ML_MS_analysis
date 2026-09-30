"""
Retrieval and evaluation of one nearest-neighbour setting on one dataset / split, and the summary tables (called by
run_nn.py, which lists the settings: setting <sim>_<mode>, or subformula_only).

For every test spectrum training spectra are scored with one similarity and the fingerprint of the best-scoring
training compound is the prediction (with the sub-formula rule, the softmax-weighted consensus of the top-k
fingerprints). Retrieval is always against the training split.

Similarities
    cosine / modified_cosine / neutral_loss   peak matching, matchms CosineGreedy / ModifiedCosine / NeutralLossesCosine
                                              (utils.peak_similarity; checked against matchms before every full search)
    dreams                                    cosine between DreaMS embeddings (utils/cache_dreams_embeddings.py)
    binned                                    cosine between 0.25 Da binned spectra with the precursor peak (as run_nn.py)

Modes
    no_formula_filter    the whole training split is searched; the formula is never read.
    with_formula_filter  when a training spectrum has the test spectrum's neutral formula, only the training spectra
                         with that formula are searched; otherwise the whole training split is searched with the same
                         similarity, i.e. the no_formula_filter result is used.
    subformula_match     formula-matched test spectra as in with_formula_filter (same-formula search with the chosen
                         similarity); the others get the validation-selected sub-formula fallback of
                         utils.subformula_fallback (top-k softmax consensus of fingerprints, scored by sub-formula token
                         cosine + alpha * heteroatom-weighted formula similarity), which uses no spectral similarity and
                         is therefore shared by all five similarities.
    subformula_only      every test spectrum gets the sub-formula rule of subformula_match (no hard formula filter;
                         the formula enters only through the alpha x formula-similarity term), with (w, alpha, T, k)
                         selected on all validation spectra (utils.subformula_fallback, scope "all"); sim = "subformula".

The formula-matched / unmatched subsets (a training spectrum with the same neutral formula exists or not) are reported
in both modes; in no_formula_filter mode the formula is used for that report only.

Outputs, per dataset / split, in results/nearest_neighbour/<dataset>_<split>/ (<mode> as above):
    <mode>_<sim>.pkl              {"params", "ids", "train_ids", "top", "topk_ids", "topk_scores", "topk_matches"
                                  (None for dreams / binned)} and, for with_formula_filter, "matched" (bool per test id):
                                  retrieval for every test spectrum (independent of the fingerprint). subformula_match
                                  writes no retrieval file of its own: it combines with_formula_filter_<sim>.pkl with
                                  subformula_fallback_<fp>.pkl (see utils.subformula_fallback)
    <mode>_jaccard_<fp>.pkl       {sim: {test id: Jaccard}}
    <mode>_metrics_<fp>.json      {sim: numbers}: full test set, formula-matched / unmatched subsets, MIST-common spectra
and, for --fingerprint mist, the predictions in the layout of the trained models under
results/FP_prediction/nearest_neighbour/<dataset>/<TAG>_NN-<sim>_<mode>_4096_<split>/.
"""

import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

from . import nn_lib as nn
from . import peak_similarity as ps
from . import subformula_fallback as sf

MODES = ("no_formula_filter", "with_formula_filter", "subformula_match", "subformula_only")
EMBEDDINGS = {"dreams": "DreaMS ssl_model.ckpt (cache_dreams_embeddings.py)",
              "binned": "nn_lib.bin_spectrum: 0.25 Da bins up to 2000 Da, precursor added with intensity 100"}
SIMILARITIES = ps.SIMILARITIES + tuple(EMBEDDINGS)
DEFAULTS = {"tolerance": 0.1, "mz_power": 0.0, "intensity_power": 1.0, "max_peaks": 500, "topk": 10}


def load_json(path, default):
    return json.load(open(path)) if Path(path).exists() else default

def dump_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def mean_on(j_by_id, ids):
    vals = [j_by_id[i] for i in ids if i in j_by_id]
    return (float(np.mean(vals)) if vals else None), len(vals)

def sim_params(sim, params):
    """The parameters that determine the retrieval of `sim` (the peak-matching ones do not apply to embeddings)."""
    return {"embedding": EMBEDDINGS[sim], "topk": params["topk"]} if sim in EMBEDDINGS else dict(params)

def load_cache(path, params, train_ids, test_ids):
    """A cached retrieval, only when it was computed with the same parameters and the same train / test ids."""
    if path.exists():
        res = nn.load_pickle(path)
        if res["params"] == params and res["ids"] == list(test_ids) and res.get("train_ids") == list(train_ids):
            print(f"reusing {path}")
            return res
        print(f"{path} was computed with other parameters or ids; recomputing")
    return None


# --------------------------------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------------------------------

def embedding_features(sim, data, train_ids, test_ids, dataset, split):
    """L2-normalised (train, test) vectors: cached DreaMS embeddings (rows checked against the split order by
    nn.load_dreams) or the 0.25 Da binned spectra of run_nn.py."""
    if sim == "dreams":
        return (nn.l2_normalise(nn.load_dreams(dataset, split, "train", train_ids)),
                nn.l2_normalise(nn.load_dreams(dataset, split, "test", test_ids)))
    return (nn.l2_normalise(np.stack([nn.bin_spectrum(data[i]) for i in train_ids])),
            nn.l2_normalise(np.stack([nn.bin_spectrum(data[i]) for i in test_ids])))

def search(sim, data, train_ids, test_ids, params, dataset, split, candidates=None, n_check=0, query_pos=None):
    """(idx, score, n_matches or None), each (n_query x topk), best first, idx into train_ids (-1 = empty slot).
    The queries are test_ids, or test_ids[query_pos] when query_pos is given (test_ids must then still be the whole
    test split, in split order, which the cached DreaMS embeddings are checked against).
    candidates: None (whole training split) or, per query, the sorted training indices to search."""
    k = params["topk"]
    if query_pos is None:
        query_pos = np.arange(len(test_ids))
    if sim in EMBEDDINGS:
        train_X, test_X = embedding_features(sim, data, train_ids, test_ids, dataset, split)
        test_X = test_X[query_pos]
        if candidates is None:
            idx, score = nn.cosine_topk(test_X, train_X, k)
            return idx, score, None
        idx = np.full((len(query_pos), k), -1, dtype=np.int64)
        score = np.full((len(query_pos), k), -1.0, dtype=np.float32)
        groups = defaultdict(list)             # queries sharing a candidate set are scored together
        for q, c in enumerate(candidates):
            groups[tuple(c)].append(q)
        for c, qs in groups.items():
            c = np.array(c, dtype=np.int64); kk = min(k, len(c))
            i, v = nn.cosine_topk(test_X[qs], train_X[c], kk)
            idx[np.ix_(qs, range(kk))] = c[i]; score[np.ix_(qs, range(kk))] = v
        return idx, score, None
    train_rec = [data[i] for i in train_ids]
    test_rec = [data[test_ids[p]] for p in query_pos]
    if n_check:
        chk = ps.check_against_matchms(train_rec, test_rec, sim, params["max_peaks"], params["tolerance"],
                                       params["mz_power"], params["intensity_power"], n_pairs=n_check)
        print(f"matchms check passed: {chk['n_pairs']} pairs ({chk['n_nonzero']} non-zero), max |diff| {chk['max_abs_diff']:.1e}")
    return ps.search(ps.pack_spectra(train_rec, sim, params["max_peaks"]), ps.pack_spectra(test_rec, sim, params["max_peaks"]),
                     sim, params["tolerance"], params["mz_power"], params["intensity_power"], k, candidates=candidates)

def pack_result(params, train_ids, test_ids, idx, score, matches, **extra):
    return {"params": params, "ids": list(test_ids), "train_ids": list(train_ids),
            "top": [train_ids[k] for k in idx[:, 0]],
            "topk_ids": [[train_ids[k] for k in row if k >= 0] for row in idx],
            "topk_scores": score.astype(np.float32),
            "topk_matches": None if matches is None else matches.astype(np.int32), **extra}

def retrieve_no_filter(sim, data, train_ids, test_ids, params, dataset, split, n_check, out):
    """Whole-training-split search, cached in out/no_formula_filter_<sim>.pkl."""
    params = sim_params(sim, params)
    path = out / f"no_formula_filter_{sim}.pkl"
    res = load_cache(path, params, train_ids, test_ids)
    if res is not None:
        return res
    t0 = time.time()
    idx, score, matches = search(sim, data, train_ids, test_ids, params, dataset, split, n_check=n_check)
    print(f"{sim}: {len(test_ids)} x {len(train_ids)} pairs in {time.time() - t0:.0f} s")
    res = pack_result(params, train_ids, test_ids, idx, score, matches)
    nn.save_pickle(res, path)
    return res

def retrieve_with_filter(sim, data, train_ids, test_ids, params, dataset, split, n_check, out):
    """Same-formula search for formula-matched test spectra, whole-training-split search (the no_formula_filter
    retrieval, reused from its cache) for the others; cached in out/with_formula_filter_<sim>.pkl."""
    params = sim_params(sim, params)
    path = out / f"with_formula_filter_{sim}.pkl"
    res = load_cache(path, params, train_ids, test_ids)
    if res is not None:
        return res
    full = retrieve_no_filter(sim, data, train_ids, test_ids, params, dataset, split, n_check, out)
    groups = defaultdict(list)
    for k, i in enumerate(train_ids):
        groups[data[i]["formula"]].append(k)
    matched = np.array([data[i]["formula"] in groups for i in test_ids])
    m_pos = np.where(matched)[0]
    t0 = time.time()
    m_idx, m_score, m_matches = search(sim, data, train_ids, test_ids, params, dataset, split, query_pos=m_pos,
                                       candidates=[groups[data[test_ids[p]]["formula"]] for p in m_pos])
    print(f"{sim}: same-formula search for {len(m_pos)} formula-matched test spectra in {time.time() - t0:.0f} s")

    # Unmatched rows come from the whole-split search
    train_pos = {i: k for k, i in enumerate(train_ids)}
    k = params["topk"]
    idx = np.full((len(test_ids), k), -1, dtype=np.int64)
    score = np.array(full["topk_scores"], dtype=np.float32, copy=True)
    for q, row in enumerate(full["topk_ids"]):
        idx[q, :len(row)] = [train_pos[t] for t in row]
    matches = None if full["topk_matches"] is None else np.array(full["topk_matches"], copy=True)
    idx[m_pos], score[m_pos] = m_idx, m_score
    if matches is not None:
        matches[m_pos] = m_matches
    res = pack_result(params, train_ids, test_ids, idx, score, matches, matched=matched.tolist())
    nn.save_pickle(res, path)
    return res


# --------------------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------------------

def retrieve_subformula_match(sim, data, train_ids, val_ids, test_ids, params, dataset, split, n_check, out,
                              train_FP, fp_def, fp_cache, n_workers):
    """with_formula_filter retrieval for the formula-matched test spectra and the sub-formula fallback for the others.
    Rows of unmatched spectra hold the fallback's top-k training ids, NaN scores and, in "fallback_weights", the softmax
    weights of the consensus (the prediction is not a single neighbour's fingerprint)."""
    filt = retrieve_with_filter(sim, data, train_ids, test_ids, params, dataset, split, n_check, out)
    matched = np.array(filt["matched"])
    unmatched_pos = np.where(~matched)[0]
    fb = sf.load_or_run(data, train_ids, val_ids, test_ids, unmatched_pos, train_FP, fp_def, fp_cache, n_workers, out)
    res = {k: (list(v) if isinstance(v, list) else v) for k, v in filt.items()}
    res["topk_ids"] = [list(r) for r in filt["topk_ids"]]
    res["topk_scores"] = np.array(filt["topk_scores"], dtype=np.float32, copy=True)
    res["topk_matches"] = None if filt["topk_matches"] is None else np.array(filt["topk_matches"], copy=True)
    res["params"] = dict(filt["params"], fallback=fb["hparams"])
    res["fallback_weights"] = {}
    for n, p in enumerate(unmatched_pos):
        res["top"][p] = fb["topk"][n][0]
        res["topk_ids"][p] = list(fb["topk"][n])
        res["topk_scores"][p] = np.nan
        if res["topk_matches"] is not None:
            res["topk_matches"][p] = -1
        res["fallback_weights"][test_ids[p]] = np.asarray(fb["weights"][n], dtype=np.float32)
    return res


def retrieve_subformula_only(data, train_ids, val_ids, test_ids, train_FP, fp_def, fp_cache, n_workers, out):
    """The sub-formula rule for every test spectrum (scope "all"), in the layout of retrieve_subformula_match."""
    fb = sf.load_or_run(data, train_ids, val_ids, test_ids, np.arange(len(test_ids)), train_FP, fp_def, fp_cache,
                        n_workers, out, scope="all")
    assert fb["ids"] == list(test_ids)
    k = len(fb["topk"][0])
    return {"params": {"fallback": fb["hparams"]}, "ids": list(test_ids), "train_ids": list(train_ids),
            "top": [row[0] for row in fb["topk"]], "topk_ids": [list(row) for row in fb["topk"]],
            "topk_scores": np.full((len(test_ids), k), np.nan, dtype=np.float32), "topk_matches": None,
            "fallback_weights": {i: np.asarray(w, dtype=np.float32) for i, w in zip(test_ids, fb["weights"])}}


def export_mist_format(mode, dataset, split, sim, test_ids, test_FP, pred_FP, j_by_id, params, c_by_id):
    """Full-test-set predictions in the layout of the trained models. `pred` is the raw
    prediction (continuous for the sub-formula consensus), like MIST's probabilities; `jaccard` is on pred >= 0.5."""
    out = nn.ROOT / "results" / "FP_prediction" / "nearest_neighbour" / dataset / f"{nn.MIST_TAG.get(dataset, dataset)}_NN-{sim}_{mode}_4096_{split}"
    out.mkdir(parents=True, exist_ok=True)
    nn.save_pickle({i: {"pred": pred_FP[k].astype(np.float32), "GT": test_FP[k].astype(np.float32), "jaccard": j_by_id[i],
                        "cosine": c_by_id[i]} for k, i in enumerate(test_ids)}, out / "test_results.pkl")
    dump_json({"jaccard": float(np.mean([j_by_id[i] for i in test_ids])),
               "cosine": float(np.mean([c_by_id[i] for i in test_ids])), "n": len(test_ids),
               "similarity": sim, "mode": mode, **params}, out / "test_performance.json")

def run(sim, mode, dataset, split, fp_def, params, n_check, n_workers=32):
    out = nn.RESULTS_FOLDER / f"{dataset}_{split}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n===== {dataset} / {split} | {sim} ({mode}) | fingerprint {fp_def}"
          + ("" if mode == "subformula_only" else f" | {sim_params(sim, params)}"))

    data = nn.load_dataset(dataset)
    train_ids, val_ids, test_ids = nn.load_split(dataset, split)
    fp_cache = {}
    train_FP = nn.fingerprints([data[i] for i in train_ids], fp_def, fp_cache)
    test_FP = nn.fingerprints([data[i] for i in test_ids], fp_def, fp_cache)
    if mode == "subformula_only":
        res = retrieve_subformula_only(data, train_ids, val_ids, test_ids, train_FP, fp_def, fp_cache, n_workers, out)
    elif mode == "subformula_match":
        res = retrieve_subformula_match(sim, data, train_ids, val_ids, test_ids, params, dataset, split, n_check, out,
                                        train_FP, fp_def, fp_cache, n_workers)
    else:
        retrieve = retrieve_no_filter if mode == "no_formula_filter" else retrieve_with_filter
        res = retrieve(sim, data, train_ids, test_ids, params, dataset, split, n_check, out)

    # raw prediction: the retrieved fingerprint, or for the sub-formula rule the softmax-weighted average of the top-k
    # fingerprints; the binary prediction (used for the Jaccard) thresholds it at 0.5
    train_pos = {i: k for k, i in enumerate(train_ids)}
    pred_raw = train_FP[[train_pos[t] for t in res["top"]]].astype(np.float32)
    for q, i in enumerate(test_ids):
        w = res.get("fallback_weights", {}).get(i)
        if w is not None:
            pred_raw[q] = w @ train_FP[[train_pos[t] for t in res["topk_ids"][q][:len(w)]]].astype(np.float32)
    pred_FP = (pred_raw >= 0.5).astype(np.uint8)
    j_by_id = dict(zip(test_ids, map(float, nn.jaccard(pred_FP, test_FP))))
    c_by_id = dict(zip(test_ids, map(float, nn.cosine(pred_raw, test_FP))))

    train_formula_set = set(data[i]["formula"] for i in train_ids)
    matched_ids = [i for i in test_ids if data[i]["formula"] in train_formula_set]
    unmatched_ids = [i for i in test_ids if data[i]["formula"] not in train_formula_set]
    top_score = res["topk_scores"][:, 0]
    same_formula = np.mean([data[t]["formula"] == data[i]["formula"] for i, t in zip(test_ids, res["top"])])

    m = {"dataset": dataset, "split": split, "n_train": len(train_ids), "n_test": len(test_ids), "params": res["params"],
         "full": mean_on(j_by_id, test_ids)[0],
         "formula_matched": mean_on(j_by_id, matched_ids)[0], "n_formula_matched": len(matched_ids),
         "formula_unmatched": mean_on(j_by_id, unmatched_ids)[0], "n_formula_unmatched": len(unmatched_ids),
         "cosine": {"full": mean_on(c_by_id, test_ids)[0], "formula_matched": mean_on(c_by_id, matched_ids)[0],
                    "formula_unmatched": mean_on(c_by_id, unmatched_ids)[0]},
         "frac_retrieved_same_formula": float(same_formula),
         "n_best_score_zero": int((top_score <= 0).sum()),       # NaN rows (sub-formula fallback) are not counted
         "median_best_score": float(np.nanmedian(top_score))}
    mist_j = nn.load_mist_jaccard(dataset, split) if fp_def == "mist" else None
    if mist_j is not None:
        mist_ids = [i for i in test_ids if i in mist_j]
        m["mist_common"] = {"n": len(mist_ids), "nn": mean_on(j_by_id, mist_ids)[0], "mist": mean_on(mist_j, mist_ids)[0]}
        mist_c = nn.load_mist_cosine(dataset, split)
        m["mist_common"].update({"nn_cosine": mean_on(c_by_id, mist_ids)[0], "mist_cosine": mean_on(mist_c, mist_ids)[0]})
        m["mist"] = {"jaccard": {"full": mean_on(mist_j, test_ids)[0], "formula_matched": mean_on(mist_j, matched_ids)[0],
                                 "formula_unmatched": mean_on(mist_j, unmatched_ids)[0]},
                     "cosine": {"full": mean_on(mist_c, test_ids)[0], "formula_matched": mean_on(mist_c, matched_ids)[0],
                                "formula_unmatched": mean_on(mist_c, unmatched_ids)[0]}}

    metrics = load_json(out / f"{mode}_metrics_{fp_def}.json", {})
    metrics[sim] = m
    dump_json(metrics, out / f"{mode}_metrics_{fp_def}.json")
    jpath = out / f"{mode}_jaccard_{fp_def}.pkl"
    jac = nn.load_pickle(jpath) if jpath.exists() else {}
    jac[sim] = j_by_id
    nn.save_pickle(jac, jpath)
    cpath = out / f"{mode}_cosine_{fp_def}.pkl"
    cos = nn.load_pickle(cpath) if cpath.exists() else {}
    cos[sim] = c_by_id
    nn.save_pickle(cos, cpath)
    if fp_def == "mist":
        export_mist_format(mode, dataset, split, sim, test_ids, test_FP, pred_raw, j_by_id, res["params"], c_by_id)
    print(json.dumps(m, indent=2))

SUMMARY_TEXT = {
    "no_formula_filter": ("Formula-blind nearest neighbour",
                          "Jaccard of the best-scoring training spectrum's fingerprint; the search never uses the formula."),
    "subformula_match": ("Nearest neighbour with formula filter and sub-formula fallback",
                         "Formula-matched test spectra: fingerprint of the best-scoring training spectrum with their "
                         "formula; the others: validation-selected sub-formula consensus (shared by all similarities, "
                         "so the `unmatched` column is identical across rows of a dataset / split)."),
    "subformula_only": ("Sub-formula rule for every test spectrum",
                        "Softmax top-k consensus of the training fingerprints ranked by sub-formula token cosine + "
                        "alpha x heteroatom-weighted formula similarity; (w, alpha, T, k) selected on all validation "
                        "spectra; no hard formula filter."),
    "with_formula_filter": ("Nearest neighbour with formula filter",
                            "Jaccard of the best-scoring training spectrum's fingerprint; formula-matched test spectra "
                            "are searched among the training spectra with their formula, the others over the whole "
                            "training split with the same similarity."),
}

def result_folders(filename):
    """[((dataset, split), path)] of every results/nearest_neighbour/<dataset>_<split>/<filename>, built-in datasets
    and splits first. dataset / split are read from the metrics themselves when present."""
    order = {(d, s): n for n, (d, s) in enumerate((d, s) for d in nn.DATASETS for s in nn.SPLITS)}
    found = []
    for path in nn.RESULTS_FOLDER.glob(f"*/{filename}"):
        m = next(iter(load_json(path, {}).values()), {}) if filename.endswith(".json") else {}
        d, s = (m["dataset"], m["split"]) if isinstance(m, dict) and "dataset" in m else path.parent.name.rsplit("_", 1)
        found.append(((d, s), path))
    return sorted(found, key=lambda x: (order.get(x[0], len(order)), x[0]))


def setting_name(sim, mode):
    return "subformula_only" if mode == "subformula_only" else f"{sim}_{mode}"


def summarize_all(fp_def):
    """One table of every setting: Jaccard / cosine on the full test set per dataset / split, plus MIST
    -> results/nearest_neighbour/summary_nn_<fp>.md"""
    cols, rows, mist = [], {}, {}
    for mode in MODES:
        for (d, s), path in result_folders(f"{mode}_metrics_{fp_def}.json"):
            if (d, s) not in cols:
                cols.append((d, s))
            for sim, m in load_json(path, {}).items():
                rows.setdefault(setting_name(sim, mode), {})[(d, s)] = m
                if "mist" in m and (d, s) not in mist:
                    mist[(d, s)] = m["mist"]
    order = {(d, s): n for n, (d, s) in enumerate((d, s) for d in nn.DATASETS for s in nn.SPLITS)}
    cols.sort(key=lambda c: (order.get(c, len(order)), c))
    names = [setting_name(sim, mode) for mode in MODES for sim in (SIMILARITIES if mode != "subformula_only" else ("subformula",))]
    f = lambda x: "-" if x is None else f"{x:.3f}"
    lines = [f"# Nearest-neighbour baselines (fingerprint: {fp_def})", "",
             "Each cell: Jaccard of the binary prediction (threshold 0.5) / cosine similarity of the raw prediction "
             "(no threshold) vs. the true fingerprint, mean over the full test split. MIST: its own test spectra.", "",
             "| setting | " + " | ".join(f"{d} {s}" for d, s in cols) + " |", "|---|" + "---|" * len(cols)]
    for name in names:
        if name in rows:
            lines.append(f"| {name} | " + " | ".join(
                f"{f(rows[name][c]['full'])} / {f(rows[name][c].get('cosine', {}).get('full'))}" if c in rows[name] else "-"
                for c in cols) + " |")
    if mist:
        lines.append("| MIST | " + " | ".join(
            f"{f(mist[c]['jaccard']['full'])} / {f(mist[c]['cosine']['full'])}" if c in mist else "-" for c in cols) + " |")
    path = nn.RESULTS_FOLDER / f"summary_nn_{fp_def}.md"
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines)); print(f"\n-> {path}")


def summarize(mode, fp_def):
    """Markdown table of every <mode>_metrics_<fp>.json -> results/nearest_neighbour/summary_<mode>_<fp>.md"""
    title, text = SUMMARY_TEXT[mode]
    lines = [f"# {title} (fingerprint: {fp_def})", "",
             text + " `matched` / `unmatched` split the test set by whether a training spectrum has the same formula.", "",
             "Jaccard of the binary prediction (threshold 0.5) and cosine similarity of the raw prediction (retrieved "
             "fingerprint, sub-formula consensus average, MIST probabilities; no threshold) vs. the true fingerprint; "
             "MIST is compared on the spectra it has predictions for.", "",
             "| dataset | split | similarity | Jaccard full | matched | unmatched | cosine full | matched | unmatched | "
             "retrieved same formula | best score = 0 | MIST Jaccard (common) | NN Jaccard (common) | MIST cosine (common) | NN cosine (common) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x: "-" if x is None else f"{x:.3f}"
    for (d, s), path in result_folders(f"{mode}_metrics_{fp_def}.json"):
        metrics = load_json(path, {})
        if True:
            for sim in SIMILARITIES + ("subformula",):
                if sim not in metrics:
                    continue
                m = metrics[sim]; mc = m.get("mist_common", {}); c = m.get("cosine", {})
                lines.append(f"| {d} | {s} | {sim} | {f(m['full'])} | {f(m['formula_matched'])} | {f(m['formula_unmatched'])} | "
                             f"{f(c.get('full'))} | {f(c.get('formula_matched'))} | {f(c.get('formula_unmatched'))} | "
                             f"{m['frac_retrieved_same_formula']:.3f} | {m['n_best_score_zero']} | {f(mc.get('mist'))} | {f(mc.get('nn'))} | "
                             f"{f(mc.get('mist_cosine'))} | {f(mc.get('nn_cosine'))} |")
    path = nn.RESULTS_FOLDER / f"summary_{mode}_{fp_def}.md"
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines)); print(f"\n-> {path}")
