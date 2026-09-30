"""
Peak-matching spectral similarities for exhaustive nearest-neighbour search (similarities cosine, modified_cosine, neutral_loss of run_nn.py).

The three similarities are the matchms ones, re-implemented as one numba kernel so that every test spectrum can be scored against every
training spectrum (~1e9 pairs on MassSpecGym; matchms' per-pair Python calls would take days). The kernel follows the
matchms 0.28 code path pair by pair -- same peak matching, same pair ordering, same greedy assignment, same
normalisation -- and `check_against_matchms` asserts that the scores agree on random pairs before every run.

    cosine            matchms CosineGreedy: peaks match when |mz_ref - mz_query| <= tolerance
    modified_cosine   matchms ModifiedCosine: peaks also match after shifting the query by the precursor m/z difference
    neutral_loss      matchms NeutralLossesCosine: peaks at or above the precursor m/z are dropped and only the shifted
                      match is used, i.e. peaks match when their neutral losses (precursor - mz) agree

In every case the matched pairs are sorted by intensity product and assigned greedily (each peak used once); the score
is the sum of the assigned products divided by the norms of the two spectra (after the neutral-loss filter).
Reference = training spectrum, query = test spectrum, as in matchms.calculate_scores(references=train, queries=test).
"""

import numpy as np
import numba
from numba import prange

SIMILARITIES = ("cosine", "modified_cosine", "neutral_loss")
_MODE = {"cosine": 0, "modified_cosine": 1, "neutral_loss": 2}

# --------------------------------------------------------------------------------------------------
# Packing
# --------------------------------------------------------------------------------------------------

def pack_spectra(records, similarity, max_peaks=None):
    """Concatenated peak arrays for a list of records: (offsets, mz, intensity, precursor_mz).
    Peaks are the stored (`mz`, `intensity_norm`) pairs. When `max_peaks` is set, only the `max_peaks` most intense
    peaks are kept (matchms `reduce_to_number_of_peaks`); peaks are then sorted by m/z, which matchms' matching assumes.
    For `neutral_loss` the peaks with mz >= precursor m/z are dropped afterwards, as NeutralLossesCosine does."""
    offsets = np.zeros(len(records) + 1, dtype=np.int64)
    mz_parts, int_parts = [], []
    prec = np.empty(len(records), dtype=np.float64)
    for k, r in enumerate(records):
        mz = np.array([p["mz"] for p in r["peaks"]], dtype=np.float64)
        it = np.array([p["intensity_norm"] for p in r["peaks"]], dtype=np.float64)
        if max_peaks is not None and len(mz) > max_peaks:
            keep = np.argsort(-it, kind="stable")[:max_peaks]
            mz, it = mz[keep], it[keep]
        order = np.argsort(mz, kind="stable")
        mz, it = mz[order], it[order]
        prec[k] = float(r["precursor_MZ_final"])
        if similarity == "neutral_loss":
            keep = mz < prec[k]
            mz, it = mz[keep], it[keep]
        mz_parts.append(mz); int_parts.append(it)
        offsets[k + 1] = offsets[k] + len(mz)
    mz = np.concatenate(mz_parts) if mz_parts else np.zeros(0)
    it = np.concatenate(int_parts) if int_parts else np.zeros(0)
    return offsets, mz, it, prec

def spectrum_norms(offsets, mz, intensity, mz_power, intensity_power):
    """sqrt(sum_peaks (mz^mz_power * intensity^intensity_power)^2) per spectrum, the matchms normalisation."""
    w = (mz ** mz_power) * (intensity ** intensity_power)
    sq = np.concatenate([[0.0], np.cumsum(w * w)])
    return np.sqrt(sq[offsets[1:]] - sq[offsets[:-1]])

# --------------------------------------------------------------------------------------------------
# Kernel
# --------------------------------------------------------------------------------------------------

@numba.njit(cache=True)
def _collect(mz1, w1, mz2, w2, tol, shift, n, i1, i2, prod):
    """Append the matches of matchms.find_matches (spec2 shifted by `shift`) to the buffers from position n, in the
    same order; returns the new length."""
    lowest = 0
    for a in range(mz1.shape[0]):
        lo = mz1[a] - tol
        hi = mz1[a] + tol
        for b in range(lowest, mz2.shape[0]):
            m = mz2[b] + shift
            if m > hi:
                break
            if m < lo:
                lowest = b + 1
            else:
                i1[n] = a; i2[n] = b; prod[n] = w1[a] * w2[b]
                n += 1
    return n

@numba.njit(cache=True)
def _pair_score(mz1, w1, norm1, prec1, mz2, w2, norm2, prec2, mode, tol, i1, i2, prod, used1, used2):
    """(score, n_matches) of reference 1 vs query 2; w = mz^mz_power * intensity^intensity_power per peak."""
    if mz1.shape[0] == 0 or mz2.shape[0] == 0:
        return 0.0, 0
    shift = prec1 - prec2
    n = 0
    if mode == 0 or mode == 1:
        n = _collect(mz1, w1, mz2, w2, tol, 0.0, n, i1, i2, prod)
    if mode == 1 or mode == 2:
        n = _collect(mz1, w1, mz2, w2, tol, shift, n, i1, i2, prod)
    if n == 0:
        return 0.0, 0
    # matchms: np.argsort(prod, kind="mergesort")[::-1], then greedy assignment
    order = np.argsort(prod[:n], kind="mergesort")
    for a in range(mz1.shape[0]):
        used1[a] = False
    for b in range(mz2.shape[0]):
        used2[b] = False
    score = 0.0
    used = 0
    for t in range(n - 1, -1, -1):
        k = order[t]
        if not used1[i1[k]] and not used2[i2[k]]:
            score += prod[k]
            used1[i1[k]] = True
            used2[i2[k]] = True
            used += 1
    return score / (norm1 * norm2), used

@numba.njit(parallel=True, cache=True)
def _search(r_off, r_mz, r_w, r_norm, r_prec, q_off, q_mz, q_w, q_norm, q_prec, c_ptr, c_idx, mode, tol, topk):
    """Top-k references per query. With an empty c_ptr every reference is scored; otherwise query q scores only the
    references c_idx[c_ptr[q]:c_ptr[q + 1]] (in that order, so ties go to the first of them)."""
    full = c_ptr.shape[0] == 0
    n_q = q_off.shape[0] - 1
    n_r = r_off.shape[0] - 1
    max_r = 0
    for j in range(n_r):
        max_r = max(max_r, r_off[j + 1] - r_off[j])
    top_idx = np.full((n_q, topk), -1, dtype=np.int64)
    top_score = np.full((n_q, topk), -1.0, dtype=np.float64)
    top_match = np.zeros((n_q, topk), dtype=np.int64)
    for q in prange(n_q):
        qs, qe = q_off[q], q_off[q + 1]
        nq = qe - qs
        cap = max(1, 2 * nq * max_r)            # every (a, b) pair at most once per shift
        i1 = np.empty(cap, dtype=np.int64)
        i2 = np.empty(cap, dtype=np.int64)
        prod = np.empty(cap, dtype=np.float64)
        used1 = np.empty(max(1, max_r), dtype=np.bool_)
        used2 = np.empty(max(1, nq), dtype=np.bool_)
        n_c = n_r if full else c_ptr[q + 1] - c_ptr[q]
        for t in range(n_c):
            j = t if full else c_idx[c_ptr[q] + t]
            rs, re = r_off[j], r_off[j + 1]
            s, m = _pair_score(r_mz[rs:re], r_w[rs:re], r_norm[j], r_prec[j],
                               q_mz[qs:qe], q_w[qs:qe], q_norm[q], q_prec[q], mode, tol, i1, i2, prod, used1, used2)
            # keep the topk best, best first; strict '>' so that among equal scores the earliest training spectrum wins
            if s > top_score[q, topk - 1]:
                p = topk - 1
                while p > 0 and s > top_score[q, p - 1]:
                    top_score[q, p] = top_score[q, p - 1]
                    top_idx[q, p] = top_idx[q, p - 1]
                    top_match[q, p] = top_match[q, p - 1]
                    p -= 1
                top_score[q, p] = s
                top_idx[q, p] = j
                top_match[q, p] = m
    return top_idx, top_score, top_match

def _weights(pack, mz_power, intensity_power):
    offsets, mz, it, prec = pack
    return (mz ** mz_power) * (it ** intensity_power), spectrum_norms(offsets, mz, it, mz_power, intensity_power)

def search(train_pack, test_pack, similarity, tolerance=0.1, mz_power=0.0, intensity_power=1.0, topk=10, candidates=None):
    """For every test spectrum the `topk` training spectra with the highest similarity, over the whole training set or,
    when `candidates` is given, over candidates[q] (sorted training indices) only.
    Returns (idx, score, n_matches), each (n_test x topk), best first; idx indexes the training list, and slots beyond
    the number of candidates hold idx -1 / score -1."""
    r_w, r_norm = _weights(train_pack, mz_power, intensity_power)
    q_w, q_norm = _weights(test_pack, mz_power, intensity_power)
    if candidates is None:
        c_ptr, c_idx = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    else:
        assert len(candidates) == len(test_pack[0]) - 1
        c_ptr = np.concatenate([[0], np.cumsum([len(c) for c in candidates])]).astype(np.int64)
        c_idx = np.concatenate([np.asarray(c, dtype=np.int64) for c in candidates]) if candidates else np.zeros(0, np.int64)
    return _search(train_pack[0], train_pack[1], r_w, r_norm, train_pack[3],
                   test_pack[0], test_pack[1], q_w, q_norm, test_pack[3], c_ptr, c_idx,
                   _MODE[similarity], float(tolerance), int(topk))

def pair_scores(train_pack, test_pack, pairs, similarity, tolerance=0.1, mz_power=0.0, intensity_power=1.0):
    """Kernel scores for explicit (train_index, test_index) pairs (used by the matchms check)."""
    r_w, r_norm = _weights(train_pack, mz_power, intensity_power)
    q_w, q_norm = _weights(test_pack, mz_power, intensity_power)
    ro, rm, _, rp = train_pack
    qo, qm, _, qp = test_pack
    out = []
    for j, q in pairs:
        nr, nq = ro[j + 1] - ro[j], qo[q + 1] - qo[q]
        cap = max(1, 2 * nr * nq)
        s, m = _pair_score(rm[ro[j]:ro[j + 1]], r_w[ro[j]:ro[j + 1]], r_norm[j], rp[j],
                           qm[qo[q]:qo[q + 1]], q_w[qo[q]:qo[q + 1]], q_norm[q], qp[q], _MODE[similarity], float(tolerance),
                           np.empty(cap, np.int64), np.empty(cap, np.int64), np.empty(cap), np.empty(max(1, nr), np.bool_),
                           np.empty(max(1, nq), np.bool_))
        out.append((s, m))
    return out

# --------------------------------------------------------------------------------------------------
# Check against matchms
# --------------------------------------------------------------------------------------------------

def _matchms_spectrum(records_pack_entry):
    from matchms import Spectrum
    mz, it, prec = records_pack_entry
    return Spectrum(mz=mz, intensities=it, metadata={"precursor_mz": float(prec)})

def check_against_matchms(train_records, test_records, similarity, max_peaks, tolerance, mz_power, intensity_power,
                          n_pairs=300, seed=0, atol=1e-6):
    """Score `n_pairs` random (train, test) pairs -- plus, for every sampled test spectrum, the training spectrum with
    the closest precursor m/z, so that pairs with many (and shifted) matches are covered -- with the kernel and with
    matchms itself on the same preprocessed peaks, and raise if any score or match count differs.
    matchms receives the peaks before the neutral-loss filter, since NeutralLossesCosine applies that filter itself."""
    from matchms.similarity import CosineGreedy, ModifiedCosine, NeutralLossesCosine
    cls = {"cosine": CosineGreedy, "modified_cosine": ModifiedCosine, "neutral_loss": NeutralLossesCosine}[similarity]
    fn = cls(tolerance=tolerance, mz_power=mz_power, intensity_power=intensity_power)
    rng = np.random.default_rng(seed)
    tr = rng.choice(len(train_records), size=min(n_pairs, len(train_records)), replace=False)
    te = rng.choice(len(test_records), size=len(tr), replace=True)
    train_prec = np.array([float(r["precursor_MZ_final"]) for r in train_records])
    near = [int(np.argmin(np.abs(train_prec - float(test_records[q]["precursor_MZ_final"])))) for q in te]
    pairs = list(zip(tr.tolist(), te.tolist())) + list(zip(near, te.tolist()))
    tr_ids = sorted(set(j for j, _ in pairs)); te_ids = sorted(set(q for _, q in pairs))
    tr_pos = {j: k for k, j in enumerate(tr_ids)}; te_pos = {q: k for k, q in enumerate(te_ids)}
    sub_tr = [train_records[j] for j in tr_ids]; sub_te = [test_records[q] for q in te_ids]
    kernel = pair_scores(pack_spectra(sub_tr, similarity, max_peaks), pack_spectra(sub_te, similarity, max_peaks),
                         [(tr_pos[j], te_pos[q]) for j, q in pairs], similarity, tolerance, mz_power, intensity_power)
    # matchms reference on the unfiltered (capped, m/z-sorted) peaks
    raw_tr = pack_spectra(sub_tr, "cosine", max_peaks); raw_te = pack_spectra(sub_te, "cosine", max_peaks)
    def spec(pack, k):
        o, mz, it, prec = pack
        return _matchms_spectrum((mz[o[k]:o[k + 1]], it[o[k]:o[k + 1]], prec[k]))
    n_bad, worst, n_nonzero = 0, 0.0, 0
    for (j, q), (s, m) in zip(pairs, kernel):
        ref = fn.pair(spec(raw_tr, tr_pos[j]), spec(raw_te, te_pos[q]))
        ref_s, ref_m = float(ref["score"]), int(ref["matches"])
        n_nonzero += ref_s > 0
        diff = abs(ref_s - s)
        worst = max(worst, diff)
        if diff > atol or ref_m != m:
            n_bad += 1
    if n_bad:
        raise AssertionError(f"{similarity}: kernel disagrees with matchms on {n_bad} of {len(pairs)} pairs "
                             f"(max |score diff| {worst:.2e})")
    return {"n_pairs": len(pairs), "n_nonzero": int(n_nonzero), "max_abs_diff": worst}
