# Detailed Benchmark Results

This file contains supplementary result tables for the reproducibility report in
[README.md](README.md). The main README keeps the headline result tables and the
MassSpecGym official retrieval split; this file keeps the additional
full-test retrieval, fingerprint-cosine, k-nearest-neighbour sensitivity, and
formula-filtered diagnostic tables.

## MIST Configuration Notes

The main README reports three MIST rows: `MIST in Comment` at the original
hard-coded `0.5` threshold, the same model at a validation-fitted threshold, and
`Corrected MIST`. The configuration differences are documented in
[benchmarked_models/mist/all_configs/README.md](benchmarked_models/mist/all_configs/README.md).

`Corrected MIST` is deliberately a **single-variable** change from the Comment's own
configuration: same architecture, same batch size, same epoch budget, same seed,
same MAGMa auxiliary supervision, trained with MIST's default cosine objective
instead of BCE (with `val_monitor` moved to match, without which the cosine run
silently keeps its epoch-1 checkpoint). It is not a separately tuned model
family, and no fork of MIST is involved: both rows run against unmodified
upstream MIST from the [`main_v2` branch](https://github.com/samgoldman97/mist/tree/main_v2).

Both rows also fit the fingerprint binarization threshold on the **validation**
split rather than assuming `0.5`. This matters more than it might appear: the
optimal cut is objective-dependent (0.78-0.84 under BCE, 0.12-0.24 under cosine),
so a fixed `0.5` penalizes the cosine models specifically. Evaluated at `0.5`,
`Corrected MIST` scores 0.018 on NPLIB1 scaffold; at its validation-selected cut it scores
0.317. Any comparison across objectives at a shared fixed threshold is measuring
calibration, not fingerprint quality.

## Fingerprint Cosine and Candidate Retrieval

In addition to the Jaccard score used in the Comment, we report two
application-adjacent metrics: a continuous metric for fingerprint accuracy and
database retrieval.

The same full-test prediction artifacts were evaluated by cosine similarity
between predicted and ground-truth Morgan4096 fingerprints. This is still an
intermediate fingerprint metric, but unlike binary Jaccard it does not discard
confidence information before scoring and is more in line with the original
model's training objectives.

| Dataset | Model | Scaffold FP cosine (↑) | Random FP cosine (↑) |
| ----------- | ------------------ | ---------------------: | -------------------: |
| NPLIB1 | MIST in Comment | 0.429 (n=2,689) | 0.718 (n=2,744) |
| NPLIB1 | Corrected MIST | **0.529** (n=2,689) | **0.819** (n=2,744) |
| NPLIB1 | Nearest neighbour | 0.321 (n=2,689) | 0.770 (n=2,744) |
| NPLIB1 | DreaMS NN | 0.376 (n=2,689) | 0.796 (n=2,744) |
| MassSpecGym | MIST in Comment | 0.457 (n=16,042) | 0.858 (n=16,250) |
| MassSpecGym | Corrected MIST | **0.571** (n=16,042) | 0.889 (n=16,250) |
| MassSpecGym | Nearest neighbour | 0.344 (n=16,042) | 0.940 (n=16,250) |
| MassSpecGym | DreaMS NN | 0.405 (n=16,042) | **0.955** (n=16,250) |

Fingerprint cosine changes the picture on one of the two random splits. Corrected MIST
leads on both scaffold splits, as it does on Jaccard, and it also leads on NPLIB1
random (0.819 versus 0.796 for DreaMS). Only MassSpecGym random, the split with 
95.3% structural leakage, keeps nearest neighbour ahead on this metric. Binarizing 
at a single cut discards confidence information that the cosine metric retains,
which is part of why the two metrics disagree.

Retrieval is evaluated by 2D InChIKey hit rate after ranking candidates with the
predicted fingerprint. For NPLIB1, candidates are PubChem structures with the
same chemical formula. The PubChem formula-to-structure map used for this
evaluation is available as `pubchem_formulae_inchikey.hdf5` from [Zenodo](https://zenodo.org/records/15529765/files/pubchem_formulae_inchikey.hdf5);
formulas absent from that file were completed with the PubChem API. For
MassSpecGym, candidates are the official formula and mass candidate sets. The
nearest-neighbour rows rank candidates with the same formula-first predictions
used in the headline table.

**Corrected MIST leads every retrieval setting, including the two random splits where it
trails on binary Jaccard.** This is the metric closest to the task the Comment
motivates, which is ranking candidate structures.

| Dataset | Candidate set | Model | Scaffold top-1 / top-5 / top-10 (↑) | Random top-1 / top-5 / top-10 (↑) |
| ----------- | ------------------ | ------------------ | ----------------------------------: | --------------------------------: |
| NPLIB1 | PubChem same formula | Corrected MIST | **0.107 / 0.229 / 0.321** (n=2,689) | **0.612 / 0.736 / 0.786** (n=2,744) |
| NPLIB1 | PubChem same formula | Nearest neighbour | 0.042 / 0.115 / 0.188 (n=2,689) | 0.348 / 0.506 / 0.556 (n=2,744) |
| NPLIB1 | PubChem same formula | DreaMS NN | 0.103 / 0.232 / 0.296 (n=2,689) | 0.419 / 0.610 / 0.665 (n=2,744) |
| MassSpecGym | Official formula | Corrected MIST | **0.219 / 0.395 / 0.490** (n=16,042) | **0.811 / 0.903 / 0.928** (n=16,250) |
| MassSpecGym | Official formula | Nearest neighbour | 0.093 / 0.181 / 0.248 (n=16,042) | 0.600 / 0.745 / 0.787 (n=16,250) |
| MassSpecGym | Official formula | DreaMS NN | 0.144 / 0.267 / 0.342 (n=16,042) | 0.633 / 0.787 / 0.832 (n=16,250) |
| MassSpecGym | Official mass | Corrected MIST | **0.349 / 0.573 / 0.661** (n=16,042) | **0.873 / 0.944 / 0.960** (n=16,250) |
| MassSpecGym | Official mass | Nearest neighbour | 0.122 / 0.181 / 0.219 (n=16,042) | 0.696 / 0.794 / 0.824 (n=16,250) |
| MassSpecGym | Official mass | DreaMS NN | 0.198 / 0.300 / 0.347 (n=16,042) | 0.737 / 0.844 / 0.877 (n=16,250) |

On MassSpecGym random, nearest neighbour leads Corrected MIST by 0.132 Jaccard but trails
it by 0.211 top-1 hit rate under the official formula candidate set. A retrieved 
training fingerprint scores well against its own molecule's fingerprint, but that 
does not translate into ranking the correct structure highly within a large candidate 
set.

## Candidate Policy and the Denominator

The nearest-neighbour implementation in the Comment filters candidates by exact
molecular formula and **skips** test entries when no formula match exists in the
training set. That is the technical issue behind the first README table: the
skipped entries never reach the mean, so the reported score is not a full-test
benchmark. The table below is the denominator check.

| Dataset | Split | Test entries covered | NN on covered entries | NN with skipped entries as 0 |
|---|---:|---:|---:|---:|
| NPLIB1 | random | 80.6% | 0.822 | 0.663 |
| NPLIB1 | scaffold | 38.2% | 0.293 | 0.112 |
| MassSpecGym | random | 97.1% | 0.953 | 0.925 |
| MassSpecGym | scaffold | 41.0% | 0.455 | 0.186 |

On both scaffold splits roughly 60% of the test set is discarded. Our default,
`same_formula_candidates_fallback`, is a best-effort implementation: keep the 
formula restriction where it can be satisfied, and fall back to
the whole training set where it cannot. Every test spectrum is answered, so the
denominator matches MIST's, and the formula prior is retained wherever it exists.
The table below compares all three policies against Corrected MIST on the full test set.

| Dataset | Split | Corrected MIST (↑) | NN, all train (↑) | NN, formula-first (default) (↑) | _NN, formula-only subset_ |
|---|---|---:|---:|---:|---:|
| NPLIB1 | scaffold | **0.317** (n=2,689) | 0.195 (n=2,689) | 0.212 (n=2,689; fallback=1,661) | _0.293 (n=1,028)_ |
| NPLIB1 | random | 0.712 (n=2,744) | 0.611 (n=2,744) | **0.720** (n=2,744; fallback=532) | _0.822 (n=2,212)_ |
| MassSpecGym | scaffold | **0.368** (n=16,042) | 0.230 (n=16,042) | 0.256 (n=16,042; fallback=9,471) | _0.455 (n=6,571)_ |
| MassSpecGym | random | 0.813 (n=16,250) | 0.850 (n=16,250) | **0.929** (n=16,250; fallback=473) | _0.953 (n=15,777)_ |

The formula prior is worth 0.017-0.109 Jaccard over searching the whole training
set, and it is what makes nearest neighbour competitive on the random splits. The
italicised column is not comparable to the others: it is scored on a different,
smaller, easier set of spectra, and it is reproduced here only to show how much of
the published nearest-neighbour advantage came from the denominator rather than
from the method.

## Leakage and the Nearest-Neighbour Ceiling

The nearest-neighbour upper bound is the best score achievable by any method that
can only ever return a training-set fingerprint: for each test spectrum, the
single training fingerprint closest to the true answer. It measures how much of
each split is solvable by retrieval alone, independently of any model.

On the full test sets:

| Dataset | Split | Leakage (test structures already in training) | FP oracle (NN ceiling) | Best NN | Corrected MIST |
|---|---|---:|---:|---:|---:|
| NPLIB1 | scaffold | **1.6%** | 0.435 | 0.258 | **0.317** |
| NPLIB1 | random | **62.9%** | 0.822 | **0.744** | 0.712 |
| MassSpecGym | scaffold | **14.3%** | 0.489 | 0.306 | **0.368** |
| MassSpecGym | random | **95.3%** | 0.972 | **0.944** | 0.813 |

The two random splits are close to lookup problems. On MassSpecGym random, 95.3%
of test structures are already present in training and the retrieval ceiling is
0.972; DreaMS nearest neighbour reaches 0.944, within 0.028 of a ceiling that
exists only because the answers are in the training set. On the scaffold splits
that ceiling falls to 0.435 and 0.489, and Corrected MIST exceeds every nearest-neighbour
variant -- on MassSpecGym scaffold it also exceeds 75% of the retrieval ceiling
while both NN baselines fall well short of it.

The same picture holds on the favourable subset the formula-filtered code
actually scores, where leakage is even more extreme on the random splits:

| Dataset | Model | Scaffold Jaccard (↑) | Random Jaccard (↑) |
| ----------- | -------------------------- | ----------------------------------: | -----------------------------------: |
| NPLIB1 | Corrected MIST | **0.340** (n=1,028; **leak=4.2%**) | 0.783 (n=2,212; leak=78.0%) |
| NPLIB1 | Formula NN | 0.293 (n=1,028; leak=4.2%) | 0.822 (n=2,212; leak=78.0%) |
| NPLIB1 | Formula DreaMS NN | 0.293 (n=1,028; leak=4.2%) | **0.830** (n=2,212; leak=78.0%) |
| NPLIB1 | FP oracle (NN upper bound) | _0.326_ (n=1,028; leak=4.2%) | _0.869_ (n=2,212; leak=78.0%) |
| MassSpecGym | Corrected MIST | 0.464 (n=6,571; leak=34.8%) | 0.827 (n=15,777; leak=98.1%) |
| MassSpecGym | Formula NN | 0.455 (n=6,571; leak=34.8%) | 0.953 (n=15,777; leak=98.1%) |
| MassSpecGym | Formula DreaMS NN | **0.470** (n=6,571; leak=34.8%) | **0.966** (n=15,777; leak=98.1%) |
| MassSpecGym | FP oracle (NN upper bound) | _0.516_ (n=6,571; leak=34.8%) | _0.987_ (n=15,777; leak=98.1%) |

On NPLIB1 scaffold, Corrected MIST **exceeds the nearest-neighbour ceiling** (0.340 versus
0.326), which a retrieval method cannot do by construction. On the random subsets
every method including the ceiling is compressed into a narrow high band, because
78% and 98% of those spectra have their own structure in the training set.

The Comment interprets the random-to-scaffold drop for DreaMS as evidence of poor
generalization:

> DreaMS, the top model under the random split, drops sharply under the scaffold split, showing poor generalization.

The ceiling calculation points to a more specific reading. Every method drops
between the random and scaffold splits, including the oracle, which contains no
model at all. The drop is therefore primarily a property of the split and its
leakage profile rather than evidence that DreaMS specifically fails to
generalize, and it is also why nearest neighbour retains an advantage over MIST
on the random splits while losing it on the scaffold splits.

## kNN Sensitivity Check

The nearest-neighbour baselines above use the single most similar training
spectrum (`k=1`). To check that the result is not an artifact of that choice, we
sweep `k` over the whole training split with
`benchmarked_models/nearest_neighbour/04_compute_knn_sensitivity.py`. Note that
this sweep uses the **all-train** candidate pool, not the formula-first default,
so its `k=1` column matches the `NN, all train` column. Two fingerprint aggregations 
are tested: majority vote on each bit, and averaging each bit before scoring the continuous fingerprint. For the binary Jaccard score the averaged fingerprint is thresholded 
at 0.5, so vote and average give identical Jaccard for odd `k`.

These sweeps do not improve the nearest-neighbour baseline. In all four
dataset/split settings `k=1` is the strongest kNN Jaccard variant, and adding
neighbours degrades it sharply on the random splits, where the single nearest
training spectrum is often the same molecule and averaging in further neighbours
only dilutes a correct answer. Averaging does help fingerprint cosine on the
scaffold splits, but not enough to reach Corrected MIST.

| Dataset     | Split    | Corrected MIST Jaccard (↑) | NN k=1 Jaccard (↑) | kNN k=3 Jaccard (↑) | kNN k=5 Jaccard (↑) |
|-------------|----------|-------------------:|-------------------:|--------------------:|--------------------:|
| NPLIB1      | scaffold | **0.317** (n=2,689) |              0.195 |               0.195 |               0.189 |
| NPLIB1      | random   | **0.712** (n=2,744) |              0.611 |               0.512 |               0.457 |
| MassSpecGym | scaffold | **0.368** (n=16,042) |             0.230 |               0.227 |               0.217 |
| MassSpecGym | random   |    0.813 (n=16,250) |          **0.850** |               0.735 |               0.668 |

| Dataset     | Split    | Corrected MIST FP Cosine (↑) | NN k=1 FP Cosine (↑) | k=3 vote (↑) | k=3 average (↑) | k=5 vote (↑) | k=5 average (↑) |
|-------------|----------|---------------------:|---------------------:|-------------:|----------------:|-------------:|----------------:|
| NPLIB1      | scaffold |            **0.529** |                0.297 |        0.315 |           0.344 |        0.323 |           0.361 |
| NPLIB1      | random   |            **0.819** |                0.672 |        0.602 |           0.654 |        0.563 |           0.633 |
| MassSpecGym | scaffold |            **0.571** |                0.314 |        0.325 |           0.349 |        0.330 |           0.363 |
| MassSpecGym | random   |            **0.889** |                0.869 |        0.780 |           0.830 |        0.733 |           0.800 |

Because this sweep uses the all-train candidate pool, Corrected MIST leads every cell here
except MassSpecGym random Jaccard. The formula-first policy used in the headline
table is stronger than any of these kNN variants on the random splits, which is
why nearest neighbour leads there in the main tables but not in this one. The
conclusion the sweep supports is narrow and is the one it was run to test: the
choice of `k=1` is not what makes the nearest-neighbour baseline competitive.


## Notes on Interpretation

- The Jaccard tables evaluate fingerprint prediction, not full candidate
  retrieval. The two metrics disagree on the random splits, and retrieval is the
  endpoint structure annotation actually uses.
- The MassSpecGym official table in the main README evaluates fixed-training set and 
  fixed-candidate retrieval and should be read as a separate benchmark setting.
- All nearest-neighbour rows in the headline tables use the
  `same_formula_candidates_fallback` policy at coverage 1.000, against the full
  MGF-derived training pool. The formula-only subset scores are reported only as a
  diagnostic and are not comparable to full-test scores.
- The FP-oracle upper bound shows the ceiling of any nearest-neighbour method that
  can only return training-set fingerprints. It is a property of the split, not of
  a model, and it is the cleanest way to see how much of each split is solvable by
  retrieval alone.
- MIST rows use a decision threshold fitted on the validation split. Comparing
  models trained under different objectives at a shared fixed threshold measures
  calibration rather than fingerprint quality.
