# Detailed Benchmark Results

This file contains supplementary result tables for the reproducibility report in
[README.md](README.md). The main README keeps the headline result tables and the
MassSpecGym official retrieval split; this file keeps the additional
full-test retrieval, fingerprint-cosine, k-nearest-neighbour sensitivity, and
formula-filtered diagnostic tables.

## MIST Configuration Notes

The main README reports both the `MIST in Comment` numbers and `Tuned MIST`
numbers. The configuration differences are documented in
[benchmarked_models/mist/all_configs/README.md](benchmarked_models/mist/all_configs/README.md).
In brief, the Comment used MIST settings that differ substantially from stronger
MIST configurations, and its supplementary information states that the evaluated
model omitted enhancements such as contrastive fine-tuning and data
augmentation. Because deep learning models are sensitive to training objectives,
optimization schedules, and data handling, substantial changes to the training
pipeline should be accompanied by hyperparameter tuning. The tuned rows in this
repository keep the architecture fixed but use stronger training hyperparameters
selected from recent MIST-style runs.

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
| NPLIB1 | Tuned MIST | **0.505** (n=2,689) | **0.838** (n=2,744) |
| NPLIB1 | Nearest neighbour | 0.296 (n=2,689) | 0.671 (n=2,744) |
| NPLIB1 | Formula-first NN | 0.321 (n=2,689) | 0.770 (n=2,744) |
| NPLIB1 | DreaMS | 0.379 (n=2,689) | 0.753 (n=2,744) |
| MassSpecGym | Tuned MIST | **0.514** (n=16,042) | **0.958** (n=16,250) |
| MassSpecGym | Nearest neighbour | 0.314 (n=16,042) | 0.869 (n=16,250) |
| MassSpecGym | Formula-first NN | 0.344 (n=16,042) | 0.940 (n=16,250) |
| MassSpecGym | DreaMS | 0.398 (n=16,042) | 0.912 (n=16,250) |

Retrieval is evaluated by 2D InChIKey hit rate after ranking candidates with the
predicted fingerprint. For NPLIB1, candidates are PubChem structures with the
same chemical formula. The PubChem formula-to-structure map used for this
evaluation is available as `pubchem_formulae_inchikey.hdf5` from [Zenodo](https://zenodo.org/records/15529765/files/pubchem_formulae_inchikey.hdf5);
formulas absent from that file were completed with the PubChem API. For
MassSpecGym, candidates are the official formula and mass candidate sets. Tuned
MIST outperforms the nearest-neighbour baseline in these candidate-ranking
evaluations.

| Dataset | Candidate set | Model | Scaffold top-1 / top-5 / top-10 (↑) | Random top-1 / top-5 / top-10 (↑) |
| ----------- | ------------------ | ------------------ | ----------------------------------: | --------------------------------: |
| NPLIB1 | PubChem same formula | Tuned MIST | **0.126 / 0.255 / 0.336** (n=2,689) | **0.636 / 0.755 / 0.798** (n=2,744) |
| NPLIB1 | PubChem same formula | Nearest neighbour | 0.055 / 0.142 / 0.195 (n=2,689) | 0.380 / 0.533 / 0.585 (n=2,744) |
| NPLIB1 | PubChem same formula | DreaMS | 0.100 / 0.230 / 0.296 (n=2,689) | 0.428 / 0.611 / 0.667 (n=2,744) |
| MassSpecGym | Official formula | Tuned MIST | **0.225 / 0.359 / 0.429** (n=16,042) | **0.891 / 0.948 / 0.960** (n=16,250) |
| MassSpecGym | Official formula | Nearest neighbour | 0.154 / 0.227 / 0.290 (n=16,042) | 0.808 / 0.861 / 0.876 (n=16,250) |
| MassSpecGym | Official formula | DreaMS | 0.158 / 0.286 / 0.357 (n=16,042) | 0.700 / 0.836 / 0.867 (n=16,250) |
| MassSpecGym | Official mass | Tuned MIST | **0.321 / 0.486 / 0.562** (n=16,042) | **0.927 / 0.968 / 0.977** (n=16,250) |
| MassSpecGym | Official mass | Nearest neighbour | 0.175 / 0.226 / 0.269 (n=16,042) | 0.838 / 0.875 / 0.885 (n=16,250) |
| MassSpecGym | Official mass | DreaMS | 0.209 / 0.313 / 0.366 (n=16,042) | 0.795 / 0.876 / 0.899 (n=16,250) |

## Formula-Filtered NN Diagnostic

The original nearest-neighbour logic filters candidates by exact molecular
formula and skips test entries when no formula match exists in the training set.
That score should not be used as a full-test benchmark unless skipped entries
are counted. The table below is the denominator check: it shows how many queries
remain after the skip and what the score becomes if skipped entries are counted
as failures.

| Dataset | Split | Test entries covered | NN on covered entries | NN with skipped entries as 0 |
|---|---:|---:|---:|---:|
| NPLIB1 | random | 80.7% | 0.822 | 0.663 |
| NPLIB1 | scaffold | 38.2% | 0.293 | 0.112 |
| MassSpecGym | random | 97.1% | 0.953 | 0.925 |
| MassSpecGym | scaffold | 41.0% | 0.455 | 0.186 |

We also evaluated a best-effort formula-first variant: use the same-formula
nearest neighbour when a formula-matched training spectrum exists, otherwise
fall back to the all-training-set nearest neighbour. This gives the
formula-filtered baseline the advantage of formula matching without changing
the denominator. It improves over the all-training-set NN baseline, especially
on random splits, but tuned MIST remains higher on all four full-test Jaccard
comparisons.

| Dataset | Split | Tuned MIST Jaccard (↑) | NN, all train (↑) | Formula-first NN (↑) | Formula-only NN subset (↑) |
|---|---|---:|---:|---:|---:|
| NPLIB1 | scaffold | **0.275** (n=2,689) | 0.195 (n=2,689) | 0.212 (n=2,689; fallback=1,661) | 0.282 (n=1,028) |
| NPLIB1 | random | **0.721** (n=2,744) | 0.611 (n=2,744) | 0.720 (n=2,744; fallback=530) | 0.822 (n=2,214) |
| MassSpecGym | scaffold | **0.365** (n=16,042) | 0.230 (n=16,042) | 0.256 (n=16,042; fallback=9,471) | 0.455 (n=6,571) |
| MassSpecGym | random | **0.930** (n=16,250) | 0.850 (n=16,250) | 0.929 (n=16,250; fallback=473) | 0.953 (n=15,777) |

The next table asks a narrower diagnostic question: on the favorable
subset that the formula-filtered nearest-neighbour code can score, how much of
the advantage is due to real generalization versus train-test redundancy?
Candidates are restricted to the same molecular formula, entries with no formula
match are omitted, and the fingerprint oracle is also restricted to same-formula
training candidates. Leakage is the fraction of evaluated test entries whose
exact 2D InChIKey also appears in the training split.

Under this favorable subset evaluation, nearest-neighbour methods outperform
tuned MIST only on the random splits. Those random subsets also have very high
exact-molecule overlap with training data: 78.0% for NPLIB1 and 98.1% for
MassSpecGym. On scaffold splits, where the exact-overlap rate is much lower,
tuned MIST outperforms both nearest-neighbour baselines and is close to the
same-formula nearest-neighbour upper bound. This supports the interpretation
that train-test redundancy in the random split is a major driver of the very
high nearest-neighbour scores, rather than evidence that nearest-neighbour
search generalizes better than MIST.

| Dataset | Model                      |                Scaffold Jaccard (↑) |                   Random Jaccard (↑) |
| ----------- | -------------------------- | ----------------------------------: | -----------------------------------: |
| NPLIB1 | Tuned MIST                 |  **0.320** (n=1,028; **leak=4.2%**) |          0.801 (n=2,214; leak=78.0%) |
| NPLIB1 | Formula NN                 |          0.283 (n=1,028; leak=4.2%) |          0.829 (n=2,214; leak=78.0%) |
| NPLIB1 | Formula DreaMS NN          |          0.293 (n=1,028; leak=4.2%) |  **0.830** (n=2,214; **leak=78.0%**) |
| NPLIB1 | FP oracle (NN upper bound) |        _0.326_ (n=1,028; leak=4.2%) |        _0.869_ (n=2,214; leak=78.0%) |
| MassSpecGym | Tuned MIST                 | **0.503** (n=6,571; **leak=34.8%**) |         0.941 (n=15,777; leak=98.1%) |
| MassSpecGym | Formula NN                 |         0.455 (n=6,571; leak=34.8%) | **0.953** (n=15,777; **leak=98.1%**) |
| MassSpecGym | Formula DreaMS NN          |         0.470 (n=6,571; leak=34.8%) | **0.966** (n=15,777; **leak=98.1%**) |
| MassSpecGym | FP oracle (NN upper bound) |       _0.516_ (n=6,571; leak=34.8%) |       _0.987_ (n=15,777; leak=98.1%) |

The Comment interprets the random-to-scaffold drop for DreaMS as evidence of
poor generalization:

> DreaMS, the top model under the random split, drops sharply under the scaffold split, showing poor generalization.

The nearest-neighbour upper-bound calculation points to a more specific
interpretation. Random splits have high exact-molecule overlap with the training
set, and all methods, including the upper bound, drop when that overlap is
reduced in scaffold splits. This makes the random-to-scaffold change primarily a
property of the split and its leakage profile, rather than evidence that DreaMS
alone fails to generalize.

## kNN Sensitivity Check

The corrected nearest-neighbour baseline above uses the single most similar
training spectrum (`k=1`) without formula filtering. To check sensitivity to the
number of neighbours, we also evaluated binned-spectrum kNN variants with `k=3`
and `k=5`. The neighbour set is selected by spectrum cosine similarity from all
training spectra. We tested two fingerprint aggregations: majority vote on each
bit, and averaging each bit before scoring the continuous fingerprint. For the
binary Jaccard score, the
averaged fingerprint is thresholded at 0.5, so vote and average give identical
Jaccard for odd `k`.

These sweeps did not improve the nearest-neighbour baseline. In all four
dataset/split settings, `k=1` remains the strongest nearest-neighbour Jaccard
variant, and tuned MIST remains ahead of all tested kNN variants. Averaging
multiple neighbours can improve fingerprint cosine on scaffold splits, but it
does not close the gap to tuned MIST.

| Dataset     | Split    | Tuned MIST Jaccard (↑) | NN k=1 Jaccard (↑) | kNN k=3 Jaccard (↑) | kNN k=5 Jaccard (↑) |
|-------------|----------|-----------------------:|-------------------:|--------------------:|--------------------:|
| NPLIB1      | scaffold |    **0.275** (n=2,689) |              0.195 |               0.195 |               0.189 |
| NPLIB1      | random   |    **0.721** (n=2,744) |              0.611 |               0.512 |               0.456 |
| MassSpecGym | scaffold |   **0.365** (n=16,042) |              0.230 |               0.227 |               0.217 |
| MassSpecGym | random   |   **0.930** (n=16,250) |              0.850 |               0.735 |               0.668 |

| Dataset     | Split    | Tuned MIST FP Cosine (↑) |  NN k=1 FP Cosine (↑) | k=3 vote FP Cosine (↑) | k=3 average FP Cosine (↑) | k=5 vote FP Cosine (↑) | k=5 average FP Cosine (↑) |
|-------------|----------|-------------------------:|----------------------:|-----------------------:|--------------------------:|-----------------------:|--------------------------:|
| NPLIB1      | scaffold |                **0.505** |                 0.296 |                  0.315 |                     0.344 |                  0.322 |                     0.361 |
| NPLIB1      | random   |                **0.838** |                 0.671 |                  0.602 |                     0.654 |                  0.562 |                     0.633 |
| MassSpecGym | scaffold |                **0.514** |                 0.314 |                  0.325 |                     0.349 |                  0.331 |                     0.363 |
| MassSpecGym | random   |                **0.958** |                 0.869 |                  0.780 |                     0.830 |                  0.733 |                     0.800 |

## Notes on Interpretation

- The Jaccard tables evaluate fingerprint prediction, not full candidate
  retrieval.
- The MassSpecGym official table in the main README evaluates fixed-candidate
  retrieval and should be read as a separate benchmark setting.
- The formula-filtered nearest-neighbour score is conditional on candidate
  availability; the full-test score is the primary nearest-neighbour baseline.
- The FP-oracle upper bound shows the ceiling of any nearest-neighbour method
  that can only return training-set fingerprints. This gives an intuition for
  how difficult each split is.
