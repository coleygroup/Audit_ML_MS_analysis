# Detailed Benchmark Results

This file contains supplementary result tables for the reproducibility report in
[README.md](README.md). The main README keeps the headline result tables and the
MassSpecGym official retrieval split; this file keeps the additional
full-test retrieval, fingerprint-cosine, k-nearest-neighbour sensitivity, and
formula-filtered diagnostic tables.

## MIST Configuration Notes

The main README reports four MIST rows: `MIST in Comment` at the original
hard-coded `0.5` threshold, the same model at a validation-fitted threshold, the
same model again at the final checkpoint of a 600-epoch budget, and `Corrected
MIST`. The configuration differences are documented in
[benchmarked_models/mist/all_configs/README.md](benchmarked_models/mist/all_configs/README.md).

The `MIST, default objective` row changes exactly one thing about the Comment's **model**: it trains
with MIST's default cosine objective instead of BCE, with `val_monitor` moved to
match (without which the cosine run silently keeps its epoch-1 checkpoint).
Architecture, batch size, seed and MAGMa auxiliary supervision are unchanged, and
no fork of MIST is involved — every row runs against unmodified upstream MIST
from the [`main_v2` branch](https://github.com/samgoldman97/mist/tree/main_v2).

Two **evaluation** choices separate the corrected rows from the published ones, and
they account for most of the difference between them. Neither touches the model, so
both can be — and in the headline table are — applied to the Comment's own
configuration and objective as well:

1. The epoch budget is raised to 600 and the score is read from the **final**
   checkpoint rather than the one the validation loss selects. The validation loss
   and the reported Jaccard do not move together, and under the Comment's BCE
   objective the monitor returns an epoch-5 (MassSpecGym scaffold) or epoch-10
   (NPLIB1 scaffold) checkpoint. See
   [Why the Published MIST Is Under-Trained](#why-the-published-mist-is-under-trained) below.
2. The fingerprint binarization threshold is fitted on the **validation** split
   rather than assumed to be `0.5`. The optimal cut is objective-dependent
   (0.82-0.86 under BCE, 0.20-0.32 under cosine), so a fixed `0.5` does not treat
   the two objectives alike.

## Why the Published MIST Is Under-Trained

The Comment's configuration selects the MIST checkpoint with
`val_monitor: val_bce_loss` and reports mean Jaccard after binarizing the
predicted fingerprint. Those two quantities do not move together. The validation
loss flattens early while thresholded Jaccard keeps improving, so the checkpoint
the monitor hands back is not the checkpoint that performs best on the metric
being reported.

Holding the Comment's configuration fixed, including its BCE objective, and
changing only which epoch of the same 600-epoch run is read out:

| Dataset     | Split    | Epoch the monitor selects | Jaccard there | Jaccard at epoch 600 |          Δ |
| ----------- | -------- | ------------------------: | ------------: | -------------------: | ---------: |
| MassSpecGym | scaffold |               **epoch 5** |         0.325 |                0.371 | **+0.046** |
| NPLIB1      | scaffold |              **epoch 10** |         0.295 |                0.324 | **+0.029** |
| NPLIB1      | random   |                  epoch 45 |         0.636 |                0.743 | **+0.107** |
| MassSpecGym | random   |                 epoch 526 |         0.835 |                0.838 |     +0.003 |

On the two splits that actually test generalization, the MIST row in the Comment is
a five-epoch and a ten-epoch model. The same effect exists under MIST's default
cosine objective but is weaker, because `val_cos_loss` tracks the reported
metric better: it selects epoch 18 (MassSpecGym scaffold) and epoch 41 (NPLIB1
scaffold), and reading epoch 600 instead is worth +0.011 and +0.010 rather than
+0.046 and +0.029.

Two consequences follow. First, **raising the epoch budget alone does almost
nothing, because the monitor discards the extra epochs.** On NPLIB1 random, going
from a 200- to a 600-epoch budget moves the loss-selected score from 0.638 to
0.636 — unchanged within run-to-run variation — while reading the final
checkpoint of that same 600-epoch run gives 0.743. Second, because this is a
checkpoint-selection effect and not a tuning effect, **it requires no
hyperparameter search**: it is a change to which file is loaded at evaluation
time.

### The epoch at which validation plateaus tracks leakage

Which epoch the monitor settles on is not arbitrary. It orders with how much of
the test set is already present in training:

| Dataset     | Split    | Test structures already in training | Monitor plateaus (BCE) | Monitor plateaus (cosine) |
| ----------- | -------- | ----------------------------------: | ---------------------: | ------------------------: |
| NPLIB1      | scaffold |                                1.6% |               epoch 10 |                  epoch 41 |
| MassSpecGym | scaffold |                               14.3% |                epoch 5 |                  epoch 18 |
| NPLIB1      | random   |                               62.9% |               epoch 45 |                 epoch 567 |
| MassSpecGym | random   |                               95.3% |              epoch 526 |                 epoch 595 |

On the scaffold splits the model has extracted what it can within a few dozen
epochs. On the random splits validation keeps improving almost to the end of a
600-epoch budget. A split on which a model never stops benefiting from more
gradient steps over data it has already seen is a split that rewards
memorization, and these are exactly the splits where 62.9% and 95.3% of test
structures are already in the training set.


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
| NPLIB1 | MIST in Comment, final checkpoint | 0.486 (n=2,689) | 0.821 (n=2,744) |
| NPLIB1 | MIST, default objective | **0.520** (n=2,689) | **0.827** (n=2,744) |
| NPLIB1 | Nearest neighbour | 0.321 (n=2,689) | 0.770 (n=2,744) |
| NPLIB1 | DreaMS NN | 0.376 (n=2,689) | 0.796 (n=2,744) |
| MassSpecGym | MIST in Comment | 0.457 (n=16,042) | 0.858 (n=16,250) |
| MassSpecGym | MIST in Comment, final checkpoint | 0.517 (n=16,042) | 0.884 (n=16,250) |
| MassSpecGym | MIST, default objective | **0.550** (n=16,042) | 0.914 (n=16,250) |
| MassSpecGym | Nearest neighbour | 0.344 (n=16,042) | 0.940 (n=16,250) |
| MassSpecGym | DreaMS NN | 0.405 (n=16,042) | **0.955** (n=16,250) |

Fingerprint cosine changes the picture on one of the two random splits. Corrected
MIST leads on both scaffold splits, as it does on Jaccard, and it also leads on
NPLIB1 random (0.827 versus 0.796 for DreaMS). Only MassSpecGym random, the split
with 95.3% structural leakage, keeps nearest neighbour ahead on this metric.
Binarizing at a single cut discards confidence information that the cosine metric
retains, which is part of why the two metrics disagree.

One consequence of reading the final checkpoint rather than the loss-selected one
is visible here: it is chosen to maximize thresholded Jaccard, and that is not the
same checkpoint that maximizes fingerprint cosine. On the scaffold splits the
earlier, `val_cos_loss`-selected checkpoint of the 200-epoch run scored slightly
*higher* FP cosine (0.529 and 0.571) than the final checkpoint reported here
(0.520 and 0.550), while scoring lower Jaccard (0.317 and 0.368 against 0.330 and
0.375). The two metrics prefer different checkpoints; we report the one that
optimizes the metric the Comment reports, and note the trade-off.

Retrieval is evaluated by 2D InChIKey hit rate after ranking candidates with the
predicted fingerprint. For NPLIB1, candidates are PubChem structures with the
same chemical formula. The PubChem formula-to-structure map used for this
evaluation is available as `pubchem_formulae_inchikey.hdf5` from [Zenodo](https://zenodo.org/records/15529765/files/pubchem_formulae_inchikey.hdf5);
formulas absent from that file were completed with the PubChem API. For
MassSpecGym, candidates are the official formula and mass candidate sets. The
nearest-neighbour rows rank candidates with the same formula-first predictions
used in the headline table.

**MIST leads every retrieval setting, including the two random splits
where it trails on binary Jaccard.** This is the metric closest to the task the
Comment motivates, which is ranking candidate structures.

| Dataset | Candidate set | Model | Scaffold top-1 / top-5 / top-10 (↑) | Random top-1 / top-5 / top-10 (↑) |
| ----------- | ------------------ | ------------------ | ----------------------------------: | --------------------------------: |
| NPLIB1 | PubChem same formula | MIST, default objective | **0.124 / 0.261 / 0.348** (n=2,689) | **0.618 / 0.738 / 0.785** (n=2,744) |
| NPLIB1 | PubChem same formula | Nearest neighbour | 0.050 / 0.126 / 0.193 (n=2,689) | 0.439 / 0.611 / 0.663 (n=2,744) |
| NPLIB1 | PubChem same formula | DreaMS NN | 0.088 / 0.190 / 0.259 (n=2,689) | 0.456 / 0.638 / 0.688 (n=2,744) |
| MassSpecGym | Official formula | MIST, default objective | **0.231 / 0.382 / 0.473** (n=16,042) | **0.840 / 0.919 / 0.938** (n=16,250) |
| MassSpecGym | Official formula | Nearest neighbour | 0.093 / 0.181 / 0.248 (n=16,042) | 0.600 / 0.745 / 0.787 (n=16,250) |
| MassSpecGym | Official formula | DreaMS NN | 0.144 / 0.267 / 0.342 (n=16,042) | 0.633 / 0.787 / 0.832 (n=16,250) |
| MassSpecGym | Official mass | MIST, default objective | **0.347 / 0.541 / 0.632** (n=16,042) | **0.891 / 0.952 / 0.965** (n=16,250) |
| MassSpecGym | Official mass | Nearest neighbour | 0.122 / 0.181 / 0.219 (n=16,042) | 0.696 / 0.794 / 0.824 (n=16,250) |
| MassSpecGym | Official mass | DreaMS NN | 0.198 / 0.300 / 0.347 (n=16,042) | 0.737 / 0.844 / 0.877 (n=16,250) |

On MassSpecGym random, nearest neighbour leads MIST by 0.070 Jaccard but
trails it by 0.240 top-1 hit rate under the official formula candidate set. A
retrieved training fingerprint scores well against its own molecule's fingerprint,
but that does not translate into ranking the correct structure highly within a
large candidate set.

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
The table below compares all three policies against MIST on the full test set.

| Dataset | Split | MIST, default objective (↑) | NN, all train (↑) | NN, formula-first (default) (↑) | _NN, formula-only subset_ |
|---|---|---:|---:|---:|---:|
| NPLIB1 | scaffold | **0.330** (n=2,689) | 0.195 (n=2,689) | 0.212 (n=2,689; fallback=1,661) | _0.293 (n=1,028)_ |
| NPLIB1 | random | **0.733** (n=2,744) | 0.611 (n=2,744) | 0.720 (n=2,744; fallback=532) | _0.822 (n=2,212)_ |
| MassSpecGym | scaffold | **0.375** (n=16,042) | 0.230 (n=16,042) | 0.256 (n=16,042; fallback=9,471) | _0.455 (n=6,571)_ |
| MassSpecGym | random | 0.859 (n=16,250) | 0.850 (n=16,250) | **0.929** (n=16,250; fallback=473) | _0.953 (n=15,777)_ |

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

| Dataset | Split | Leakage (test structures already in training) | FP oracle (NN ceiling) | Best NN | MIST, default objective |
|---|---|---:|---:|---:|---:|
| NPLIB1 | scaffold | **1.6%** | 0.435 | 0.258 | **0.330** |
| NPLIB1 | random | **62.9%** | 0.822 | **0.744** | 0.733 |
| MassSpecGym | scaffold | **14.3%** | 0.489 | 0.306 | **0.375** |
| MassSpecGym | random | **95.3%** | 0.972 | **0.944** | 0.859 |

The two random splits are close to lookup problems. On MassSpecGym random, 95.3%
of test structures are already present in training and the retrieval ceiling is
0.972; DreaMS nearest neighbour reaches 0.944, within 0.028 of a ceiling that
exists only because the answers are in the training set. On the scaffold splits
that ceiling falls to 0.435 and 0.489, and MIST exceeds every nearest-neighbour
variant — on MassSpecGym scaffold it also exceeds 75% of the retrieval ceiling
while both NN baselines fall well short of it.

The same picture holds on the favourable subset the formula-filtered code
actually scores, where leakage is even more extreme on the random splits:

| Dataset | Model | Scaffold Jaccard (↑) | Random Jaccard (↑) |
| ----------- | -------------------------- | ----------------------------------: | -----------------------------------: |
| NPLIB1 | MIST, default objective | **0.359** (n=1,028; **leak=4.2%**) | 0.808 (n=2,212; leak=78.0%) |
| NPLIB1 | Formula NN | 0.293 (n=1,028; leak=4.2%) | 0.822 (n=2,212; leak=78.0%) |
| NPLIB1 | Formula DreaMS NN | 0.293 (n=1,028; leak=4.2%) | **0.830** (n=2,212; leak=78.0%) |
| NPLIB1 | FP oracle (NN upper bound) | _0.326_ (n=1,028; leak=4.2%) | _0.869_ (n=2,212; leak=78.0%) |
| MassSpecGym | MIST, default objective | **0.498** (n=6,571; leak=34.8%) | 0.875 (n=15,777; leak=98.1%) |
| MassSpecGym | Formula NN | 0.455 (n=6,571; leak=34.8%) | 0.953 (n=15,777; leak=98.1%) |
| MassSpecGym | Formula DreaMS NN | 0.470 (n=6,571; leak=34.8%) | **0.966** (n=15,777; leak=98.1%) |
| MassSpecGym | FP oracle (NN upper bound) | _0.516_ (n=6,571; leak=34.8%) | _0.987_ (n=15,777; leak=98.1%) |

On NPLIB1 scaffold, MIST **exceeds the nearest-neighbour ceiling** (0.359
versus 0.326), which a retrieval method cannot do by construction. On MassSpecGym
scaffold it reaches 0.498 against a ceiling of 0.516 — 96% of the best score any
retrieval-only method could achieve — and leads both nearest-neighbour variants on
this subset, which it did not in revisions of this file before 20261003. On the
random subsets every method including the ceiling is compressed into a narrow high
band, because 78% and 98% of those spectra have their own structure in the training
set.

The Comment interprets the random-to-scaffold drop for DreaMS as evidence of poor
generalization:

> DreaMS, the top model under the random split, drops sharply under the scaffold split, showing poor generalization.

The ceiling calculation points to a more specific reading. Every method drops
between the random and scaffold splits, including the oracle, which contains no
model at all. The drop is therefore primarily a property of the split and its
leakage profile rather than evidence that DreaMS specifically fails to
generalize, and it is also why nearest neighbour retains an advantage over MIST
on the random splits while losing it on the scaffold splits.

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
