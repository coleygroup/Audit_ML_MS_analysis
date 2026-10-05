# Does Machine Learning Really Fail at Mass Spectrometry? Re-evaluating the MIST and Nearest-Neighbour Benchmarks

This repository contains code, configuration files, and evaluation scripts for reproducing fingerprint-prediction and nearest-neighbour baselines on open small-molecule MS/MS datasets. It is organized as a technical reproducibility report for the benchmark setting discussed in the *Nature Metabolism* Comment by Khoo and Barzilay (2026), "Why machine learning fails at mass spectrometry for small molecules" ([DOI: 10.1038/s42255-026-01544-6](https://doi.org/10.1038/s42255-026-01544-6)).

The Comment frames the problem around the observation that current models often fail to outperform "simple baseline methods." The title and claims are primarily supported by a single data table in the main text. After a systematic reproduction and reanalysis, **our central finding is that the benchmarking table is not a reliable basis for that conclusion because (1) the nearest-neighbour evaluation is not computed on the full test set, (2) the MIST models were evaluated at a checkpoint selected by a validation loss that does not track the reported metric, leaving them substantially under-trained, and with an unfitted decision threshold, and (3) the two random splits it relies on are dominated by train-test structural overlap.** We remain optimistic that machine learning is helping practitioners, and that continued empirical work will make a practical impact in this scientific domain.

Notes as of 10/05/2026:
> 1. The authors force-removed their git history and replaced it with a single commit. The numbers no longer match the paper anymore. The original repository that is consistent with the paper can be found at the branch [``original_repo_bkup``](https://github.com/coleygroup/Audit_ML_MS_analysis/tree/original_repo_bkup)
> 2. Given that the authors provide a new "nearest neighbor" method that uses subformula and [requires fitting parameters on the validaiton set](https://github.com/serenaklm/ML_MS_analysis/blob/main/benchmarked_models/nearest_neighbour/utils/subformula_fallback.py#L92), we show that how can you simply train MIST for more iterations to improve its accuracy. 

## Summary of Findings

Five issues change the interpretation of the Comment's benchmark:

- **The nearest-neighbour rows are not full-test scores.** They skip test entries without a same-formula training candidate through [this line of code](https://github.com/serenaklm/ML_MS_analysis/blob/1f88b9eab2656eb75c5915bf6ae4575848b44fe7/benchmarked_models/nearest_neighbour/02_compute_nn.py#L128) / [snapshot](https://github.com/coleygroup/Audit_ML_MS_analysis/blob/1f88b9eab2656eb75c5915bf6ae4575848b44fe7/benchmarked_models/nearest_neighbour/02_compute_nn.py#L128), discarding as much as 62% of some splits, while MIST is evaluated on the full test set. The discarded entries are the hard ones -- molecules whose formula was never seen in training -- so the reported score is not a full-test baseline. Restoring a common denominator removes the advantage of the machine-learning-free nearest neighbour on both scaffold splits, where MIST's published configuration then leads it (Table 1). The advantage survives on the two random splits, for reasons covered below.
- **The MIST models are under-trained, because the metric used to select them is not the metric being reported.** The Comment selects the MIST checkpoint by validation BCE loss but reports Jaccard after binarization. On the scaffold splits the loss stops improving within a handful of epochs while Jaccard keeps rising, so the reported model is an **epoch-10 (NPLIB1) and epoch-5 (MassSpecGym)** checkpoint. The predicted fingerprint is also binarized at a hard-coded `0.5` rather than at a fitted cut point; see the [MIST configuration comparison](benchmarked_models/mist/all_configs/README.md). Fixing only these two evaluation choices -- fitting the threshold on validation and taking the final checkpoint -- and changing **nothing about the Comment's model or objective**, raises MIST by **+0.086 and +0.101 Jaccard** on the two scaffold splits -- the single largest correction in this report. With every improvement applied to both sides (Table 2), MIST leads the tuned nearest-neighbour baselines on NPLIB1 scaffold by 0.018 and is level with them on MassSpecGym scaffold, while they remain ahead on the two random splits.
- **Fingerprint Jaccard is not the whole task.** Fingerprint Jaccard, the only metric benchmarked in the Comment, is not the standard endpoint for structure annotation. The practical task is to rank candidate molecules, usually by comparing predicted and candidate fingerprints with cosine similarity. Under candidate retrieval, MIST outperforms both nearest-neighbour baselines on **all four** dataset/split settings and at every reported cut-off, including MassSpecGym random, where it trails on binary Jaccard.
- **Train-test overlap explains the random splits, where nearest neighbour still wins.** On the full test sets, **62.9%** of NPLIB1 random and **95.3%** of MassSpecGym random test structures already appear in training by 2D InChIKey, against **1.6%** and **14.3%** on the scaffold splits. Retrieving a training fingerprint is close to a lookup in that regime: the nearest-neighbour upper bound reaches 0.972 Jaccard on MassSpecGym random and falls to 0.489 on its scaffold split. The random-to-scaffold drop the Comment attributes to [DreaMS](https://www.nature.com/articles/s41587-025-02663-3) generalizing poorly is therefore better read as a property of the split, and it is also where nearest neighbour retains an advantage over MIST.
- **Recent model development and standard benchmarks are not represented.** On the official MassSpecGym retrieval benchmark, a standard setting for method development, nearest neighbour does not dominate recent learned methods. Forward simulation and MIST-style models improve substantially when trained with better data, but these advances are not reflected in the Comment's benchmark.

This repository provides the corrected full-test scores, the formula-filtered diagnostic behind the inflated subset result, retrieval metrics, nearest-neighbour upper bounds, and the MIST configuration files used for reproduction.

## Main Results

Values are mean Jaccard similarity between predicted Morgan4096 fingerprints and ground-truth fingerprints. The first table reproduces the comparison as presented in the Comment. Daggered nearest-neighbour rows are formula-filtered subset results: **test entries without a same-formula training candidate are skipped**. Each split score includes the number of test entries contributing to that score. The differing `n` values are the technical issue: these rows are not full-test baselines and are not directly comparable to MIST. The skip is introduced in the nearest-neighbour implementation by [this line of code](https://github.com/serenaklm/ML_MS_analysis/blob/1f88b9eab2656eb75c5915bf6ae4575848b44fe7/benchmarked_models/nearest_neighbour/02_compute_nn.py#L128) / [snapshot](https://github.com/coleygroup/Audit_ML_MS_analysis/blob/1f88b9eab2656eb75c5915bf6ae4575848b44fe7/benchmarked_models/nearest_neighbour/02_compute_nn.py#L128), but MIST is scored on every test entry.

| Dataset     | Model              | Scaffold Jaccard (↑) | Random Jaccard (↑) |
| ----------- | ------------------ | -------------------: | -----------------: |
| NPLIB1      | MIST in Comment    |      0.238 (n=2,689) |    0.539 (n=2,744) |
| NPLIB1      | Nearest neighbour† |      0.293 (n=1,028) |    0.822 (n=2,212) |
| NPLIB1      | DreaMS NN†         |      0.293 (n=1,028) |    0.830 (n=2,212) |
| MassSpecGym | MIST in Comment    |     0.270 (n=16,042) |   0.766 (n=16,250) |
| MassSpecGym | Nearest neighbour† |      0.455 (n=6,571) |   0.953 (n=15,777) |
| MassSpecGym | DreaMS NN†         |      0.470 (n=6,571) |   0.966 (n=15,777) |

NIST2023 is not included in the reproduction. The authors declined our request to
access the exact NIST'23-derived artifacts used in the Comment. All MIST rows are
trained on the training split only, and the test split is never used for any
selection; the validation split is used to fit the binarization threshold and, for
the rows that follow the Comment's own protocol, to select the checkpoint.


### Table 1. The test-set denominator fix alone

Each method exactly as its authors specified it, with the single change that every
method is now scored on the same full test set. MIST is the Comment's own
configuration at its published `0.5` threshold and validation-loss-selected
checkpoint. The nearest-neighbour rows use the repository default
`same_formula_candidates_fallback` policy: candidates are still restricted to
training spectra sharing the query's molecular formula, but when no such spectrum
exists the search falls back to the whole training set instead of dropping the
query. 

| Dataset     | Model                                 | Scaffold Jaccard (↑) |   Random Jaccard (↑) |
| ----------- | ------------------------------------- | -------------------: | -------------------: |
| NPLIB1      | MIST in Comment, as published         |      0.238 (n=2,689) |      0.539 (n=2,744) |
| NPLIB1      | Nearest neighbour, formula-first      |      0.212 (n=2,689) |      0.720 (n=2,744) |
| NPLIB1      | DreaMS NN, formula-first              |      0.258 (n=2,689) |      0.744 (n=2,744) |
| _NPLIB1_    | _test structures already in training_ |               _1.6%_ |              _62.9%_ |
| MassSpecGym | MIST in Comment, as published         |     0.270 (n=16,042) |     0.766 (n=16,250) |
| MassSpecGym | Nearest neighbour, formula-first      |     0.256 (n=16,042) |     0.929 (n=16,250) |
| MassSpecGym | DreaMS NN, formula-first              |     0.306 (n=16,042) |     0.944 (n=16,250) |
| _MassSpecGym_ | _test structures already in training_ |            _14.3%_ |              _95.3%_ |

After fixing the denominator, MIST already outperforms machine-learning-free NN
baseline on both scaffold splits (0.238 against 0.212, 0.270 against 0.256), and
DreaMS NN -- itself a machine-learning method -- is ahead of both on every split.
Correcting the denominator alone therefore removes the headline result the
Comment's table rests on, without yet improving either method.

Before Table 2, it is worth separating what each change to MIST actually buys,
because almost all of it comes from two evaluation choices rather than from
changing the model. Both can be applied to the Comment's own configuration with
its BCE objective left untouched:

| Step (Comment's own BCE objective throughout)           | NPLIB1 scaffold | MassSpecGym scaffold | NPLIB1 random | MassSpecGym random |
| ------------------------------------------------------- | --------------: | -------------------: | ------------: | -----------------: |
| As published (threshold `0.5`, loss-selected checkpoint) |          0.238 |                0.270 |         0.539 |              0.766 |
| &nbsp;&nbsp;+ binarization threshold fitted on validation |         0.298 |                0.318 |         0.638 |              0.805 |
| &nbsp;&nbsp;+ 600-epoch budget, final checkpoint        |       **0.324** |            **0.371** |     **0.743** |          **0.838** |
| **Total change**                                        |      **+0.086** |           **+0.101** |    **+0.204** |         **+0.072** |

Switching from BCE to MIST's default cosine objective adds a further **+0.006 and +0.004** on the two scaffold splits. 
This the only *model* change in this repository, and distinguishes the `MIST, default objective` row.

### Table 2. Both methods with every improvement applied

Each side now gets its best available configuration. Nearest neighbour gets the
sub-formula fallback introduced in the authors' newly released code, whose four
hyperparameters `(w, alpha, T, k)` are selected over a 1,080-point grid on the
validation split. MIST gets the two evaluation fixes of this report -- a
validation-fitted binarization threshold and the final checkpoint of a 600-epoch
budget -- plus MIST's own default cosine objective.

| Dataset     | Model                                      | Scaffold Jaccard (↑) |   Random Jaccard (↑) |
| ----------- | ------------------------------------------ | -------------------: | -------------------: |
| NPLIB1      | MIST, default objective                    |  **0.330** (n=2,689) |      0.733 (n=2,744) |
| NPLIB1      | NN, sub-formula fallback‡                  |      0.312           |      0.752           |
| NPLIB1      | DreaMS NN, sub-formula fallback‡           |      0.311           |  **0.758**           |
| _NPLIB1_    | _test structures already in training_      |               _1.6%_ |              _62.9%_ |
| MassSpecGym | MIST, default objective                    | **0.375** (n=16,042) |     0.859 (n=16,250) |
| MassSpecGym | NN, sub-formula fallback‡                  |     0.365            |     0.935            |
| MassSpecGym | DreaMS NN, sub-formula fallback‡           |     0.371            | **0.947**            |
| _MassSpecGym_ | _test structures already in training_    |              _14.3%_ |              _95.3%_ |

‡ As reported in the authors' repository
([serenaklm/ML_MS_analysis](https://github.com/serenaklm/ML_MS_analysis), commit
`6817e9b`).

**Both tables tell the same story about the splits.** The settings where nearest
neighbour leads are the settings where the answer is already in the training set:
62.9% and 95.3% of test structures on the two random splits, against 1.6% and
14.3% on the scaffold splits. **But how a scaffold split carries exact-structural leakage? That's a separate question we cannot answer with code auditing.** 
Returning a memorized training fingerprint is close
to a lookup rather than a prediction in that regime, and the nearest-neighbour
upper bound reaches 0.972 Jaccard on MassSpecGym random. It is also worth noting
that DreaMS NN, which is itself built on a machine-learning embedding, outperforms
the vanilla nearest neighbour on every split in both tables -- a result that does
not support the general claim that machine learning does not work here.

The checkpoint-selection analysis behind the ladder above, together with additional
fingerprint-cosine, candidate-retrieval, and formula-filtered nearest-neighbour
diagnostic tables, is in [DETAILED_RESULTS.md](DETAILED_RESULTS.md).

## MassSpecGym Official Retrieval Split

A more established benchmarking approach in this field is to use open datasets with carefully curated splits, predefined candidate sets, and consistent evaluation metrics across papers. The official MassSpecGym split is one such setting. On this split, all methods use a fixed train/test split and fixed candidate sets. This differs from the Jaccard table above, but it is a widely used retrieval benchmark and is technically important for interpreting the role of nearest-neighbour baselines in application-relevant settings.

The Comment discusses MIST as the machine-learning representative, but MIST is only one inverse model: it maps a spectrum to a molecular fingerprint, which must then be used to rank candidate structures. Recent structure-annotation systems also include stronger inverse models such as [JESTR](https://arxiv.org/abs/2411.14464) and [FLARE](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12873900/), and forward models such as [ICEBERG](https://www.biorxiv.org/content/10.1101/2025.05.28.656653), which score candidates by predicting or simulating fragmentation behavior from molecular structures. The table below includes these more recent methods because they are directly relevant to the claim that nearest-neighbour search is a stronger practical baseline than modern learned models.

The following result is part of our paper, "MassSpecGym in the Wild: Uncovering and Correcting Evaluation Pitfalls in AI-Driven Molecule Discovery" ([arXiv:2606.19624](https://arxiv.org/abs/2606.19624)). Values are reported as top-k hit rate percentages and top-1 MCES distance. Brackets are 99.9% BCa bootstrap confidence intervals from 20,000 resamples.

| Method                                                                                                                |           Hit rate@1 (↑) |           Hit rate@5 (↑) |          Hit rate@20 (↑) |           MCES@1 (↓) |
|-----------------------------------------------------------------------------------------------------------------------|-------------------------:|-------------------------:|-------------------------:|---------------------:|
| Random                                                                                                                |        3.06 (2.64-3.52)% |    11.35 (10.60-12.12)%  |     27.74 (26.52-28.84)% |  13.87 (13.70-14.03) |
| [Nearest Neighbor](https://doi.org/10.1038/s42255-026-01544-6)                                                        |       9.58 (8.84-10.30)% |     22.26 (21.25-23.33)% |     39.92 (38.75-41.09)% |  13.82 (13.64-13.99) |
| [MIST](https://www.nature.com/articles/s42256-023-00708-3) (from [MassSpecGym 1.0](https://arxiv.org/abs/2410.23326)) |       9.57 (8.88-10.30)% |     22.11 (21.10-23.13)% |     41.12 (39.98-42.34)% |  12.75 (12.59-12.91) |
| [JESTR](https://arxiv.org/abs/2411.14464)                                                                             |     11.82 (11.03-12.68)% |     33.48 (32.33-34.68)% |     61.46 (60.21-62.63)% |  11.71 (11.54-11.87) |
| [FLARE](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12873900/)                                                       |     22.66 (21.63-23.74)% |     50.00 (48.78-51.22)% |     75.15 (74.10-76.22)% |     9.00 (8.82-9.18) |
| [ICEBERG 2.0](https://www.biorxiv.org/content/10.1101/2025.05.28.656653)                                              |     **36.18 (34.96-37.32)%** |     **60.70 (59.58-61.86)%** | **78.83 (77.74-79.78)%** |     **6.61 (6.43-6.79)** |

In this fixed-candidate retrieval setting, nearest neighbour is a stronger baseline than random and is close to the MassSpecGym 1.0 version of MIST, but it does not dominate recent fingerprint prediction methods. ICEBERG 2.0 substantially outperforms nearest neighbour, showing the advancements of machine learning overlooked in the original Comment.

## Conclusion

The Comment concludes
> Machine learning fails for small-molecule mass spectrometry

**After putting every method on the same test set, fitting the decision
threshold, and reading the MIST checkpoint at the end of training rather than at
the epoch a mismatched validation loss happens to favour, the empirical picture
changes substantially.** The largest single correction is not a change to the
model at all: on the Comment's own configuration and objective, fitting the
threshold and taking the final checkpoint are together worth +0.086 and +0.101
Jaccard on the two scaffold splits, because the published numbers come from
five- and ten-epoch checkpoints.

On the denominator fix alone, MIST's published configuration already leads the
machine-learning-free nearest neighbour on both scaffold splits. With every
improvement applied to both sides -- including the authors' own sub-formula
fallback, whose four hyperparameters are selected on validation -- MIST leads on
NPLIB1 scaffold by 0.018 and is level on MassSpecGym scaffold, within the
run-to-run spread. Nearest neighbour remains ahead on the two random splits, by
0.025 and 0.088, and those are the splits carrying 62.9% and 95.3% exact
structural overlap between test and training data, where the nearest-neighbour
upper bound reaches 0.972 Jaccard; that regime measures retrieval of memorized
structures more than prediction. Throughout, DreaMS NN -- a method built on a
machine-learning embedding -- outperforms the vanilla nearest neighbour on every
split, a result that does not support the claim that machine learning fails. And
under candidate retrieval, the endpoint that structure annotation actually uses,
MIST is ahead on all four settings at every reported cut-off. The strong form of
the Comment's conclusion does not survive these corrections; what remains is a
narrower and more interesting claim, that the random splits in this benchmark
measure memorization rather than prediction.

Zooming out, MIST was our group's first foray into mass spectrometry. We posted the MIST preprint at the end of 2022. At the time, deep neural network architectures were just beginning to be deployed on MS/MS data, building on a rich tradition of cheminformatics in mass spectrometry dating all the way back to [Dendral](https://en.wikipedia.org/wiki/Dendral) in the 1960s. This was part of the modern renaissance in machine learning methodologies for cheminformatics that many researchers, including the Comment's authors, helped catalyze. Since then, a wealth of empirical evidence has accumulated showing that newer machine-learning methods can outperform these baselines under standardized evaluations. The negative result in the Comment therefore appears to be specific to its evaluation choices and model settings, rather than a general limitation of machine learning for mass spectrometry.

Looking forward, there is an abundance of new developments in [modeling](https://arxiv.org/abs/2502.17874), [agentic](https://www.biorxiv.org/content/10.64898/2026.04.22.720103v1) [systems](https://www.biorxiv.org/content/10.64898/2026.06.23.734138v1), and [data](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15004319/v1). Now, more than ever, is precisely the time to be excited about applying AI and machine learning methods to metabolomics and mass spectrometry data.

## Reproducing the Experiments

The technical reproduction checklist is in [REPRODUCING_RESULTS.md](REPRODUCING_RESULTS.md). The checklist documents the expected file layout, MIST training and prediction commands, corrected nearest-neighbour commands, PubChem retrieval extraction, MassSpecGym candidate retrieval, and the table-building scripts used to regenerate the CSVs behind the README.

## Repository Layout

```text
benchmarked_models/
  mist/                  MIST configuration files, training, prediction, and model wrapper
  nearest_neighbour/     Corrected binned-spectrum, DreaMS, and oracle NN baselines
  evaluation/            Result collection and retrieval evaluation helpers
  common/                Shared parsing and benchmark utilities
FP_prediction/           Legacy code, not used
data/
  MGF_files/             Open split definitions and spectra inputs
  metadata/              Dataset metadata used for ID and formula mapping
scripts/                 Dataset preparation, remote launch, and summary helpers
results/                 Local result summaries and comparison CSVs
```

## Notes on Interpretation

NIST'23 results are omitted from this reproduction because those spectra are
licensed. We also avoid reporting a potentially inconsistent reproduction
because we do not have access to the exact NIST'23 data artifacts used in the
Comment, and NIST extraction and preprocessing choices can materially affect the
results. Additional metric-specific caveats are documented in
[DETAILED_RESULTS.md](DETAILED_RESULTS.md).

## Citation and Source Data

- Khoo, L.M.S. and Barzilay, R. "Why machine learning fails at mass spectrometry for small molecules." *Nature Metabolism* 8, 1247-1249 (2026). [https://doi.org/10.1038/s42255-026-01544-6](https://doi.org/10.1038/s42255-026-01544-6)
  - The code is available in the original repository at [https://github.com/serenaklm/ML_MS_analysis](https://github.com/serenaklm/ML_MS_analysis).
- Liu, H., Bushuiev, R., Lightheart, I. et al. "MassSpecGym in the Wild: Uncovering and Correcting Evaluation Pitfalls in AI-Driven Molecule Discovery." arXiv:2606.19624 (2026). [https://arxiv.org/abs/2606.19624](https://arxiv.org/abs/2606.19624)

### Benchmarked Methods

- **MIST**: Goldman, S., Wohlwend, J., Stražar, M. et al. "Annotating metabolite mass spectra with domain-inspired chemical formula transformers." *Nature Machine Intelligence* 5, 1140-1150 (2023). [https://www.nature.com/articles/s42256-023-00708-3](https://www.nature.com/articles/s42256-023-00708-3)
- **MassSpecGym**: Bushuiev, R. et al. "MassSpecGym: A benchmark for the discovery and identification of molecules." *NeurIPS 2024 Spotlight*. arXiv:2410.23326. [https://arxiv.org/abs/2410.23326](https://arxiv.org/abs/2410.23326)
- **DreaMS**: Bushuiev, R. et al. "Self-supervised learning of molecular representations from millions of tandem mass spectra using DreaMS." *Nature Biotechnology* (2025). [https://www.nature.com/articles/s41587-025-02663-3](https://www.nature.com/articles/s41587-025-02663-3)
- **JESTR**: Kalia, A., Chen, Y.Z., Krishnan, D. and Hassoun, S. "JESTR: Joint Embedding Space Technique for Ranking Candidate Molecules for the Annotation of Untargeted Metabolomics Data." *Bioinformatics* (2025). arXiv:2411.14464. [https://arxiv.org/abs/2411.14464](https://arxiv.org/abs/2411.14464)
- **FLARE**: Chen, Y.Z., Rushing, B. and Hassoun, S. "FLARE: Fine-grained Learning for Alignment of spectra-molecule REpresentation Enhances Metabolite Annotation." (2026). [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12873900/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12873900/)
- **ICEBERG 2.0**: Wang, R., Manjrekar, M., Mahjour, B. et al. "Neural Spectral Prediction for Structure Elucidation with Tandem Mass Spectrometry." *bioRxiv* (2025). [https://www.biorxiv.org/content/10.1101/2025.05.28.656653](https://www.biorxiv.org/content/10.1101/2025.05.28.656653)

## Changelog

### 20261005

#### `Corrected MIST` renamed to `MIST, default objective`; the Comment's own run is now the headline

A naming consequence of the 20261003 finding. The two corrections that carry the
result -- the validation-fitted threshold and the final checkpoint -- are applied
to every MIST row, including the Comment's own BCE configuration. The row
previously called `Corrected MIST` is therefore not distinguished by a
*correction* at all; it is distinguished by using MIST's default cosine
objective, which on its own is worth +0.006 and +0.004 Jaccard on the two
scaffold splits and **-0.010** on NPLIB1 random. The old name implied the fix was
the loss function, which the matched-budget runs refuted.

- The row is renamed **`MIST, default objective`** throughout the tables, the
  configuration notes and the table-building scripts. Earlier changelog entries
  keep the old name, since that is what the row was called at the time.
- **`MIST in Comment, final checkpoint` is now the headline row.** It changes
  nothing about the Comment's model or objective -- only the epoch budget and
  which checkpoint is read -- and reaches 0.324 / 0.743 / 0.371 / 0.838. The
  default-objective row is reported beside it rather than in place of it, because
  cosine is MIST's documented default and because it remains the best
  configuration we found on both scaffold splits (0.330 and 0.375).
- `*_original_mist_e600_config.yaml` is correspondingly labelled the headline
  configuration in
  [all_configs/README.md](benchmarked_models/mist/all_configs/README.md), and
  section 4 of [REPRODUCING_RESULTS.md](REPRODUCING_RESULTS.md) reorders so that
  the headline configuration is trained first.

#### Main Results restructured into two tables

The headline comparison is now staged, so that each table holds one kind of
comparison rather than mixing them:

- **Table 1** applies only the denominator fix. Every method is exactly as its
  authors specified it -- MIST at its published `0.5` threshold and
  validation-loss-selected checkpoint, nearest neighbour with its formula-first
  candidate policy -- and nothing is tuned on either side. MIST's published
  configuration leads the machine-learning-free nearest neighbour on both scaffold
  splits; DreaMS NN leads both on every split.
- **Table 2** gives each side every improvement available to it. Nearest neighbour
  gets the sub-formula fallback from the authors' released code, whose four
  hyperparameters `(w, alpha, T, k)` are selected over a 1,080-point grid on the
  validation split -- so it is a tuned method, not a parameter-free baseline. MIST
  gets the fitted threshold, the final checkpoint and its default objective. MIST
  leads NPLIB1 scaffold by 0.018 and is **level** on MassSpecGym scaffold (+0.004,
  inside the ±0.003 run-to-run spread); nearest neighbour leads the random splits
  by 0.025 and 0.088.

The threshold / budget / checkpoint ladder now sits between the two tables, where
it explains how MIST gets from one to the other.

The sub-formula rows are quoted from the authors' repository
([serenaklm/ML_MS_analysis](https://github.com/serenaklm/ML_MS_analysis), commit
`6817e9b`) and have **not** been reproduced in our pipeline. Two cross-checks bound
the risk and are stated in the README: their `best-match fallback` nearest
neighbour, the same method as our formula-first rows, agrees with ours to 0.000 on
MassSpecGym random and 0.002 on MassSpecGym scaffold; and their MIST column agrees
with our reproduction of the same configuration to within 0.003 on both scaffold
splits, but differs by 0.092 on MassSpecGym random, which makes that one cell the
least safe in the table.

Summary-of-findings bullets 1 and 2 and the Conclusion were reconciled with the
new tables: the earlier claim that MIST "outperforms the corrected
nearest-neighbour baselines on both scaffold splits by 0.072 and 0.069" held
against our own nearest-neighbour baselines but not against the authors' tuned
sub-formula variants, and has been replaced by the narrower statement above.

#### README narrowed to the headline comparison

Two sections were taken out of [README.md](README.md):

- **"Why the Published MIST Is Under-Trained" and its subsection "The epoch at
  which validation plateaus tracks leakage" moved to
  [DETAILED_RESULTS.md](DETAILED_RESULTS.md).** The finding is unchanged and still
  drives the headline tables; it is supporting analysis rather than a result, so it
  now sits with the other diagnostics. The threshold / budget / checkpoint ladder
  stays in the README between Tables 1 and 2.
- **"If You Want the Random Splits, Overfit Harder" was removed.** With Table 2
  giving both methods their best configuration, a separate section on what further
  MIST tuning buys no longer carried the argument. Removing it also drops the
  spectral-augmentation, nearest-training-fingerprint and model-ensemble
  observations and the `Tuned MIST` comparison table from the README. The
  `*_mistv2_h1024_sched_e600_config.yaml` family remains in the repository and is
  documented in
  [all_configs/README.md](benchmarked_models/mist/all_configs/README.md); its
  scores are read from each run's `test_performance.json` and are no longer
  reported in any document.

No MIST or nearest-neighbour numbers were recomputed in this revision; the
renaming, reordering and restructuring are presentational, and the sub-formula
figures are newly quoted from the authors' repository.

### 20261003

#### Checkpoint selection identified as the dominant correction; all MIST rows now report the final checkpoint

Earlier revisions of this file attributed the gap between the published MIST row
and ours to two things: the unfitted `0.5` binarization threshold and the BCE
objective. Re-running both objectives to a matched 600-epoch budget shows that
attribution was wrong in its emphasis.

The Comment selects its checkpoint with `val_monitor: val_bce_loss` but reports
Jaccard after binarization, and those two quantities part company early. Under the
Comment's own configuration, `val_bce_loss` is minimized at **epoch 5** on
MassSpecGym scaffold and **epoch 10** on NPLIB1 scaffold, while thresholded
Jaccard keeps improving for hundreds of epochs. Reading the final checkpoint of
the same run instead of the monitor-selected one is worth **+0.046 and +0.029**
Jaccard on those two splits, and **+0.107** on NPLIB1 random.

Consequences for this repository:

- **All MIST rows now report the final checkpoint of a 600-epoch budget**
  (`save_last: True`, scored from `last.ckpt`). `Corrected MIST` moves to 0.330 /
  0.733 / 0.375 / 0.859 (NPLIB1 scaffold / NPLIB1 random / MassSpecGym scaffold /
  MassSpecGym random) from 0.317 / 0.712 / 0.368 / 0.813.
- **A `MIST in Comment, final checkpoint` row was added** to the headline table:
  the Comment's own configuration and objective, corrected only in threshold,
  budget and checkpoint, reaching 0.324 / 0.743 / 0.371 / 0.838. Fitting the
  threshold and taking the final checkpoint are together worth **+0.086 and
  +0.101** on the two scaffold splits with no change to the model.
- **The contribution of the cosine objective is much smaller than previously
  reported**: +0.006 and +0.004 on the two scaffold splits, against the +0.019 and
  +0.051 reported before, because most of that apparent advantage was the BCE run
  being stopped at epoch 5-10. On NPLIB1 random the Comment's BCE configuration is
  in fact slightly stronger (0.743 against 0.733). The objective change is kept
  because it is MIST's documented default, not because it carries the result.
- **Nearest neighbour's remaining advantage narrows to one cell.** On NPLIB1 random
  the methods are level (0.733 and 0.743 for MIST against 0.720 and 0.744 for the
  two nearest-neighbour baselines). MassSpecGym random, at 95.3% leakage, is the
  only setting where nearest neighbour still leads clearly, by 0.085.
- Adds `*_original_mist_e600_config.yaml`: the Comment's BCE objective at the same
  600-epoch budget, differing from `*_mist_e600_config.yaml` in exactly three lines
  (`exp_name`, `loss_fn`, `val_monitor`).
- `predict.py` gains `--output_dir`, so the monitor-selected and final checkpoints
  of one run can both be scored without the second overwriting the first.

Two complications are worth stating, because the final checkpoint is not better at
everything:

- **Fingerprint cosine on the scaffold splits is slightly lower** at the final
  checkpoint (0.520 and 0.550) than at the `val_cos_loss`-selected one (0.529 and
  0.571), and **candidate retrieval past top-1 on the scaffold splits is lower
  too** (MassSpecGym scaffold top-5 falls from 0.395 to 0.382, top-10 from 0.490 to
  0.473, while top-1 rises from 0.219 to 0.231). The checkpoint that maximizes
  thresholded Jaccard does not have the best-calibrated continuous output, and
  cosine and deep-ranking metrics depend on that calibration. We report one
  checkpoint for all metrics -- the one that optimizes the metric the Comment
  reports -- rather than selecting per metric, and record the trade-off here and in
  [DETAILED_RESULTS.md](DETAILED_RESULTS.md).
- **The NPLIB1 nearest-neighbour and DreaMS retrieval rows have been corrected
  upward.** They predated the nearest-neighbour realignment in the 20260910 entry
  below and had never been regenerated against it, which understated both NPLIB1
  baselines (random top-1 0.348 to 0.439 for nearest neighbour, 0.419 to 0.456 for
  DreaMS NN). The MassSpecGym rows were unaffected, as that realignment was
  NPLIB1-specific. Corrected MIST still leads every retrieval cell.

The previous section "If You Want the Random Splits, Just Train Longer" has been
replaced by two sections: "Why the Published MIST Is Under-Trained", which reports
the checkpoint-selection effect, and "If You Want the Random Splits, Overfit
Harder", which keeps the capacity-versus-leakage result. The epoch-budget framing
was demoted because raising the budget alone changes little: the monitor discards
the extra epochs unless the checkpoint rule is changed too.

### 20261001

#### Epoch budget isolated as the simplest random-split lever

Adds `*_mist_e600_config.yaml`: `Corrected MIST` with `max_epochs: 600` and
`patience` raised so early stopping cannot truncate the budget. Nothing else
changes -- same `hidden_size: 256`, batch size, seed, objective and absence of an
LR schedule.

Running it against the 200-epoch budget isolates the epoch effect: **+0.011 and
+0.032 Jaccard on the two random splits, and exactly 0.000 on both scaffold
splits**, where the same checkpoint is selected in both runs because validation
stops improving at epoch 18 (MassSpecGym) and epoch 41 (NPLIB1). The epoch at
which validation plateaus tracks leakage directly: 18 and 41 on the scaffold
splits versus 197 and 199 of 200 on the random splits. This is the cleanest
single piece of evidence that the random-split gap is memorization rather than a
model limitation, and it is reported in the README section "If You Want the
Random Splits, Just Train Longer".

Separately, we found that `val_monitor: val_cos_loss` and the reported
Jaccard-after-thresholding metric disagree on the scaffold splits: taking the
final checkpoint instead of the `val_cos_loss`-selected one is worth **+0.010
(NPLIB1 scaffold) and +0.011 (MassSpecGym scaffold)** Jaccard, and under 0.002 on
both random splits, with validation Jaccard agreeing in every case. At the time of
this entry the published MIST rows still used `val_cos_loss` selection. This was
adopted repository-wide in the 20261003 entry above, where the same effect under
the Comment's BCE objective turns out to be several times larger.

### 20260924

#### Added a `Tuned MIST` row showing what hyperparameter search buys

`*_mistv2_h1024_sched_e600_config.yaml` adds a fourth config family: the Corrected
MIST architecture widened to `hidden_size: 1024` and trained for `600` epochs under
a cosine schedule with warmup and EMA. It is reported as `Tuned MIST` and is **not**
the headline configuration. It gains +0.028 and +0.118 Jaccard on the two random
splits and loses 0.010 on MassSpecGym scaffold, which is the point of including it:
the benefit tracks train-test leakage rather than model quality. `Corrected MIST`
remains the headline row and is the best configuration we found on both scaffold
splits, on test and on validation.

### 20260910

#### Corrected MIST replaces the previously reported tuned MIST configuration

The `Tuned MIST` rows previously reported here came from a hyperparameter family
derived from MIST vFRIGID, the MIST variant used inside the FRIGID model. Those
rows have been replaced by **`Corrected MIST`**, which keeps the Comment's own
architecture and training budget and changes only the training objective
(`loss_fn: bce` to `cosine`, with `val_monitor` moved to match) plus the fitted
decision threshold. Every MIST number in [README.md](README.md) and
[DETAILED_RESULTS.md](DETAILED_RESULTS.md) is now produced against unmodified
upstream MIST from the [`main_v2` branch](https://github.com/samgoldman97/mist/tree/main_v2);
no fork or vendored MIST variant is used anywhere in this repository. This makes
the MIST comparison a single-variable change from the configuration the Comment
reported, rather than a comparison against a separately tuned model family.

#### Nearest neighbour now defaults to formula-first with a fallback

The default candidate policy is `same_formula_candidates_fallback`: candidates are
restricted to training spectra sharing the query's molecular formula, and when no
such spectrum exists the search falls back to the whole training set rather than
skipping the query. Previously the headline nearest-neighbour rows used
`all_train_candidates`, which discards the formula prior and understates the
baseline. All nearest-neighbour rows are scored at coverage 1.000 against the full
MGF-derived training pool. This raises the nearest-neighbour numbers relative to
earlier revisions of this file and is the reason nearest neighbour now leads on
both random splits.

### 20260806

#### MIST rows regenerated on the training split only

All MIST numbers were regenerated from checkpoints trained on the **training split
only** (`train_with_val: False`).
[Earlier commits](https://github.com/coleygroup/Audit_ML_MS_analysis/tree/8327c017ef7fd78dd63ec97c5f8309dedd42392d)
merged the validation split into training, which supplied about 21% more training
data than the `MIST in Comment` configs received, though `MIST in Comment` used the
validation split to select the best model.
