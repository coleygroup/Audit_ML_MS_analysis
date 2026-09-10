# Does Machine Learning Really Fail at Mass Spectrometry? Re-evaluating the MIST and Nearest-Neighbour Benchmarks

This repository contains code, configuration files, and evaluation scripts for reproducing fingerprint-prediction and nearest-neighbour baselines on open small-molecule MS/MS datasets. It is organized as a technical reproducibility report for the benchmark setting discussed in the *Nature Metabolism* Comment by Khoo and Barzilay (2026), "Why machine learning fails at mass spectrometry for small molecules" ([DOI: 10.1038/s42255-026-01544-6](https://doi.org/10.1038/s42255-026-01544-6)).

The Comment frames the problem around the observation that current models often fail to outperform "simple baseline methods." The title and claims are primarily supported by a single data table in the main text. After a systematic reproduction and reanalysis, **our central finding is that the benchmarking table is not a reliable basis for that conclusion because (1) the nearest-neighbour evaluation is not computed on the full test set, (2) the MIST models were evaluated with under-optimized training settings and an unfitted decision threshold, and (3) the two random splits it relies on are dominated by train-test structural overlap.** We remain optimistic that machine learning is helping practitioners, and that continued empirical work will make a practical impact in this scientific domain.

## Summary of Findings

Five issues change the interpretation of the Comment's benchmark:

- **The nearest-neighbour rows are not full-test scores.** They skip test entries without a same-formula training candidate through [this line of code](https://github.com/serenaklm/ML_MS_analysis/blob/1f88b9eab2656eb75c5915bf6ae4575848b44fe7/benchmarked_models/nearest_neighbour/02_compute_nn.py#L128), discarding as much as 62% of some splits, while MIST is evaluated on the full test set. The discarded entries are the hard ones -- molecules whose formula was never seen in training -- so the reported score is not a full-test baseline. Restoring a common denominator removes the headline nearest-neighbour advantage on both scaffold splits; it survives on the two random splits, for reasons covered below.
- **MIST was evaluated with under-optimized settings and an unfitted decision threshold.** The Comment trained MIST with a BCE objective rather than MIST's own default cosine objective, and binarized the predicted fingerprint at a hard-coded `0.5` rather than fitting that cut point; see the [MIST configuration comparison](benchmarked_models/mist/all_configs/README.md). Fitting the threshold on the validation split and restoring the cosine objective raises MIST on every split. With both corrections, **MIST outperforms the corrected nearest-neighbour baselines on both scaffold splits**, the settings that require generalization to unseen scaffolds.
- **Fingerprint Jaccard is not the whole task.** Fingerprint Jaccard, the only metric benchmarked in the Comment, is not the standard endpoint for structure annotation. The practical task is to rank candidate molecules, usually by comparing predicted and candidate fingerprints with cosine similarity. Under candidate retrieval, MIST outperforms both nearest-neighbour baselines on **all four** dataset/split settings, including the two random splits where it trails on binary Jaccard.
- **Train-test overlap explains the random splits, where nearest neighbour still wins.** On the full test sets, **62.9%** of NPLIB1 random and **95.3%** of MassSpecGym random test structures already appear in training by 2D InChIKey, against **1.6%** and **14.3%** on the scaffold splits. Retrieving a training fingerprint is close to a lookup in that regime: the nearest-neighbour upper bound reaches 0.972 Jaccard on MassSpecGym random and falls to 0.489 on its scaffold split. The random-to-scaffold drop the Comment attributes to [DreaMS](https://www.nature.com/articles/s41587-025-02663-3) generalizing poorly is therefore better read as a property of the split, and it is also where nearest neighbour retains an advantage over MIST.
- **Recent model development and standard benchmarks are not represented.** On the official MassSpecGym retrieval benchmark, a standard setting for method development, nearest neighbour does not dominate recent learned methods. Forward simulation and MIST-style models improve substantially when trained with better data, but these advances are not reflected in the Comment's benchmark.

This repository provides the corrected full-test scores, the formula-filtered diagnostic behind the inflated subset result, retrieval metrics, nearest-neighbour upper bounds, and the MIST configuration files used for reproduction.

## Main Results

Values are mean Jaccard similarity between predicted Morgan4096 fingerprints and ground-truth fingerprints. The first table reproduces the comparison as presented in the Comment. Daggered nearest-neighbour rows are formula-filtered subset results: **test entries without a same-formula training candidate are skipped**. Each split score includes the number of test entries contributing to that score. The differing `n` values are the technical issue: these rows are not full-test baselines and are not directly comparable to MIST. The skip is introduced in the nearest-neighbour implementation by [this line of code](https://github.com/serenaklm/ML_MS_analysis/blob/1f88b9eab2656eb75c5915bf6ae4575848b44fe7/benchmarked_models/nearest_neighbour/02_compute_nn.py#L128), but MIST is scored on every test entry.

| Dataset     | Model              | Scaffold Jaccard (↑) | Random Jaccard (↑) |
| ----------- | ------------------ | -------------------: | -----------------: |
| NPLIB1      | MIST in Comment    |      0.238 (n=2,689) |    0.539 (n=2,744) |
| NPLIB1      | Nearest neighbour† |      0.293 (n=1,028) |    0.822 (n=2,212) |
| NPLIB1      | DreaMS NN†        |      0.293 (n=1,028) |    0.830 (n=2,212) |
| MassSpecGym | MIST in Comment    |     0.270 (n=16,042) |   0.766 (n=16,250) |
| MassSpecGym | Nearest neighbour† |      0.455 (n=6,571) |   0.953 (n=15,777) |
| MassSpecGym | DreaMS NN†         |      0.470 (n=6,571) |   0.966 (n=15,777) |

The corrected reproduction below keeps every method on the full test set. The
nearest-neighbour rows use the repository default `same_formula_candidates_fallback`
policy: candidates are still restricted to training spectra sharing the query's
molecular formula, but when no such spectrum exists the search falls back to the
whole training set instead of dropping the query. This preserves the formula
prior that makes nearest neighbour strong while putting every method on one
denominator. NIST2023 is not included in the reproduction. The authors declined
our request to access the exact NIST'23-derived artifacts used in the Comment.

All MIST rows are trained on the training split only. The validation split is 
used solely to select the checkpoint and to fit the binarization threshold, both 
before the test split is touched. Every row is scored on the same full test set:
n=2,744 random and n=2,689 scaffold for NPLIB1, n=16,250 and n=16,042 for MassSpecGym. 
It is worth noting that `MIST in Comment` already outperforms the machine-learning-free
nearest neighbour on scaffold splits after fixing the testing size.

| Dataset     | Model                          | Scaffold Jaccard (↑) |   Random Jaccard (↑) |
| ----------- | ------------------------------ | -------------------: | -------------------: |
| NPLIB1      | MIST in Comment, threshold 0.5  |      0.238 (n=2,689) |      0.539 (n=2,744) |
| NPLIB1      | MIST in Comment, fitted threshold |    0.298 (n=2,689) |      0.638 (n=2,744) |
| NPLIB1      | Corrected MIST                         |  **0.317** (n=2,689) |      0.712 (n=2,744) |
| NPLIB1      | Nearest neighbour              |      0.212 (n=2,689) |      0.720 (n=2,744) |
| NPLIB1      | DreaMS NN                      |      0.258 (n=2,689) |  **0.744** (n=2,744) |
| _NPLIB1_    | _test structures already in training_ |        _1.6%_ |              _62.9%_ |
| MassSpecGym | MIST in Comment, threshold 0.5  |     0.270 (n=16,042) |     0.766 (n=16,250) |
| MassSpecGym | MIST in Comment, fitted threshold |   0.318 (n=16,042) |     0.805 (n=16,250) |
| MassSpecGym | Corrected MIST                         | **0.368** (n=16,042) |     0.813 (n=16,250) |
| MassSpecGym | Nearest neighbour              |     0.256 (n=16,042) |     0.929 (n=16,250) |
| MassSpecGym | DreaMS NN                      |     0.306 (n=16,042) | **0.944** (n=16,250) |
| _MassSpecGym_ | _test structures already in training_ |     _14.3%_ |              _95.3%_ |

Two corrections separate the `MIST in Comment` row from `Corrected MIST`, and they are
reported separately above. Fitting the decision threshold on the validation split
is worth +0.060 and +0.048 Jaccard on the two scaffold splits with **no change to
the trained model**; switching from the BCE objective to MIST's default cosine
objective adds a further +0.019 and +0.051. 

**The corrected picture is split-dependent.** MIST is ahead
of both nearest-neighbour baselines on the two scaffold splits, by 0.059 on NPLIB1
and 0.062 on MassSpecGym. Nearest neighbour remains ahead on the two random
splits, by 0.032 and 0.132. Those are precisely the splits in which 62.9% and
95.3% of test structures are already present in the training set, where returning
a memorized training fingerprint is close to a lookup rather than a prediction.
DreaMS NN, which is still based on a machine learning method, outperforms vanilla 
nearest-neighbour on all splits, contradicting the general claim that "Machine
learning does not work".

Additional fingerprint-cosine, candidate-retrieval, and formula-filtered nearest-neighbour diagnostic tables are in [DETAILED_RESULTS.md](DETAILED_RESULTS.md).

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

**After correcting the evaluation, fitting the decision threshold, and restoring
MIST's default training objective, the empirical picture changes substantially.**
On the two scaffold splits, which require generalization to unseen scaffolds,
MIST outperforms both corrected nearest-neighbour baselines. On the two random
splits nearest neighbour remains ahead, but those splits carry 62.9% and 95.3%
exact structural overlap between test and training data, and the
nearest-neighbour upper bound there reaches 0.972 Jaccard; that regime measures
retrieval of memorized structures more than prediction. Even on that split, DreaMS
NN, a method that utilized machine-learning embedding, outperforms vanilla NN -- a result that
does not support the claim that machine learning fails. Under candidate
retrieval, the endpoint that structure annotation actually uses, MIST is ahead on
all four settings. These results are difficult to reconcile with the broader claim 
that machine learning fails for small-molecule mass spectrometry.

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
