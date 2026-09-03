# Why Does Machine Learning Fail for Small-Molecule Mass Spectrometry?

This repository contains the code, analysis, and evaluation pipelines used for benchmarking machine learning models for small-molecule mass spectrometry tasks.

---

# Data

All datasets used in this project will be made available via Google Drive.

You can access the data here:

https://drive.google.com/drive/folders/1v11lTwFSdlSRJ6ETLHqkbT809Ji9w0OY?usp=drive_link

The folder contains the processed datasets and intermediate files required to reproduce the experiments reported in the paper.

---

> **This section was not produced by the original authors.** It reports an independent
> reproduction of the benchmark behind the Nature Metabolism Comment
> [*Why does machine learning fail for small-molecule mass spectrometry?*](https://www.nature.com/articles/s42255-026-01544-6),
> starting from commit `da67d9d0` of this repository, and documents three evaluation
> problems that together flip the conclusion on both scaffold splits.
>
> Scope: NPLIB1 and MassSpecGym fingerprint prediction. NIST'23, the retrieval task, and
> the non-MIST baselines were not re-run. Every number below is reproducible from the code
> in our fork, [coleygroup/Audit_ML_MS_analysis](https://github.com/coleygroup/Audit_ML_MS_analysis).
>
> We welcome correction if any of this is wrong, and we encourage the authors to reproduce
> these fixes and consider updating or retracting the Comment.

# ~~Why~~ Does Machine Learning REALLY Fail for Small-Molecule Mass Spectrometry?

Fingerprint-prediction Jaccard on NPLIB1 and MassSpecGym. Every row below the rule is
scored on the **full** test split, so those methods are directly comparable.

| Method | NPLIB1 scaffold | NPLIB1 random | MSG scaffold | MSG random |
|---|---:|---:|---:|---:|
| _NN, formula-first — [reported in Nature Metabolism](https://www.nature.com/articles/s42255-026-01544-6), scored on the reduced subset_ | _0.2929_ | _0.8218_ | _0.4553_ | _0.9528_ |
| _spectra actually scored in that row_ | _n=1,028_ | _n=2,212_ | _n=6,571_ | _n=15,777_ |
| — | — | — | — | — |
| NN, binned spectrum, formula-first | 0.1945 | 0.6942 | 0.2564 | 0.9285 |
| NN, DreaMS, formula-first | 0.2579 | **0.7433** | 0.3063 | **0.9443** |
| MIST, BCE @ fixed 0.5 threshold | 0.2380 | 0.5389 | 0.2703 | 0.7650 |
| MIST, BCE @ validation-tuned threshold | 0.2979 | 0.6382 | 0.3177 | 0.8046 |
| MIST, **cosine** @ validation-tuned threshold | **0.3170** | 0.7116 | **0.3684** | 0.8125 |
| _test spectra_ | _n=2,689_ | _n=2,744_ | _n=16,042_ | _n=16,250_ |

**The conclusion flips on both scaffold splits** — the splits meant to test generalisation.
On MassSpecGym scaffold, MIST goes from losing by 0.185 (0.2703 vs 0.4553) to winning by
0.062 (0.3684 vs 0.3063). Two problems produced that gap, and a third explains why the
random splits look so different.

## 1. The test split is wrong

The nearest-neighbour baselines restrict candidates to training spectra sharing the
query's molecular formula, and when no such spectrum exists the test entry is silently
dropped from the evaluation by [this line of code](https://github.com/serenaklm/ML_MS_analysis/blob/1f88b9eab2656eb75c5915bf6ae4575848b44fe7/benchmarked_models/nearest_neighbour/02_compute_nn.py#L128):

```python
# Let us sieve out the train
sieved_idx = [idx for idx, f in enumerate(train_formula) if f == test_formula]
if len(sieved_idx) == 0: continue
```

The `continue` skips the query entirely rather than falling back to a wider candidate set,
so it never reaches `computed_test_ids` and never enters the mean. The DreaMS variant in
`01b_compute_nn_dreaMS.py` carries the same line. MIST, meanwhile, is scored on every test
spectrum, so the two were never measured on the same set.

| | NPLIB1 scaffold | NPLIB1 random | MSG scaffold | MSG random |
|---|---:|---:|---:|---:|
| test spectra in the split | 2,689 | 2,744 | 16,042 | 16,250 |
| test spectra the NN code actually scored | 1,028 | 2,212 | 6,571 | 15,777 |
| dropped | **1,661 (62%)** | 532 (19%) | **9,471 (59%)** | 473 (3%) |
| NN Jaccard on the dropped entries | 0.1336 | 0.1639 | 0.1184 | 0.1176 |

On both scaffold splits roughly 60% of the test set was discarded, and the discarded
entries score 0.12–0.16 against 0.29–0.46 on the entries that were kept — they are
precisely the hard cases: molecules whose formula was never seen in training. Removing
them inflates the baseline by a wide margin.

[This commit](https://github.com/serenaklm/ML_MS_analysis/commit/0ad9da7a3546f047c547ad38347fc3dc22a042f3)
by the original authors does not fix the issue. It adds a notebook that scores MIST on the
same reduced subset the nearest-neighbour code produced, which equalises the two methods
but leaves roughly 60% of each scaffold test split outside the benchmark entirely.

The real fix is to fall back to the whole training set instead of skipping. MassSpecGym
scaffold drops from 0.4553 to 0.2564; NPLIB1 scaffold from 0.2929 to 0.1945.

Note that the NPLIB1 numbers here cover 2,689 scaffold and 2,744 random test spectra: the
MIST data export omits 27 spectra that the MGF export contains, so those entries must be
recovered (their `.ms` files and subformulae regenerated) before MIST can be scored on the
same test set as the nearest-neighbour baselines.

## 2. Tuning MIST

The MIST pipeline leaves two choices unexamined.

**The decision threshold was hard-coded.** The predicted fingerprint must be binarised
before Jaccard can be computed, and the cut point was fixed at `0.5` rather than fitted.
It is now chosen on the **validation** split (grid 0.02–0.98) before test is touched.

**The loss was not MIST's default.** The shipped config trains with `loss_fn: bce`;
MIST's own default is cosine.

| | NPLIB1 scaffold | NPLIB1 random | MSG scaffold | MSG random |
|---|---:|---:|---:|---:|
| BCE @ 0.5 | 0.2380 | 0.5389 | 0.2703 | 0.7650 |
| BCE @ validation-tuned | 0.2979 | 0.6382 | 0.3177 | 0.8046 |
| cosine @ validation-tuned | **0.3170** | **0.7116** | **0.3684** | **0.8125** |
| selected threshold (BCE / cosine) | 0.80 / 0.16 | 0.82 / 0.14 | 0.78 / 0.12 | 0.82 / 0.24 |

Fixing both improves MIST accuracy by a wide margin, outperforming every nearest-neighbour
variant on the scaffold splits.

## 3. Train-test leakage

On the random splits nearest-neighbour retrieval still wins, but those splits barely test
generalisation. Counting the test spectra whose 2D InChIKey skeleton already appears in the
training set:

| | NPLIB1 scaffold | NPLIB1 random | MSG scaffold | MSG random |
|---|---:|---:|---:|---:|
| test structures also seen in training | 1.6% | **62.9%** | 14.3% | **95.3%** |

On MassSpecGym random, 95.3% of test structures are already in the training set, so
retrieving the nearest training spectrum is close to a lookup — which is what the 0.94
Jaccard reflects, and why no fingerprint predictor beats it there. The scaffold splits cut
leakage to 1.6% and 14.3%, and that is exactly where the ranking changes.

Conclusions drawn from the random splits therefore say more about the split than about the
methods.

<!-- # Quick Start

## 1. Clone repository
```bash
git clone https://github.com/ML_MS_analysis.git
cd your-repo-name

## 2. Installation
conda create -n massspec_benchmark python=3.10
conda activate massspec_benchmark
pip install -r requirements.txt

# Data Availability
Due to licensing restrictions, we are **unable to redistribute the NIST 2023 Mass Spectral Library**.

# What we provide

We include all materials that can be legally shared:

- Preprocessing and data construction scripts  
- Train/test split definitions  
- Evaluation pipelines and metrics  
- Metadata and derived files that do not contain restricted spectra  
- Scripts to reproduce all results and figures  

These resources allow full reproduction of our experiments for open-sourced datasets.

# Benchmarked_models

This folder contains implementations and prediction outputs for all benchmarked models.

Subfolders:

    - mist/
    Implementation and outputs for the MIST model.

    - nearestneighbour/
    Nearest-neighbour baselines (MS similarity, embedding-based retrieval, etc.).

    - our_models/
    Implementations and outputs for other models proposed in this work.

    Each subfolder contains:
        - Training scripts
        - Inference scripts
        - Evaluation utilities

        saved predictions (where applicable)
            These subfolders contain all fo teh different odels. 
            Due to the file size, we will upload the checkpoints to zenodo and provide instructions on how to download the checkpoints. 

# Installation Instructions

# Folders

## data 

## DreaMS

10.5281/zenodo.19007173

## FP_prediction  -->