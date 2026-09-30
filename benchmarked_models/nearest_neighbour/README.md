# Nearest-neighbour baselines

Retrieval baselines for fingerprint prediction from MS/MS spectra. For every test spectrum, training spectra are
retrieved and the Morgan fingerprint of the retrieved compound(s) is the prediction. Nothing is learned, so the
baselines measure how much of a benchmark can be solved by memorisation. Retrieval is always against the training
split; the validation split is used only to select the hyperparameters of the sub-formula rule; the test split is never
used for any choice.

`run_nn.py` is the single entry point: pick the dataset(s), split(s) and setting(s), or `--setting all`.

## Folder

```
nearest_neighbour/
├── run_nn.py                    # entry point: every setting, own datasets, audit, summary tables
└── utils/
    ├── nn_driver.py             # retrieval + evaluation of one setting on one dataset / split, summary tables
    ├── peak_similarity.py       # numba peak-matching kernel (cosine / modified cosine / neutral loss), matchms check
    ├── subformula_fallback.py   # sub-formula rule: validation selection of (w, alpha, T, k), top-k consensus
    ├── nn_lib.py                # data, fingerprints, binning, sub-formula tokens, formula similarity, metrics
    ├── audit_metrics.py         # split properties: coverage, leakage, fingerprint oracle (run_nn.py --audit)
    ├── cache_dreams_embeddings.py  # DreaMS embeddings of the train / test MGF files (conda env `dreams`)
    └── utils.py                 # load_pickle / pickle_data (used by cache_dreams_embeddings.py)
```

## Settings

A test spectrum is **formula-matched** when at least one training spectrum has the same neutral formula (`formula`).

| setting | formula | formula-matched test spectra | other test spectra |
|---|---|---|---|
| `<sim>_no_formula_filter` | unknown | best `<sim>` match over the whole training split | same |
| `<sim>_with_formula_filter` | known | best `<sim>` match among the training spectra with that formula | best `<sim>` match over the whole training split |
| `<sim>_subformula_match` | known | best `<sim>` match among the training spectra with that formula | sub-formula rule |
| `subformula_only` | known | sub-formula rule | sub-formula rule |

`<sim>` is one of:

| `<sim>` | similarity |
|---|---|
| `cosine` | matchms `CosineGreedy`: peaks match when \|mz_train - mz_test\| <= tolerance (open search, no precursor filter) |
| `modified_cosine` | matchms `ModifiedCosine`: as cosine, or after shifting by the precursor m/z difference |
| `neutral_loss` | matchms `NeutralLossesCosine`: neutral losses (precursor - mz) agree; peaks >= precursor dropped |
| `dreams` | cosine of DreaMS embeddings (self-supervised checkpoint, `utils/cache_dreams_embeddings.py`) |
| `binned` | cosine of 0.25 Da binned spectra (`intensity_norm` summed per bin up to 2000 Da, precursor added as a peak of intensity 100) |

`--setting all` runs all 16; a mode name (`no_formula_filter`, `with_formula_filter`, `subformula_match`) runs its five
similarities.

**Peak matching (cosine, modified_cosine, neutral_loss).** Matched pairs are assigned greedily by intensity product
and normalised by the two spectrum norms, exactly as in matchms 0.28. The scoring is a numba re-implementation
(`utils/peak_similarity.py`) so that all ~1.2e9 test x train pairs of MassSpecGym run in under a minute on 80 cores;
before every full search `--n_check` (default 300) random pairs plus 300 closest-precursor pairs are scored with
matchms itself and the run aborts on any difference. Defaults are the matchms defaults: tolerance 0.1 Da, `mz_power` 0,
`intensity_power` 1, on the stored `intensity_norm` peaks. `--max_peaks 500` keeps the 500 most intense peaks per spectrum; this
only affects NPLIB1 (~15% of spectra, some profile-like with up to 361k peaks); `--max_peaks 0` keeps all.

**Sub-formula rule.** Every peak (100 most intense) is annotated with a sub-formula of the precursor ion formula
(`formula_corrected`) by mass decomposition (20 ppm / 0.01 Da, loose valence check); a spectrum becomes a bag of
fragment (`F:C7H7`) and neutral-loss (`L:C9H11NO3`) tokens weighted by sqrt(intensity), L2-normalised. Training spectra
are scored by

    score(q, t) = s_SF(q, t) + alpha * s_F(q, t; w)

with `s_SF` the cosine between token vectors and `s_F` the heteroatom-weighted Bray-Curtis similarity of the
element-count vectors (a heteroatom mismatch counts `w` times a C/H mismatch; `s_F = 1` for the same formula). The
prediction is the elementwise threshold at 0.5 of the softmax(T)-weighted average of the fingerprints of the `k`
best-scoring training compounds (`k = 1` copies the best match). `(w, alpha, T, k)` are selected per dataset / split
over `w in {1,2,4,8,16}`, `alpha in {0,0.25,0.5,1,2,4}`, `T in {0.01,...,0.5}`, `k in {1,...,50}`: on the validation
spectra without a same-formula training spectrum for `*_subformula_match` (where the rule is the fallback, so it is
shared by all five similarities), and on all validation spectra for `subformula_only`.

**Ties** are broken by training order. Test spectra whose same-formula pool shares no peak with them score 0 against
every candidate and get the first training spectrum of their formula (counted as `best score = 0` in the tables).

## Metrics

* **Jaccard** (Tanimoto) of the binary prediction -- the retrieved fingerprint, or the sub-formula consensus
  thresholded at 0.5 -- against the true fingerprint.
* **Cosine similarity** of the raw prediction, without threshold -- the retrieved fingerprint (binary), the
  sub-formula consensus average (continuous) -- against the true fingerprint. MIST's cosine uses its predicted
  probabilities, its Jaccard the probabilities thresholded at 0.5 (as stored by the MIST evaluation).

Both are reported on the full test split and on the formula-matched / unmatched subsets, and next to MIST on the test
spectra MIST has predictions for.

**Fingerprint definition.** `--fingerprint mist` (default) uses RDKit Morgan radius 2, 4096 bits computed from SMILES;
`--fingerprint dataset` uses the stored `morgan4_4096` bits, computed on the kekulised molecule.

## Run

```bash
conda activate mist                                   # numpy scipy numba matchms torch tqdm rdkit
cd benchmarked_models/nearest_neighbour
python utils/cache_dreams_embeddings.py               # once, in conda env `dreams` (only for the dreams settings)
python run_nn.py --all                                # built-in datasets x scaffold / random, every setting
python run_nn.py --all --audit                        # + coverage / leakage / fingerprint oracle
python run_nn.py --summarize                          # only rewrite the tables from existing results
```

Single runs:

```bash
python run_nn.py --dataset NPLIB1 --split scaffold                                   # every setting
python run_nn.py --dataset massspecgym --split random --setting dreams_subformula_match cosine_no_formula_filter
python run_nn.py --dataset NPLIB1 massspecgym --split scaffold --setting with_formula_filter
```

Retrievals are cached per dataset / split and reused when their parameters and ids match: `*_with_formula_filter`
reuses `*_no_formula_filter` for its unmatched spectra, `*_subformula_match` reuses `*_with_formula_filter` and the
sub-formula selection, so running the settings in the order above (as `all` does) never searches twice. A full `--all`
run from scratch takes ~2 h on 80 cores, most of it the `subformula_only` validation grid on MassSpecGym
(single-threaded).

**Own dataset.** Either store it in the repository layout (`data/processed_data/<name>.pkl`,
`data/splits/<name>/<split>.json`) and pass `--dataset <name>`, or point to it:

```bash
python run_nn.py --dataset mydata --data_file /path/records.pkl --split_dir /path/splits --split fold0 fold1
```

Records are a pickled list of dicts with `id_`, `peaks` (`[{"mz", "intensity_norm"}]`), `precursor_MZ_final`,
`formula` (neutral; formula filter), `formula_corrected` (precursor ion formula; sub-formula rule), `smiles`
(`--fingerprint mist`) or `FPs["morgan4_4096"]` (`--fingerprint dataset`). A split is a json
`{"train": [ids], "val": [ids], "test": [ids]}`. The `dreams` settings additionally need
`data/MGF_files/<name>/<split>/{train,test}.mgf` and their embeddings (`utils/cache_dreams_embeddings.py --datasets
<name> --splits <split>`); with `all` they are skipped when missing. MIST rows appear when
`results/FP_prediction/mist/<name>/<name>_MIST_4096_<split>/test_results.pkl` exists.

**Own model.** Export predictions as described in the top-level README (`test_results.pkl` with `pred`, `GT`,
`jaccard` per test id) and report them on the same subsets as the baselines: the formula-matched test ids are those
with `matched` in `with_formula_filter_<sim>.pkl`. Use the same fingerprint definition as the baseline you compare with.

## Outputs

In `results/nearest_neighbour/<dataset>_<split>/`, with `<mode>` in `no_formula_filter`, `with_formula_filter`,
`subformula_match`, `subformula_only`:

| file | content |
|---|---|
| `no_formula_filter_<sim>.pkl`, `with_formula_filter_<sim>.pkl` | retrieval: top-k training ids / scores per test spectrum, parameters (`matched` for the latter) |
| `subformula_selection_<fp>.json`, `subformula_fallback_<fp>.pkl` | sub-formula rule as fallback: validation grid + selection, top-k ids + softmax weights |
| `subformula_all_selection_<fp>.json`, `subformula_all_<fp>.pkl` | the same for `subformula_only` |
| `<mode>_jaccard_<fp>.pkl`, `<mode>_cosine_<fp>.pkl` | per-test-id metrics, per `<sim>` |
| `<mode>_metrics_<fp>.json` | every number in the tables, per `<sim>` |
| `audit_metrics_<fp>.json` | split properties (`--audit`) |

Tables in `results/nearest_neighbour/`: `summary_nn_<fp>.md` (every setting, Jaccard / cosine, full test split),
`summary_<mode>_<fp>.md` (per mode, with subsets and MIST-common numbers) and `audit_metrics_<fp>.md`. With
`--fingerprint mist` the predictions are also written in the layout of the trained models,
`results/FP_prediction/nearest_neighbour/<dataset>/<TAG>_NN-<sim>_<mode>_4096_<split>/`, so
`diagnosis/01b_get_FP_prediction_results.ipynb` lists them next to the trained models.

## Notes for interpretation

* numpy in the `mist` env is linked against reference BLAS; all large matrix products here run in torch on CPU.
