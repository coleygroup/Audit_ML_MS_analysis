# Why Does Machine Learning Fail for Small-Molecule Mass Spectrometry?

Code, analysis and evaluation pipelines used to benchmark machine-learning models that predict molecular
fingerprints from small-molecule MS/MS spectra, and to diagnose *why* they fail: retrieval baselines that
expose how much of the task is memorisation, adversarial **learning-to-split** partitions that expose the
hardest generalisation gap, and analyses of out-of-vocabulary chemical formulas and experimental conditions.

If you want to run the same analysis on **your own model**, jump to [Evaluating your own model](#evaluating-your-own-model).

---

## Repository layout

```
ML_MS_analysis/
├── benchmarked_models/
│   ├── nearest_neighbour/     # Retrieval baselines: same-formula NN, DreaMS NN, formula-aware fallbacks, MIST comparison
│   ├── learning_to_split/     # `learning_to_split` package: losses + sampler for adversarial train/test splitting
│   ├── other_baselines/       # Binned-MLP, MS-transformer and formula-transformer fingerprint predictors (+ LS drivers)
│   └── mist/                  # Re-implementation of MIST used for error analysis (+ LS driver)
├── diagnosis/                 # Notebooks that turn cached predictions into the tables and figures of the paper
├── figures/                   # Figures produced by the diagnosis notebooks
├── data/                      # (not tracked) processed datasets, splits and MGF caches -- see Data
├── results/                   # (not tracked) all cached predictions and metrics -- created by the scripts
└── download.sh                # Helper to fetch the original CANOPUS export
```

Every folder under `benchmarked_models/` has its own README with the exact commands.

## Data

All processed datasets, split definitions and intermediate files are available on Google Drive:

https://drive.google.com/drive/folders/1v11lTwFSdlSRJ6ETLHqkbT809Ji9w0OY?usp=drive_link

Download them into `data/` so that the layout is

```
data/
├── processed_data/
│   ├── NPLIB1.pkl                 # list of records (see below)
│   └── massspecgym.pkl
├── splits/{NPLIB1,massspecgym}/{scaffold,random,LS}.json   # {"train": [...], "val": [...], "test": [...]} record ids
└── MGF_files/{dataset}/{split}/{train,test}.mgf              # same spectra, same order, for DreaMS
```

Each record in `processed_data/*.pkl` is a dict with (at least) `id_`, `smiles`, `formula`, `formula_corrected`
(precursor **ion** formula), `precursor_type` (adduct), `precursor_MZ_final`, `instrument_type`,
`collision_energy`, `peaks` (list of `{"mz", "intensity", "intensity_norm", "comment": {"f_pred": <sub-formula>}}`)
and `FPs` (`MACCS`, `morgan4_{256,1024,2048,4096}`, `morgan6_{...}` as bit strings).

**Fingerprint definition.** `morgan4_4096` is an RDKit Morgan fingerprint of radius 2 (diameter 4) with 4096
bits computed on the *kekulized* molecule with aromatic flags cleared (`MS_processing/dataprocessing/utils/chem_utils.py`).
It is **not** bit-identical to the plain RDKit `GetMorganGenerator(radius=2, fpSize=4096)` fingerprint that MIST
is trained on; use the same definition when comparing numbers across models.

The NIST 2023 library cannot be redistributed; the scripts that mention `nist2023` require your own licensed copy.

## Environment

The retrieval and analysis code needs only `numpy scipy scikit-learn tqdm rdkit matplotlib pandas`
(plus `dreams` for DreaMS embeddings and `torch transformers matchms` for the retrieval-task script).
Model training additionally needs `torch pytorch_lightning wandb pyyaml` and, for MIST, the original
[`mist`](https://github.com/samgoldman97/mist/tree/main_v2) package. Install the learning-to-split package with

```bash
pip install -e benchmarked_models/learning_to_split
```

Weights & Biases logging is used by the training scripts: run `wandb login` or export `WANDB_API_KEY` first.

## Quick start

```bash
# 1. Nearest-neighbour baselines (CPU only, ~2 h for everything from scratch)
cd benchmarked_models/nearest_neighbour
python run_nn.py --all           # all 16 settings (5 spectral similarities x formula filter / sub-formula rule), vs MIST
python run_nn.py --summarize     # tables -> results/nearest_neighbour/summary_nn_mist.md (+ per-mode summaries)

# 2. Fingerprint predictors (GPU)
cd ../other_baselines && python train.py --config_file w_meta_config.yaml
cd ../mist            && python train.py --config_file NPLIB1_scaffold_original.yaml

# 3. Learning to split (GPU)
cd ../other_baselines && python train_LS.py --config_file LS_config.yaml
cd ../mist            && python train_LS.py --config_file LS_config.yaml

# 4. Tables and figures
jupyter lab ../../diagnosis
```

## Evaluating your own model

The analyses only need your model's per-spectrum predictions on the provided splits, in the same format as
the models benchmarked here.

1. **Train on the provided splits.** Use `data/splits/{dataset}/{split}.json` (`train`/`val`/`test` ids). Do not
   touch `test` during model selection. The `LS.json` split is the learned adversarial split.
2. **Export predictions** to `results/FP_prediction/<your_model>/{dataset}/<run_name>_{split}/`:
   * `test_results.pkl`: `dict` mapping test id (string) to `{"pred": [4096 floats], "GT": [4096 floats], "jaccard": float}`,
   * `test_performance.json`: `{"jaccard": <mean over the test split>}`.
   `jaccard` is the Tanimoto similarity between `pred > 0.5` and `GT`. The run name must end with `_{split}`
   (`_scaffold`, `_random`, `_LS`). Use the **same fingerprint definition** for `GT` as the baselines you compare
   against; the baselines are scored with RDKit Morgan radius 2 / 4096 bits from SMILES
   (`nearest_neighbour/utils/nn_lib.mist_fps`), or with the dataset bits via `run_nn.py --fingerprint dataset`.
3. **Compare with the retrieval baselines.** `diagnosis/01b_get_FP_prediction_results.ipynb` lists every model
   folder found under `results/FP_prediction/`, including the nearest-neighbour baselines exported by `run_nn.py`.
   `results/nearest_neighbour/summary_<mode>_mist.md` reports the baselines (Jaccard and cosine) on the full test
   split, on the *formula-matched* subset and on the *no-formula-match* subset; report your model on the same subsets
   (the formula-matched ids are those with `matched` in
   `results/nearest_neighbour/{dataset}_{split}/with_formula_filter_binned.pkl`). `run_nn.py` also runs the baselines
   on your own dataset (`--dataset <name> --data_file ... --split_dir ...`, see its README).
4. **Learn the hardest split for your model.** Implement a splitter head for your architecture and reuse
   `learning_to_split` (losses + sampler) with the outer loop in `benchmarked_models/other_baselines/train_LS.py`.
   `benchmarked_models/learning_to_split/README.md` describes the interface in detail; the output `best_split.pkl`
   can be turned into a `splits/{dataset}/LS.json` and analysed with `diagnosis/02a_analyze_LS_split.ipynb`.

## Results

Full-test-set mean Tanimoto (MIST fingerprint definition; sub-formula fallback hyperparameters `(w, alpha, T, k)`
selected on the validation split):

| dataset / split | NN, best-match fallback | NN, sub-formula fallback | DreaMS, best-match fallback | DreaMS, sub-formula fallback | MIST |
|---|---|---|---|---|---|
| NPLIB1 / scaffold | 0.197 | 0.312 | 0.259 | 0.311 | 0.241 |
| NPLIB1 / random | 0.696 | 0.752 | 0.745 | 0.758 | 0.547 |
| MassSpecGym / scaffold | 0.258 | 0.365 | 0.308 | 0.371 | 0.267 |
| MassSpecGym / random | 0.929 | 0.935 | 0.945 | 0.947 | 0.674 |

The table is generated by `python run_nn.py --summarize` (`results/nearest_neighbour/summary_nn_mist.md`, which
also covers the other nearest-neighbour variants and the cosine similarity); the columns above are the settings
`binned_with_formula_filter`, `binned_subformula_match`, `dreams_with_formula_filter` and `dreams_subformula_match`.

## License

See `LICENSE`.
