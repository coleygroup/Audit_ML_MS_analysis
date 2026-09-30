# Fingerprint predictors (binned MLP, MS transformer, formula transformer)

Three fingerprint-prediction models used as trained baselines, plus the scripts to fine-tune them, train them
with sampling, run ablations, compute EK-FAC influence scores and learn adversarial splits.

```
other_baselines/
├── all_configs/            # YAML configs: w_meta_config.yaml, wo_meta_config.yaml, LS_config.yaml, FT_config_*.yaml, sampling_config_*.yaml, ablations_*.yaml
├── dataloader/             # MSDataset (Lightning datamodule reading the split json) and Data (plain Dataset)
├── modules/                # MSBinnedModel, MSTransformerEncoder, FormulaTransformerEncoder (+ their *Splitter variants)
├── utils/                  # config / data / training helpers
├── train.py                # train one model from a config
├── predict.py              # predict with a checkpoint, writes <split>_results.pkl / <split>_performance.json
├── train_LS.py             # learning-to-split outer loop (see ../learning_to_split/README.md)
├── train_finetune.py       # fine-tune a checkpoint on another dataset
├── train_w_sampling.py     # training with re-sampling of the training set
├── train_ablations.py      # feature ablations (adduct / instrument / collision energy)
├── get_EKFAC_influence.py  # training-data influence scores (EK-FAC)
└── optimize_threshold.ipynb  # decision threshold selected on validation (see ../tune_threshold.py)
```

## Models

| `model.name` in the config | input |
|---|---|
| `binned_MS_encoder` | 0.25 Da binned spectrum (MLP) |
| `MS_encoder` | set of peaks (transformer encoder) |
| `formula_encoder` | set of peaks with sub-formula annotations (transformer encoder) |

`model.feats_params` switches the metadata embeddings (`include_adduct`, `include_CE`, `include_instrument`) on
or off; `w_meta_config.yaml` / `wo_meta_config.yaml` are the two settings used in the paper. All models predict
`data.FP_type` (default `morgan4_4096`, see the fingerprint note in the top-level README).

## Usage

```bash
wandb login                                                         # or export WANDB_API_KEY
python train.py --config_file w_meta_config.yaml --results_dir ./results
python predict.py --checkpoint results/<run>                          # -> test_results.pkl, test_performance.json in the run folder
python predict.py --checkpoint results/<run> --split val             # -> val_results.pkl, val_performance.json
python ../tune_threshold.py --runs results/<run>                     # -> test_performance_val_threshold.json
python train_LS.py --config_file LS_config.yaml --results_dir ./ls_results
```

`predict.py` binarises at a fixed 0.5. `../tune_threshold.py` sweeps the threshold on the **validation**
predictions and applies the argmax to test; `optimize_threshold.ipynb` is a notebook front end for the same routine.

`data.data_folder`, `data.splits_folder` and `data.split_file` in the configs must point to your copy of the
processed data (one pickle per spectrum under `<data_folder>/<dataset>/frags_preds/`, `all_adducts.pkl`,
`all_instruments.pkl`, and a split json with `train` / `val` / `test` file names); the shipped paths are absolute
and need editing. Prediction outputs follow the format described in the top-level README, so they are picked up by
`diagnosis/01b_get_FP_prediction_results.ipynb` when copied to `results/FP_prediction/<model>/<dataset>/<run>_<split>/`.
