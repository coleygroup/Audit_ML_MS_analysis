# MIST (re-implementation for error analysis)

A re-implementation of MIST (Goldman et al., 2023) used for the error analyses in this repository. Please refer to
the original repository, https://github.com/samgoldman97/mist/tree/main_v2, for the reference implementation; the
`mist` package from that repository must be installed, because the featurisers and datasets are imported from it.

```
mist/
├── all_configs/                 # YAML configs (NPLIB1_*, MSG_* for the paper runs; LS_config.yaml; FT_config_*.yaml)
├── model/mist_model.py          # MistNet (predictor) and MistNetSplitter (learning-to-split head)
├── modules/                     # data module, form embedders, transformer layers
├── train.py                     # train from a config
├── predict.py                   # predict from a run folder -> <split>_results.pkl / <split>_performance.json
├── train_LS.py                  # learning-to-split outer loop (see ../learning_to_split/README.md)
├── finetune.py                  # fine-tune a checkpoint
├── train_w_sampling.py          # training with re-sampling
└── get_EKFAC_influence_mist.py  # EK-FAC influence scores
```

## Usage

```bash
wandb login                                                         # or export WANDB_API_KEY
python train.py --config_file NPLIB1_scaffold_original.yaml --results_dir ./results
python predict.py --checkpoint results/<run>                         # run FOLDER, not the .ckpt: run.yaml is read from it
python predict.py --checkpoint results/<run> --split val            # -> val_results.pkl, val_performance.json
python ../tune_threshold.py --runs results/<run>                    # -> test_performance_val_threshold.json
python train_LS.py --config_file LS_config.yaml --results_dir ./ls_results
```

`predict.py` stores the raw per-bit probabilities and binarises at 0.5 for the `jaccard` it records. 

`dataset.data_folder` in the configs points to a MIST-format data folder (`labels.tsv`, `spec_files/*.ms`,
`subformulae/default_subformulae/*.json`, `magma_outputs/`, `splits/*.tsv` with `name` / `split` columns); the
shipped paths are absolute and need editing. `dataset.fp_names: [morgan4096]` is the plain RDKit Morgan
radius-2, 4096-bit fingerprint of the original MIST code.