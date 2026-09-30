# Learning to split (adversarial train/test partitioning)

An adaptation of *Learning to Split for Automatic Bias Detection* (Bao & Barzilay, 2022) to fingerprint
prediction from MS/MS spectra. Instead of a fixed scaffold or random split, a **splitter** network learns which
spectra to send to the test set so that a **predictor** trained on the remaining spectra generalises as badly as
possible. The resulting split (`data/splits/{dataset}/LS.json`) exposes the failure modes of a model family,
and `diagnosis/02a_analyze_LS_split.ipynb` characterises what the splitter picked (instruments, adducts,
collision energies, formulas).

## What is in this folder

```
learning_to_split/
├── setup.py                                  # pip install -e .
└── learning_to_split/
    ├── training/split_data.py                # split_data(): sample a train/test assignment from the splitter
    ├── training/splitter.py                  # losses: compute_gap_loss, compute_marginal_z_loss, compute_y_given_z_loss
    └── utils/data_utils.py                   # set_seed, load_pickle, pickle_data, read_config, write_json
```

The package is deliberately model-agnostic: it contains only the sampler and the losses. The outer loop and the
model-specific splitter heads live next to each model family:

* `../other_baselines/train_LS.py` with `MSBinnedModelSplitter`, `MSTransformerEncoderSplitter`,
  `FormulaTransformerEncoderSplitter` (`../other_baselines/modules/`),
* `../mist/train_LS.py` with `MistNetSplitter` (`../mist/model/mist_model.py`).

## Algorithm

Let z = 1 mean "train" and z = 0 mean "test". One run alternates, for `n_outer_loops` iterations:

1. **Split.** `split_data()` samples z for every spectrum, from Bernoulli(`train_ratio`) in the first iteration
   and from the splitter's softmax afterwards.
2. **Train the predictor** from scratch on the z = 1 spectra (80% train / 20% validation, early stopping on the
   validation loss) and measure the **generalisation gap** = test loss (z = 0) minus validation loss.
   The split with the largest gap so far is saved (`best_split.pkl`, `best_splitter.pth`, `splits_stats.json`).
3. **Train the splitter** on the z = 0 spectra with the frozen predictor attached (`splitter.add_predictor`).
   Its loss is `(w_gap * gap_loss + w_ratio * ratio_loss) / (w_gap + w_ratio)` with
   * `compute_gap_loss`: cross-entropy that sends spectra the predictor gets wrong (Jaccard distance above
     `jaccard_threshold`) to the test split and the others to the train split,
   * `compute_marginal_z_loss`: KL between the splitter's mean train probability and `train_ratio`, so the split
     keeps the requested size.
   (`compute_y_given_z_loss`, which equalises the fingerprint-bit marginals of the two splits, is implemented but
   switched off in the drivers.)

The loop stops after `splitter.patience` outer iterations without improving the gap.

## Running the provided drivers

```bash
pip install -e .                                     # once
cd ../other_baselines && python train_LS.py --config_file LS_config.yaml --results_dir ./ls_results
cd ../mist            && python train_LS.py --config_file LS_config.yaml --results_dir ./ls_results
```

Both drivers read a YAML config from `all_configs/`; the relevant blocks are

```yaml
train_params:   n_outer_loops: 500          # outer iterations
splitter:       train_ratio: 0.8            # target train fraction
                w_gap: 1.0                  # weight of the gap loss
                w_ratio: 1.0                # weight of the size loss
                jaccard_threshold: 0.85     # distance above which a prediction counts as a mistake (other_baselines)
                patience: 20                # outer iterations without gap improvement before stopping
                monitor: "splitter/loss"    # early-stopping metric for the splitter's own training
splitter_trainer: {max_epochs: 100, devices: 2, ...}   # pytorch_lightning.Trainer kwargs for the splitter
trainer:          {max_epochs: 100, devices: 2, ...}   # pytorch_lightning.Trainer kwargs for the predictor
```

The `data` / `dataset` blocks point to the processed data and to a split file whose ids define the pool of spectra
to partition (the whole train + val + test pool is used; the file's own train/test assignment is ignored). Paths in
the shipped configs are absolute and must be edited. Outputs, per run, in `<results_dir>/<model>_<dataset>/`:

| file | content |
|---|---|
| `best_split.pkl` | `{"train_indices", "test_indices", "val_loss", "test_loss", "split_stats", "outer_loop", "best_gap"}`; indices are positions in the dataset order (`data_ids.pkl` for MIST, `MSDataset(...).data` for the other baselines) |
| `splits_stats.json` | sizes and ratios of the best split and its gap |
| `best_splitter.pth` | state dict of the splitter that produced the best split |
| `best_predictor*.ckpt`, `latest_splitter*.ckpt` | Lightning checkpoints |

To use the split downstream, map the indices back to record ids and write
`data/splits/{dataset}/LS.json` with `train` / `val` / `test` lists (we hold out 20% of the learned train set as
`val`); every training script in this repository then accepts `LS.json` like any other split.

## Learning a split for your own model

You need two `pytorch_lightning.LightningModule`s and can reuse everything else:

1. **Predictor**: your fingerprint model. It must expose `encode_spectra(batch) -> FP probabilities`
   (or a tuple whose first element is that tensor, as MIST does) and log a validation and a test loss.
2. **Splitter**: a copy of your encoder whose head outputs **2 logits per spectrum** (column 1 = train) and that
   also sees the true fingerprint (both drivers concatenate an embedding of `batch["FP"]` to the spectrum
   embedding before the head). Implement:
   * `get_output(batch) -> logits`, used by `split_data`,
   * `add_predictor(predictor)`, storing the frozen predictor,
   * `training_step`, computing the two losses from `learning_to_split`:

   ```python
   from learning_to_split import compute_gap_loss, compute_marginal_z_loss

   def training_step(self, batch, batch_idx):
       logits = self.encode_spectra(batch)                       # (batch, 2)
       with torch.no_grad():
           FP_pred = self.predictor.encode_spectra(batch)        # (batch, n_bits) probabilities
       gap_loss = compute_gap_loss(logits, FP_pred, batch["FP"], jaccard_threshold=self.jaccard_threshold)
       ratio_loss, _ = compute_marginal_z_loss(logits, tar_ratio=self.tar_ratio)
       loss = (self.w_gap * gap_loss + self.w_ratio * ratio_loss) / (self.w_gap + self.w_ratio)
       self.log("splitter/loss", loss, on_epoch=True, sync_dist=True)
       return loss
   ```
3. **Outer loop**: copy `../other_baselines/train_LS.py`, replace `get_predictor` / `get_splitter` and the
   datamodule helpers with your own, and keep the rest. The dataloader passed to `split_data` must iterate the
   pool in a fixed order (`shuffle=False`), because the returned indices refer to that order.
