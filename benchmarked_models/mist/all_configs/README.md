# MIST Config Families

This folder contains the runnable configs used by this reproduction. All of them
run against unmodified upstream MIST from the
[`main_v2` branch](https://github.com/samgoldman97/mist/tree/main_v2) of the MIST
repository. No fork, patched copy, or vendored variant of MIST is used anywhere in
this reproduction.

There are two axes: the training **objective** (the Comment's BCE versus MIST's
default cosine) and the **epoch budget plus checkpoint rule** (200 epochs read at
the validation-loss-selected checkpoint, as the Comment does, versus 600 epochs
read at the final checkpoint). The four combinations are all checked in, so the
two effects can be separated:

|                                 | read at loss-selected checkpoint, 200 ep | read at final checkpoint, 600 ep                        |
| ------------------------------- | ---------------------------------------- | ------------------------------------------------------- |
| **BCE** (the Comment's objective) | `*_original_mist_config.yaml`            | `*_original_mist_e600_config.yaml` &nbsp;*(headline)*   |
| **cosine** (MIST's default)       | `*_mist_config.yaml`                     | `*_mist_e600_config.yaml`                               |

- `*_original_mist_config.yaml`: the configuration behind the `MIST in Comment`
  rows, reproducing the Comment's protocol exactly.
- `*_original_mist_e600_config.yaml`: **the headline configuration.** The
  Comment's own BCE objective at the 600-epoch budget, scored from `last.ckpt`.
  This is the `MIST in Comment, final checkpoint` row. It is the headline because
  it changes nothing about the Comment's model or objective — only the epoch
  budget and which checkpoint is read — and it isolates how much of the published
  shortfall is checkpoint selection rather than the objective.
- `*_mist_config.yaml`: the cosine objective under the Comment's original 200-epoch
  budget and loss-selected checkpoint. Retained for reference; it was the headline
  row in revisions of this repository before 20261003.
- `*_mist_e600_config.yaml`: the `MIST, default objective` row. Identical to the
  configuration above except for the objective (cosine, MIST's own default) and
  `val_monitor`. Reported beside the headline because the objective is MIST's
  documented default, not because it carries the result: it is worth +0.006 and
  +0.004 Jaccard on the two scaffold splits and -0.010 on NPLIB1 random.

A fifth family shows what a hyperparameter search buys and is **not** a headline
row:

- `*_mistv2_h1024_sched_e600_config.yaml`: `Tuned MIST`, widened to
  `hidden_size: 1024` and trained for `600` epochs under a cosine learning-rate
  schedule with 10% warmup and an EMA of the weights. It gains on the leakage-heavy
  random splits and loses on both scaffold splits. It is not reported in
  [README.md](../../../README.md); its scores are read directly from each run's
  `test_performance.json`, and it is kept here as a record of what a capacity and
  schedule search buys on the leakage-heavy random splits.

| Setting              |                MIST in Comment |     MIST in Comment, 600 ep |            MIST, default objective |                            Tuned MIST |
| -------------------- | -----------------------------: | --------------------------: | ------------------------: | ------------------------------------: |
| Configurations       |  `*_original_mist_config.yaml` | `*_original_mist_e600_config.yaml` | `*_mist_e600_config.yaml` | `*_mistv2_h1024_sched_e600_config.yaml` |
| Loss                 |                            BCE |                         BCE |                **cosine** |                                cosine |
| `val_monitor`        |                 `val_bce_loss` |              `val_bce_loss` |      **`val_cos_loss`**   |                        `val_cos_loss` |
| Checkpoint reported  |             loss-selected      |            **final epoch**  |          **final epoch**  |                         loss-selected |
| Hidden size          |                          `256` |                       `256` |                     `256` |                            **`1024`** |
| Batch size           |                          `512` |                       `512` |                     `512` |        `128` x 4 accumulation = `512` |
| Max epochs           |                          `200` |                   **`600`** |               **`600`**   |                             **`600`** |
| `patience`           |                           `20` |                       `600` |                     `600` |                                   n/a |
| LR schedule          |                           none |                        none |                      none |    **cosine + 10% warmup, EMA 0.995** |
| MAGMa auxiliary loss |                             on |                          on |                        on |                                    on |
| Seed                 |                           `17` |                        `17` |                      `17` |                                  `17` |
| Decision threshold   |                    fixed `0.5` | fitted on validation        |      fitted on validation |                  fitted on validation |

`*_mist_e600_config.yaml` and `*_original_mist_e600_config.yaml` differ in exactly
three lines — `exp_name`, `loss_fn` and `val_monitor` — so the difference between
those two columns is attributable to the objective alone. `Tuned MIST` changes
width, schedule and checkpoint rule together and is reported as an aggregate, not
as an attribution of any single knob. Its effective batch size matches the other
families; the `128` x 4 split only exists because `hidden_size: 1024` at batch
`512` exceeds 24 GB (the pairwise attention tensor is batch x peaks^2 x hidden).

## Why The Reported Checkpoint Is The Final One, Not The Selected One

`val_monitor` controls which checkpoint is kept, and neither `val_bce_loss` nor
`val_cos_loss` tracks the metric actually reported (mean Jaccard after
binarization). The validation loss flattens while thresholded Jaccard keeps
improving, so the monitor returns an early checkpoint. Under the Comment's BCE
objective it is severe:

| Split                | Epoch `val_bce_loss` selects | Jaccard there | Jaccard at epoch 600 |
| -------------------- | ---------------------------: | ------------: | -------------------: |
| MassSpecGym scaffold |                  **epoch 5** |         0.325 |                0.371 |
| NPLIB1 scaffold      |                 **epoch 10** |         0.295 |                0.324 |
| NPLIB1 random        |                     epoch 45 |         0.636 |                0.743 |
| MassSpecGym random   |                    epoch 526 |         0.835 |                0.838 |

`val_cos_loss` tracks the reported metric better but is still early on the scaffold
splits (epoch 41 on NPLIB1, epoch 18 on MassSpecGym; reading epoch 600 instead is
worth +0.010 and +0.011). The `*_e600` configs therefore set `save_last: True`, and
the reported score is taken from `last.ckpt`. This uses no validation information
for checkpoint selection at all, which is why it is reported for every row rather
than only for the configuration it favours.

Raising `max_epochs` without changing the checkpoint rule accomplishes very
little, because the monitor discards the extra epochs: on NPLIB1 random the
loss-selected score is 0.638 at a 200-epoch budget and 0.636 at a 600-epoch
budget, while the final checkpoint of that same 600-epoch run reaches 0.743.

## Why `val_monitor` Must Move With The Loss

`val_monitor` also selects the early-stopping signal. Under cosine training,
`val_bce_loss` reaches its minimum at epoch 1 and never improves, so early
stopping fires around epoch 20 and the epoch-1 checkpoint is the one kept. The run
still exits 0 and still writes a plausible-looking score, which makes the failure
easy to miss. Changing `loss_fn` without changing `val_monitor` therefore does not
measure the cosine objective at all.

## The Decision Threshold Is Fitted, Not Assumed

Jaccard requires binarizing the predicted fingerprint, and the cut point is a free
parameter. The original pipeline hard-coded it at `0.5`. Here it is chosen by
`predict.py`, which sweeps a `0.02`-`0.98` grid on the **validation** split before
the test split is scored. `test_performance.json` records the selected threshold,
the validation Jaccard behind it, and `jaccard_at_0.5` so the calibration can be
audited from the artifact.

The selected values differ sharply by objective, which is why a single fixed cut
cannot serve both:

| Split                | MIST in Comment (BCE, 200 ep) | MIST in Comment, 600 ep (BCE) | MIST, default objective (cosine, 600 ep) | Tuned MIST (cosine) |
| -------------------- | ----------------------------: | ----------------------------: | ------------------------------: | ------------------: |
| NPLIB1 scaffold      |                          0.80 |                          0.86 |                            0.20 |                0.14 |
| NPLIB1 random        |                          0.82 |                          0.82 |                            0.24 |                0.16 |
| MassSpecGym scaffold |                          0.78 |                          0.84 |                            0.32 |                0.12 |
| MassSpecGym random   |                          0.84 |                          0.84 |                            0.30 |                0.42 |

How much the fitted threshold is worth depends strongly on how long the model
trained. The 200-epoch, loss-selected cosine run was severely miscalibrated: it
scored 0.018 Jaccard at `0.5` on NPLIB1 scaffold against 0.317 at its fitted cut.
The same configuration at 600 epochs, read at the final checkpoint, scores 0.298 at
`0.5` and 0.330 at its fitted cut. The fitted cut is still worth having — and under
BCE it remains worth +0.010 to +0.030 — but most of what it appeared to recover
earlier was an artifact of stopping training early. Comparing models trained under
different objectives at a shared fixed threshold still measures calibration as
much as fingerprint quality.

## Reference: Official MISTv2

For context, the MIST authors' own CANOPUS configuration
([MIST repository](https://github.com/samgoldman97/mist/tree/main_v2#training))
uses cosine loss, hidden size `256` or `512`, batch size `128`, up to `600`
epochs, and `max_peaks: 15`. The `*_e600` families here reach the same `600`-epoch
budget. They still differ from the upstream recipe in batch size and `max_peaks`,
which are left at the Comment's values so that the BCE and cosine columns remain a
three-line difference from each other.
