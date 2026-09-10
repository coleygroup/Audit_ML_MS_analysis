# MIST Config Families

This folder contains the runnable configs used by this reproduction. There are
two checked-in families, and they differ by exactly two lines:

- `*_original_mist_config.yaml`: the configuration used for the `MIST in Comment`
  rows, which trains with a BCE objective.
- `*_mist_config.yaml`: `Corrected MIST`, the same architecture trained with MIST's own
  default cosine objective.

Both families are run against unmodified upstream MIST from the
[`main_v2` branch](https://github.com/samgoldman97/mist/tree/main_v2) of the MIST
repository. No fork, patched copy, or vendored variant of MIST is used anywhere
in this reproduction.

| Setting              |                MIST in Comment |                    Corrected MIST |
|----------------------|-------------------------------:|----------------------------------:|
| Configurations       | `*_original_mist_config.yaml`  |             `*_mist_config.yaml`  |
| Loss                 |                            BCE |                        **cosine** |
| `val_monitor`        |                 `val_bce_loss` |              **`val_cos_loss`**   |
| Hidden size          |                          `256` |                             `256` |
| Batch size           |                          `512` |                             `512` |
| Max epochs           |                          `200` |                             `200` |
| MAGMa auxiliary loss |                             on |                                on |
| Seed                 |                           `17` |                              `17` |
| Decision threshold   |                    fixed `0.5` | selected on the validation split |

Everything except the objective is held fixed, so the difference between the two
rows is attributable to the loss function alone.

## Why `val_monitor` Must Move With The Loss

The two changes are not independent. `val_monitor` selects both the early-stopping
signal and the saved checkpoint. Under cosine training, `val_bce_loss` reaches its
minimum at epoch 1 and never improves, so early stopping fires around epoch 20 and
the epoch-1 checkpoint is the one kept. The run still exits 0 and still writes a
plausible-looking score, which makes the failure easy to miss. Changing `loss_fn`
without changing `val_monitor` therefore does not measure the cosine objective at
all.

## The Decision Threshold Is Fitted, Not Assumed

Jaccard requires binarizing the predicted fingerprint, and the cut point is a free
parameter. The original pipeline hard-coded it at `0.5`. Here it is chosen by
`predict.py`, which sweeps a `0.02`-`0.98` grid on the **validation** split before
the test split is scored. `test_performance.json` records the selected threshold,
the validation Jaccard behind it, and `jaccard_at_0.5` so the calibration can be
audited from the artifact.

The selected values differ sharply by objective, which is why a single fixed cut
cannot serve both:

| Split                | MIST in Comment (BCE) | Corrected MIST (cosine) |
|----------------------|----------------------:|------------------------:|
| NPLIB1 scaffold      |                  0.80 |                    0.16 |
| NPLIB1 random        |                  0.82 |                    0.14 |
| MassSpecGym scaffold |                  0.78 |                    0.12 |
| MassSpecGym random   |                  0.84 |                    0.24 |

A cosine-trained model evaluated at the fixed `0.5` cut scores 0.018 Jaccard on
NPLIB1 scaffold and 0.317 at its validation-selected cut. Reporting the former
would measure calibration, not fingerprint quality.

## Reference: Official MISTv2

For context, the MIST authors' own CANOPUS configuration
([MIST repository](https://github.com/samgoldman97/mist/tree/main_v2#training))
uses cosine loss, hidden size `256` or `512`, batch size `128`, up to `600`
epochs, and `max_peaks: 15`. It is not checked in here because this reproduction
deliberately holds the Comment's architecture and training budget fixed and
varies only the objective; adopting the full upstream recipe would confound the
loss change with a larger training budget.
