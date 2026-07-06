# MIST Config Families

This folder contains the runnable configs used by this reproduction. There are
two checked-in families:

- `*_mist_config.yaml`: current default tuned configs.
- `*_original_mist_config.yaml`: underperforming configs used for the `MIST in Comment` rows.

The table below also includes two reference families that are not fully
materialized as YAML files here: Official MISTv2, and **MIST vFRIGID**, meaning
the MIST version used inside the FRIGID model. MIST vFRIGID motivated our
"Tuned MIST" defaults.

| Setting                      |                                      MIST in Comment |                                                               Official MISTv2 |                                                                                       MIST vFRIGID |               Tuned MIST |
|------------------------------|--------------------------------------------------------------:|------------------------------------------------------------------------------:|---------------------------------------------------------------------------------------------------:|-------------------------:|
| Configurations               |                                 `*_original_mist_config.yaml` | [MIST repository](https://github.com/samgoldman97/mist/tree/main_v2#training) | [FRIGID repository](https://github.com/coleygroup/FRIGID/tree/MIST-FRIGID#training-models-) |     `*_mist_config.yaml` |
| Purpose                      | Reproduce the improper MIST configuration used in the Comment |                          What a faithful MIST reproduction should be based on |                                                          MIST version used inside the FRIGID model |  Tuned MIST in this repo |
| Loss                         |                                                           BCE |                                                                        cosine |                                                                                             cosine |                   cosine |
| Hidden size                  |                                                         `256` |                                                                `256` or `512` |                                                                                              `640` |                   `1024` |
| Batch size                   |                                                         `512` |                                                                         `128` |                                                                                              `256` |                    `256` |
| Max epochs                   |                                                         `200` |                                                                         `600` |                                                                                              `150` |                    `150` |
| Max peaks                    |                                               package default |                                                                          `15` |                                                                                               `10` |                     `10` |
| EMA                          |                                                           off |                                                                           off |                                                                                        on, `0.995` |              on, `0.995` |
| LR schedule                  |                                               no LR scheduler |                                                               no LR scheduler |                                                                           cosine schedule + warmup | cosine schedule + warmup |

## Important Distinction

The `*_original_mist_config.yaml` files reproduce the improper MIST setting
used in the Comment and are reported as `MIST in Comment`. They are
**not** a faithful reimplementation of MIST. The differences are substantial:
BCE loss instead of cosine loss, batch size `512` instead of `128`, and fewer
training epochs (`200` versus `600` in the MISTv2 CANOPUS config). Our "Tuned
MIST" is based on the MIST vFRIGID hyperparameter family while skipping the
larger FRIGID workflow and ICEBERG augmentation. Therefore, the optimal
parameters here are close to, but not identical to, the full FRIGID setting.

## How the Tuned MIST Parameters Were Selected

Hyperparameter tuning was needed because the Comment changed enough of the MIST
training setting that the original hyperparameters were no longer a safe
default. The objective, training schedule, batch size, and data pipeline all
differ from existing MIST-style runs. During the reproduction, we therefore kept
the MIST architecture fixed and tuned only training hyperparameters on the open
NPLIB1 and MassSpecGym random/scaffold settings.

The search was intentionally small and directional. We started from the MIST
vFRIGID-style setting: cosine loss, batch size `256`, maximum `150` epochs,
maximum `10` peaks, EMA with decay `0.995`, cosine learning-rate schedule with
warmup, and no data augmentation. We chose this starting point because it is the
best-performing configuration family on the MassSpecGym official benchmark. We
then compared it against the Comment-style BCE configuration and checked the
main unstable choices observed during reproduction: BCE versus cosine loss, and
larger hidden sizes (`640`, `1024`, and `2048`).

The final checked-in configurations are the most consistent setting from that
sweep and are stored in `*_mist_config.yaml`. Directionally, they use a larger
hidden size (`1024`), cosine loss, EMA, cosine schedule with warmup, batch size
`256`, maximum `150` epochs, maximum `10` peaks, and no augmentation. MAGMa
auxiliary supervision is also off in all tuned configs
(`magma_aux_loss: False`). These are the only intended MIST training changes
behind the `Tuned MIST` rows; they do not add contrastive fine-tuning, MAGMa
supervision, or ICEBERG augmentation.
