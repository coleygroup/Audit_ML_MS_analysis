# Technical Reproduction Steps

This file describes how to regenerate the open-dataset results reported in
`README.md`. The public README is the narrative response; this file is the
operator-facing checklist for rerunning the experiments.

NIST'23 is intentionally excluded. The open reproduction covers NPLIB1 and
MassSpecGym, each under random and scaffold splits.

## 1. Environment

MIST training depends on **upstream MIST** from the
[`main_v2` branch](https://github.com/samgoldman97/mist/tree/main_v2) of the MIST
repository, which is that repository's default branch. No fork, patched copy, or
vendored variant of MIST is used.

`main_v2` is the branch this codebase is written against. The older
`nmi_paper_v1` branch cannot be substituted: its `get_splitter` takes a
positional `splitter_name` argument that this repository never passes, and its
MAGMa featurizer expects a different on-disk layout (`magma_smiles_fp.hdf5` plus
index files) from the per-spectrum `magma_tsv/<spec>.magma` files the published
data export ships.

```bash
git clone --branch main_v2 --depth 1 \
  https://github.com/samgoldman97/mist.git external/mist

# environment.yml declares `name: ms-gen`; -n overrides it.
mamba env create -n ml-ms-analysis -f external/mist/environment.yml
mamba activate ml-ms-analysis

cd external/mist && pip install -e . && cd ../..
pip install tqdm pyyaml
```

`environment.yml` pins Python 3.8, `pytorch=1.9.*`, `pytorch-lightning=1.6.*`,
and `cudatoolkit=11.1`. Keep `pytorch-lightning` at `1.6.*`: `train.py` relies on
the PL 1.6 `Trainer(devices=, accelerator="gpu")` signature and on
`pytorch_lightning.utilities.rank_zero`. Confirm CUDA works on your card before
training:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.zeros(1).cuda())"
```

Verify that Python resolves `mist.data` to the cloned checkout:

```bash
python - <<'EOF'
import mist.data.datasets as d
from mist.data.datasets import SpectraMolDataset, SpecDataModule

print(d.__file__)
EOF
```

The printed path must point inside `external/mist`.

Two repository-side fixes are required, and both are already checked in:

- `benchmarked_models/mist/model/mist_model.py` imports `learning_to_split`,
  which is not present in this repository or in MIST. The three functions it
  provides are used only by `MistNetSplitter`, never by `MistNet`, which is the
  class both entry points instantiate, so the import is optional and fails
  loudly only if the learning-to-split path is actually used.
- `benchmarked_models/mist/utils/split_utils.py` re-reads split TSVs with
  `dtype=str`. Upstream `PresetSpectraSplitter` uses a bare `pd.read_csv`, so
  NPLIB1's purely numeric spectrum names are inferred as `int64` and never match
  the string names from the `.ms` file stems. Without this, **every NPLIB1 split
  silently comes back 0/0/0**; MassSpecGym is unaffected because its names are
  non-numeric. Confirm the fix is active by checking that a training run logs
  `Len of train: 12665` (NPLIB1 random) rather than `0`.

The DreaMS nearest-neighbour baseline also requires DreaMS and the pretrained
DreaMS weights from [Zenodo record 10997887](https://zenodo.org/records/10997887).
Use `ssl_model.ckpt` from that record for the default embedding cache path used
by `benchmarked_models/nearest_neighbour/01a_cache_dreaMS_emb.py`.

## 2. Data

For MIST retraining, use [original authors's Google Drive artifacts](https://drive.google.com/drive/folders/1v11lTwFSdlSRJ6ETLHqkbT809Ji9w0OY?usp=drive_link)
prepared for this reproduction. They contain the required `labels.tsv`, spectra, subformulae,
MAGMa outputs, and split TSVs for both NPLIB1 and MassSpecGym, so the two
upstream Zenodo MIST data exports are not required for the tuned runs reported
here.

```bash
python scripts/prepare_mist_google_drive_run.py \
  --run-root /path/to/scratch/mist_repro \
  --repo-root "$(pwd)"
```

This writes the MIST data under
`/path/to/scratch/mist_repro/raw/google_drive_mist_outputs/`, runtime configs
under `/path/to/scratch/mist_repro/manifests/runtime_configs/`, and a manifest
at `/path/to/scratch/mist_repro/manifests/google_drive_mist_outputs_manifest.json`.
Use that manifest to check split counts, label coverage, and train/val/test
overlap before launching training. The helper emits both the tuned configs
reported as `Corrected MIST` and the underperforming paper-style configs reported as
`MIST in Comment`.

If you need to rebuild from the original upstream MIST data exports instead,
the relevant records are:

- NPLIB1/CANOPUS: [Zenodo record 8316682](https://zenodo.org/records/8316682),
  `canopus_train_export_v2.tar`.
- MassSpecGym: [Zenodo record 11580401](https://zenodo.org/records/11580401),
  `MassSpecGym_mist_data.zip`.

The nearest-neighbour and table-building scripts also expect MGF/metadata and
candidate files in the repository-local layout below:

```text
data/
  MGF_files/
    NPLIB1/
      random/train.mgf
      random/test.mgf
      scaffold/train.mgf
      scaffold/test.mgf
    massspecgym/
      random/train.mgf
      random/test.mgf
      scaffold/train.mgf
      scaffold/test.mgf
  metadata/
    NPLIB1_metadata.tsv
    massspecgym_msg_all_metadata.tsv
  massspecgym/
    cands_df_test_formula_256.tsv
    cands_df_test_mass_256.tsv
```

The Google Drive MIST payload has the same logical fields, but it is kept under
the run root and referenced by generated runtime configs. Its data layout is:

```text
/path/to/scratch/mist_repro/raw/google_drive_mist_outputs/
  NPLIB1/
    labels.tsv
    spec_files/
    subformulae/default_subformulae/
    magma_outputs/magma_tsv/
    splits/random.tsv
    splits/scaffold.tsv
  massspecgym/
    labels.tsv
    spec_files/
    subformulae/default_subformulae/
    magma_outputs/magma_tsv/
    splits/random.tsv
    splits/scaffold.tsv
```

### PubChem retrieval data

For NPLIB1 PubChem retrieval, also download:

```text
https://zenodo.org/records/15529765/files/pubchem_formulae_inchikey.hdf5
```

The commands below assume it is saved as:

```text
data/pubchem/pubchem_formulae_inchikey.hdf5
```

### MassSpecGym candidate retrieval data

The MassSpecGym formula and mass candidate sets used in Section 8 come from the
official MassSpecGym Hugging Face dataset under
[`data/molecules`](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/tree/main/data/molecules),
which provides the retrieval candidate resources
`MassSpecGym1.5_retrieval_candidates_formula.json` and
`MassSpecGym1.5_retrieval_candidates_mass.json`. Convert them to the
repository-local TSV layout with:

```bash
python scripts/make_msg_retrieval_candidates.py \
  --labels-file data/metadata/massspecgym_msg_all_metadata.tsv \
  --no-split-file \
  --mgf-root data/MGF_files \
  --mgf-dataset massspecgym \
  --mgf-splits random scaffold \
  --raw-dir data/massspecgym/hf_raw \
  --out-dir data/massspecgym \
  --candidate-types formula mass \
  --max-candidates 256 \
  --overwrite
```

The expected output files are:

```text
data/massspecgym/cands_df_test_formula_256.tsv
data/massspecgym/cands_df_test_mass_256.tsv
```

## 3. NPLIB1 Test-Set Recovery

**Mandatory before training NPLIB1.** The MGF export contains 23 random and 5
scaffold test spectra (27 unique) that the MIST export omits. Skip this and MIST
is scored on 2,721 / 2,684 test spectra against the baselines' 2,744 / 2,689,
with no error raised.

```bash
python scripts/recover_nplib1_missing_test_spectra.py \
  --mist-data-root /path/to/scratch/mist_repro/raw/google_drive_mist_outputs/NPLIB1 \
  --mgf-root data/MGF_files/NPLIB1 \
  --metadata data/metadata/NPLIB1_metadata.tsv \
  --output-root /path/to/scratch/mist_repro/nplib1_extended
```

Point the `dataset` paths of the NPLIB1 configs (`labels_file`, `spec_folder`,
`subform_folder`, `split_file`) at the extended root. The recovered spectra are
added as `test` entries only, so no leakage is introduced; if you have already
trained, re-running `predict.py` against the extended root is enough — no
retraining. Verify:

```bash
awk -F'\t' 'NR>1 && $2=="test"' /path/to/scratch/mist_repro/nplib1_extended/splits/random.tsv | wc -l   # expect 2744
awk -F'\t' 'NR>1 && $2=="test"' /path/to/scratch/mist_repro/nplib1_extended/splits/scaffold.tsv | wc -l # expect 2689
```

MassSpecGym needs no equivalent step.

## 4. Train and Evaluate MIST

Run from `benchmarked_models/mist`. The first loop reproduces the tuned MIST
configuration used as the default in this repository. NPLIB1 configs must point
at the extended data root from Section 3.

```bash
cd benchmarked_models/mist
export MIST_CONFIG_DIR=/path/to/scratch/mist_repro/manifests/runtime_configs

for cfg in \
  nplib1_random_mist_config.yaml \
  nplib1_scaffold_mist_config.yaml \
  massspecgym_random_mist_config.yaml \
  massspecgym_scaffold_mist_config.yaml
do
  python train.py --config_dir "$MIST_CONFIG_DIR" --config_file "$cfg" --results_dir results
done
```

To reproduce the underperforming `MIST in Comment` setting, run the corresponding
paper-style configs. These use the older BCE objective with positive-class
weighting, hidden size 256, batch size 512, seed 17, no EMA, no cosine schedule,
and validation-based checkpointing.

```bash
for cfg in \
  nplib1_random_original_mist_config.yaml \
  nplib1_scaffold_original_mist_config.yaml \
  massspecgym_random_original_mist_config.yaml \
  massspecgym_scaffold_original_mist_config.yaml
do
  python train.py --config_dir "$MIST_CONFIG_DIR" --config_file "$cfg" --results_dir results
done
```

Then evaluate each run. `train.py` writes runs under lowercase `results/mist/`.

```bash
for run in \
  NPLIB1_MIST_4096_random \
  NPLIB1_MIST_4096_scaffold \
  MSG_MIST_4096_random \
  MSG_MIST_4096_scaffold
do
  python predict.py --checkpoint "results/mist/$run" --device cuda
done

for run in \
  NPLIB1_ORIGINAL_MIST_4096_random \
  NPLIB1_ORIGINAL_MIST_4096_scaffold \
  MSG_ORIGINAL_MIST_4096_random \
  MSG_ORIGINAL_MIST_4096_scaffold
do
  python predict.py --checkpoint "results/mist/$run" --device cuda
done
```

`--checkpoint` accepts either a run directory or an explicit `.ckpt` path. Given
a directory, `predict.py` prefers the checkpoint with the lowest monitored
value parsed from names of the form `{epoch:03d}-{val_loss:.5f}.ckpt`, and
falls back to `last.ckpt` when no such name exists. The tuned configs set
`save_last_only: True` and therefore emit only `last.ckpt`; pass the `.ckpt`
path directly if a run directory contains several unlabelled checkpoints.

The expected per-run outputs are:

```text
benchmarked_models/mist/results/mist/<run>/run.yaml
benchmarked_models/mist/results/mist/<run>/last.ckpt
benchmarked_models/mist/results/mist/<run>/test_results.pkl
benchmarked_models/mist/results/mist/<run>/test_performance.json
```

## 5. Corrected Nearest-Neighbour Baselines

Run the nearest-neighbour baselines under the default formula-first policy
(`same_formula_candidates_fallback`): candidates are restricted to training
spectra sharing the query formula, and when no such spectrum exists the search
falls back to the whole training set rather than dropping the query. Every test
spectrum is answered, so the baseline and MIST share one denominator:

```bash
python benchmarked_models/nearest_neighbour/02_compute_nn.py \
  --input-source mgf \
  --datasets NPLIB1 \
  --splits scaffold random \
  --metadata-file data/metadata/NPLIB1_metadata.tsv \
  --candidate-policy same_formula_candidates_fallback

python benchmarked_models/nearest_neighbour/02_compute_nn.py \
  --input-source mgf \
  --datasets massspecgym \
  --splits scaffold random \
  --metadata-file data/metadata/massspecgym_msg_all_metadata.tsv \
  --candidate-policy same_formula_candidates_fallback
```

`--input-source` selects where the binned spectrum comes from, and the two
sources are not interchangeable for NPLIB1: `mgf` bins the raw MGF intensities,
while `processed_pickle` bins the pickle's base-peak-normalised
`intensity_norm` field, which is the source the Comment's own pipeline used.
The peaks, formulas and fingerprints are identical between them; only the
intensity scaling differs, which changes how much the fixed precursor bin
weighs in the cosine. Binned NN on NPLIB1 moves from 0.212 (scaffold) / 0.720
(random) under `mgf` to 0.195 / 0.695 under `processed_pickle`; MassSpecGym and
DreaMS NN are unaffected, and the scaffold conclusions hold under either source.

Cache DreaMS embeddings, then run DreaMS nearest neighbour:

```bash
cd benchmarked_models/nearest_neighbour
python 01a_cache_dreaMS_emb.py
cd ../..

python benchmarked_models/nearest_neighbour/01b_compute_nn_dreaMS.py \
  --input-source mgf \
  --datasets NPLIB1 \
  --splits scaffold random \
  --metadata-file data/metadata/NPLIB1_metadata.tsv \
  --candidate-policy same_formula_candidates_fallback

python benchmarked_models/nearest_neighbour/01b_compute_nn_dreaMS.py \
  --input-source mgf \
  --datasets massspecgym \
  --splits scaffold random \
  --metadata-file data/metadata/massspecgym_msg_all_metadata.tsv \
  --candidate-policy same_formula_candidates_fallback
```

Run the full-training-set fingerprint oracle upper bound:

```bash
python benchmarked_models/nearest_neighbour/03_compute_fp_oracle_upper_bound.py \
  --input-source mgf \
  --datasets NPLIB1 \
  --splits scaffold random \
  --metadata-file data/metadata/NPLIB1_metadata.tsv \
  --candidate-policy same_formula_candidates_fallback

python benchmarked_models/nearest_neighbour/03_compute_fp_oracle_upper_bound.py \
  --input-source mgf \
  --datasets massspecgym \
  --splits scaffold random \
  --metadata-file data/metadata/massspecgym_msg_all_metadata.tsv \
  --candidate-policy same_formula_candidates_fallback
```

## 6. Formula-Filtered Diagnostic

The diagnostic intentionally reproduces the problematic skip-missing formula
setting. Run the same methods with `same_formula_candidates_skip_missing`.
The README-side table uses one common favorable subset per dataset/split. It
starts from the same-formula evaluated IDs and, when tuned MIST predictions are
available, intersects those IDs with the MIST `test_results.pkl` IDs before
scoring every method. The MIST row in that table therefore requires both the
MIST evaluation artifacts from Section 4 and the formula-filtered artifacts from
this section.

```bash
python benchmarked_models/nearest_neighbour/02_compute_nn.py \
  --input-source mgf \
  --datasets NPLIB1 \
  --splits scaffold random \
  --metadata-file data/metadata/NPLIB1_metadata.tsv \
  --candidate-policy same_formula_candidates_skip_missing

python benchmarked_models/nearest_neighbour/02_compute_nn.py \
  --input-source mgf \
  --datasets massspecgym \
  --splits scaffold random \
  --metadata-file data/metadata/massspecgym_msg_all_metadata.tsv \
  --candidate-policy same_formula_candidates_skip_missing

python benchmarked_models/nearest_neighbour/01b_compute_nn_dreaMS.py \
  --input-source mgf \
  --datasets NPLIB1 \
  --splits scaffold random \
  --metadata-file data/metadata/NPLIB1_metadata.tsv \
  --candidate-policy same_formula_candidates_skip_missing

python benchmarked_models/nearest_neighbour/01b_compute_nn_dreaMS.py \
  --input-source mgf \
  --datasets massspecgym \
  --splits scaffold random \
  --metadata-file data/metadata/massspecgym_msg_all_metadata.tsv \
  --candidate-policy same_formula_candidates_skip_missing

python benchmarked_models/nearest_neighbour/03_compute_fp_oracle_upper_bound.py \
  --input-source mgf \
  --datasets NPLIB1 \
  --splits scaffold random \
  --metadata-file data/metadata/NPLIB1_metadata.tsv \
  --candidate-policy same_formula_candidates_skip_missing

python benchmarked_models/nearest_neighbour/03_compute_fp_oracle_upper_bound.py \
  --input-source mgf \
  --datasets massspecgym \
  --splits scaffold random \
  --metadata-file data/metadata/massspecgym_msg_all_metadata.tsv \
  --candidate-policy same_formula_candidates_skip_missing
```

## 7. NPLIB1 PubChem Retrieval

Extract same-formula PubChem candidates from the HDF5 map:

```bash
python scripts/extract_pubchem_formula_candidates.py \
  --hdf5 data/pubchem/pubchem_formulae_inchikey.hdf5 \
  --metadata data/metadata/NPLIB1_metadata.tsv \
  --mgf-root data/MGF_files/NPLIB1 \
  --candidates-output results/comparison/nplib1_pubchem_hdf5_candidates.tsv.gz \
  --counts-output results/comparison/nplib1_pubchem_hdf5_candidate_counts.csv
```

Complete formulas missing from the HDF5 map with PubChem PUG REST:

```bash
python scripts/fetch_nplib1_pubchem_api_missing.py \
  --counts results/comparison/nplib1_pubchem_hdf5_candidate_counts.csv \
  --cache results/comparison/nplib1_pubchem_api_missing_cache.jsonl \
  --output results/comparison/nplib1_pubchem_api_missing_candidates.tsv.gz
```

Score NPLIB1 retrieval:

```bash
python scripts/compute_nplib1_pubchem_retrieval.py \
  --metadata data/metadata/NPLIB1_metadata.tsv \
  --candidates \
    results/comparison/nplib1_pubchem_hdf5_candidates.tsv.gz \
    results/comparison/nplib1_pubchem_api_missing_candidates.tsv.gz \
  --candidate-set pubchem_formula_hdf5_plus_api \
  --mist-results-root benchmarked_models/mist/results/mist \
  --nn-dir results/nearest_neighbour/nn_sim/same_formula_candidates_fallback \
  --dreams-dir results/nearest_neighbour/nn_sim_dreaMS/same_formula_candidates_fallback \
  --output-prefix results/comparison/nplib1_pubchem_formula_retrieval_full_test_methods \
  --workers 8
```

## 8. MassSpecGym Candidate Retrieval

Score MassSpecGym formula and mass candidate retrieval for all three prediction
methods:

```bash
python scripts/run_mist_msg_retrieval.py \
  --labels-file data/metadata/massspecgym_msg_all_metadata.tsv \
  --candidates-root data/massspecgym \
  --mist-results-root benchmarked_models/mist/results/mist \
  --nn-dir results/nearest_neighbour/nn_sim/same_formula_candidates_fallback \
  --dreams-dir results/nearest_neighbour/nn_sim_dreaMS/same_formula_candidates_fallback \
  --output-dir results/comparison/massspecgym_retrieval
```

## 9. Build README-Side Tables

Build machine-readable tables from the generated artifacts:

```bash
python scripts/build_readme_tables.py \
  --mist-results-root benchmarked_models/mist/results/mist \
  --nn-dir results/nearest_neighbour/nn_sim/same_formula_candidates_fallback \
  --dreams-dir results/nearest_neighbour/nn_sim_dreaMS/same_formula_candidates_fallback \
  --formula-nn-dir results/nearest_neighbour/nn_sim/same_formula_candidates_skip_missing \
  --formula-dreams-dir results/nearest_neighbour/nn_sim_dreaMS/same_formula_candidates_skip_missing \
  --formula-oracle-dir results/nearest_neighbour/fp_oracle_upper_bound/same_formula_candidates_skip_missing \
  --mgf-root data/MGF_files \
  --nplib-metadata data/metadata/NPLIB1_metadata.tsv \
  --msg-metadata data/metadata/massspecgym_msg_all_metadata.tsv \
  --nplib-retrieval results/comparison/nplib1_pubchem_formula_retrieval_full_test_methods \
  --msg-retrieval-dir results/comparison/massspecgym_retrieval \
  --output-dir results/comparison/readme_tables
```

This writes:

```text
results/comparison/readme_tables/full_test_fingerprint_metrics.csv
results/comparison/readme_tables/formula_filtered_metrics.csv
results/comparison/readme_tables/retrieval_metrics.csv
```

`formula_filtered_metrics.csv` includes the tuned MIST score restricted to the
same common evaluated same-formula subset used by the formula-filtered NN rows.
`leakage_fraction` is computed once per dataset/split subset as the fraction of
those evaluated test entries whose exact 2D InChIKey appears anywhere in the
training split; it is not computed from each method's selected nearest
neighbour.

The MassSpecGym official retrieval split table in `README.md` cites external
published benchmark results from the MassSpecGym/FRIGID line of work. It is
included for context and is not generated by the MIST/nearest-neighbour scripts
above.
