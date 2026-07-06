# Technical Reproduction Steps

This file describes how to regenerate the open-dataset results reported in
`README.md`. The public README is the narrative response; this file is the
operator-facing checklist for rerunning the experiments.

NIST'23 is intentionally excluded. The open reproduction covers NPLIB1 and
MassSpecGym, each under random and scaffold splits.

## 1. Environment

Use Python 3.10 or 3.11.

```bash
mamba create -y -n ml-ms-analysis python=3.10
mamba activate ml-ms-analysis
pip install numpy pandas scipy scikit-learn tqdm h5py pyyaml torch pytorch-lightning rdkit
```

MIST training depends on **MIST vFRIGID**, meaning the MIST version used inside
the FRIGID model, not the original MIST repository/package. Install the FRIGID
checkout that contains this MIST implementation before running the MIST
commands:

```bash
mkdir -p external
git clone --recurse-submodules https://github.com/coleygroup/FRIGID.git external/FRIGID
cd external/FRIGID
git submodule update --init --recursive
pip install -r ms-pred/requirements.txt
pip install -e ./ms-pred
pip install -e .
cd ../..
```

Verify that Python resolves `import mist` to the MIST vFRIGID implementation:

```bash
python - <<'PY'
import inspect
import mist

print(inspect.getfile(mist))
PY
```

If the printed path does not point inside `external/FRIGID`, put the FRIGID
source tree containing MIST vFRIGID first on `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/external/FRIGID/src:$PWD/benchmarked_models/mist:$PYTHONPATH"
```

The DreaMS nearest-neighbour baseline also requires DreaMS and the pretrained
DreaMS weights from [Zenodo record 10997887](https://zenodo.org/records/10997887).
Use `ssl_model.ckpt` from that record for the default embedding cache path used
by `benchmarked_models/nearest_neighbour/01a_cache_dreaMS_emb.py`.

## 2. Data

For MIST retraining, use [Serena Khoo's Google Drive artifacts](https://drive.google.com/drive/folders/1v11lTwFSdlSRJ6ETLHqkbT809Ji9w0OY?usp=drive_link)
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
reported as `Tuned MIST` and the underperforming paper-style configs reported as
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

The MassSpecGym formula and mass candidate sets used in Section 7 come from the
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

## 3. Train and Evaluate MIST

Run from `benchmarked_models/mist`. The first loop reproduces the tuned MIST
configuration used as the default in this repository.

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

The expected per-run outputs are:

```text
benchmarked_models/mist/results/mist/<run>/run.yaml
benchmarked_models/mist/results/mist/<run>/last.ckpt
benchmarked_models/mist/results/mist/<run>/test_results.pkl
benchmarked_models/mist/results/mist/<run>/test_performance.json
```

## 4. Corrected Nearest-Neighbour Baselines

Run binned-spectrum nearest neighbour on the full training candidate set:

```bash
python benchmarked_models/nearest_neighbour/02_compute_nn.py \
  --input-source mgf \
  --datasets NPLIB1 \
  --splits scaffold random \
  --metadata-file data/metadata/NPLIB1_metadata.tsv \
  --candidate-policy all_train_candidates

python benchmarked_models/nearest_neighbour/02_compute_nn.py \
  --input-source mgf \
  --datasets massspecgym \
  --splits scaffold random \
  --metadata-file data/metadata/massspecgym_msg_all_metadata.tsv \
  --candidate-policy all_train_candidates
```

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
  --candidate-policy all_train_candidates

python benchmarked_models/nearest_neighbour/01b_compute_nn_dreaMS.py \
  --input-source mgf \
  --datasets massspecgym \
  --splits scaffold random \
  --metadata-file data/metadata/massspecgym_msg_all_metadata.tsv \
  --candidate-policy all_train_candidates
```

Run the full-training-set fingerprint oracle upper bound:

```bash
python benchmarked_models/nearest_neighbour/03_compute_fp_oracle_upper_bound.py \
  --input-source mgf \
  --datasets NPLIB1 \
  --splits scaffold random \
  --metadata-file data/metadata/NPLIB1_metadata.tsv \
  --candidate-policy all_train_candidates

python benchmarked_models/nearest_neighbour/03_compute_fp_oracle_upper_bound.py \
  --input-source mgf \
  --datasets massspecgym \
  --splits scaffold random \
  --metadata-file data/metadata/massspecgym_msg_all_metadata.tsv \
  --candidate-policy all_train_candidates
```

## 5. Formula-Filtered Diagnostic

The diagnostic intentionally reproduces the problematic skip-missing formula
setting. Run the same methods with `same_formula_candidates_skip_missing`.
The README-side table uses one common favorable subset per dataset/split. It
starts from the same-formula evaluated IDs and, when tuned MIST predictions are
available, intersects those IDs with the MIST `test_results.pkl` IDs before
scoring every method. The MIST row in that table therefore requires both the
MIST evaluation artifacts from Section 3 and the formula-filtered artifacts from
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

## 6. NPLIB1 PubChem Retrieval

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
  --nn-dir results/nearest_neighbour/nn_sim/all_train_candidates \
  --dreams-dir results/nearest_neighbour/nn_sim_dreaMS/all_train_candidates \
  --output-prefix results/comparison/nplib1_pubchem_formula_retrieval_full_test_methods \
  --workers 8
```

## 7. MassSpecGym Candidate Retrieval

Score MassSpecGym formula and mass candidate retrieval for all three prediction
methods:

```bash
python scripts/run_mist_msg_retrieval.py \
  --labels-file data/metadata/massspecgym_msg_all_metadata.tsv \
  --candidates-root data/massspecgym \
  --mist-results-root benchmarked_models/mist/results/mist \
  --nn-dir results/nearest_neighbour/nn_sim/all_train_candidates \
  --dreams-dir results/nearest_neighbour/nn_sim_dreaMS/all_train_candidates \
  --output-dir results/comparison/massspecgym_retrieval
```

## 8. Build README-Side Tables

Build machine-readable tables from the generated artifacts:

```bash
python scripts/build_readme_tables.py \
  --mist-results-root benchmarked_models/mist/results/mist \
  --nn-dir results/nearest_neighbour/nn_sim/all_train_candidates \
  --dreams-dir results/nearest_neighbour/nn_sim_dreaMS/all_train_candidates \
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
