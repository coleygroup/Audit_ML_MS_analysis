# Benchmarked Models

This directory contains benchmarked machine learning models used in the ML_MS_analysis project.

## Folder Structure 
``` 
benchmarked_models/
│
├── mist/
│   Implementation and evaluation scripts for the MIST model.
│
├── nearest_neighbour/
│   Nearest-neighbour baselines using spectral similarity and learned embeddings
│   (e.g., DreaMS embeddings).
│
├── other_baselines/
│   Additional baseline models used for comparison.
│
└── README.md
``` 

## Corrected Evaluation Workflow

Nearest-neighbour baselines now require an explicit candidate policy:

```bash
python benchmarked_models/nearest_neighbour/02_compute_nn.py \
  --input-source auto \
  --candidate-policy all_train_candidates

python benchmarked_models/nearest_neighbour/02_compute_nn.py \
  --input-source auto \
  --candidate-policy same_formula_candidates_full

python benchmarked_models/nearest_neighbour/01b_compute_nn_dreaMS.py \
  --input-source auto \
  --candidate-policy all_train_candidates
```

Use `same_formula_candidates_skip_missing` to reproduce the old problematic
formula-matched subset behavior: filter candidates by exact formula first, and
drop test queries with no same-formula train candidate. The older alias
`same_formula_candidates_legacy` is kept for compatibility. The corrected
same-formula mode keeps all test queries and marks no-candidate cases
explicitly.

MassSpecGym metadata is available at:

```bash
data/metadata/massspecgym_msg_all_metadata.tsv
```

This TSV was created with `scripts/recover_metadata_from_labels.py` by taking
the MassSpecGym labels export, filtering it to spectrum IDs present in
`data/MGF_files/massspecgym/*/*.mgf`, and retaining the metadata columns needed
for formula-aware evaluation (`formula`, `smiles`, `inchikey`, instrument and
precursor metadata).

Use it for formula-aware MSG NN runs:

```bash
python benchmarked_models/nearest_neighbour/02_compute_nn.py \
  --input-source mgf \
  --datasets massspecgym \
  --candidate-policy same_formula_candidates_full \
  --metadata-file data/metadata/massspecgym_msg_all_metadata.tsv
```

Compute a nearest-neighbour upper bound by directly choosing, for each test
entry, the training fingerprint with the highest Jaccard:

```bash
python benchmarked_models/nearest_neighbour/03_compute_fp_oracle_upper_bound.py \
  --input-source mgf \
  --datasets NPLIB1 massspecgym \
  --splits random scaffold \
  --candidate-policy all_train_candidates
```

NPLIB1 metadata is available at `data/metadata/NPLIB1_metadata.tsv`. Use the
matching metadata sidecar when running formula-aware or exact-structure
evaluations.

NIST23 files are not publicly available.

Build README-side metric tables after MIST, nearest-neighbour, and retrieval
outputs have been generated:

```bash
python scripts/build_readme_tables.py
```
