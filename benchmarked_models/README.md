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
├── tune_threshold.py
│   Selects the fingerprint decision threshold on the validation split and applies it to
│   test, for any model with <split>_results.pkl. Shared by mist/ and other_baselines/.
│
└── README.md
```

## Evaluation threshold

Every `predict.py` here binarises the predicted fingerprint at a fixed 0.5. To select the threshold on the
validation split instead, produce validation predictions (`predict.py --split val`) and run `tune_threshold.py`.
