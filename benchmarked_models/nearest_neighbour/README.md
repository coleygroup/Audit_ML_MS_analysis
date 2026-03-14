# Nearest Neighbour Baseline

This folder contains the scripts used to implement the **nearest neighbour (NN) baseline** for the small-molecule mass spectrometry experiments.

The NN baseline retrieves candidate molecules for each query spectrum by identifying the most similar spectra in the **training set**. Similarity is computed either using **DreaMS embeddings** or **traditional spectral similarity**.

This baseline provides a simple retrieval-based reference point for evaluating machine learning models.

---

# Folder Structure
nearest_neighbour/
│
├── utils/ # Helper functions used by the scripts
│
├── 01a_cache_dreaMS_emb.py # Compute and cache DreaMS embeddings
├── 01b_compute_nn_dreaMS.py # Compute NN retrieval using DreaMS embeddings
│
├── 02_compute_nn.py # Compute NN retrieval using spectral similarity
│
├── 03_get_nn_results.ipynb # Aggregate predictions and compute metrics
│
└── README.md


---

# Pipeline

The nearest neighbour baseline consists of three main steps.

## 1. Cache DreaMS Embeddings

Compute spectral embeddings using the pretrained **DreaMS** model.

```bash
python 01a_cache_dreaMS_emb.py
```

This script:
   1. Loads train.mgf and test.mgf files from the dataset cache
   2. Computes embeddings using the default ssl_model.ckpt
   3. Saves embeddings to disk for reuse in nearest neighbour search. The embeddings are cached to avoid recomputing them during repeated experiments.

## 2. Compute Nearest Neighbours 

### Using DreaMS Embeddings 

```bash 
python 01b_compute_nn_dreaMS.py
``` 

This script:

   1. Loads cached DreaMS embeddings
   2. Computes nearest neighbours between test spectra and training spectra
   3. Saves the top-k (k=1) nearest neighbours for each query spectrum

### Using Spectral Similarity 

```bash 
python 02_compute_nn.py
```

This script computes nearest neighbours using traditional spectral similarity (e.g., cosine similarity on binned spectra) instead of learned embeddings.

### 3. Aggregate Results 

```bash 
python 03_get_nn_results.ipynb
```

The notebook:
   1. Loads nearest neighbour predictions
   2. Computes evaluation metrics
   3. Generates result tables used in the paper


### Notes 
   - Nearest neighbour retrieval is performed only against the training split.
   - DreaMS embeddings are computed using the default SSL checkpoint without contrastive fine-tuning to avoid potential data leakage.
   - Embeddings are cached to improve reproducibility and runtime efficiency.

### Data 
The datasets used in these experiments are available here:

https://drive.google.com/drive/folders/1v11lTwFSdlSRJ6ETLHqkbT809Ji9w0OY?usp=drive_link 