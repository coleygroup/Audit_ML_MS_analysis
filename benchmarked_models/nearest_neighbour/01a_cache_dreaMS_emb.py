"""
    Compute and cache DreaMS embeddings for train/test MGF splits.

    This script iterates over a set of datasets and split definitions, loads
    `train.mgf` and `test.mgf` files from a cache directory, computes DreaMS
    embeddings using `dreams.api.dreams_embeddings`, and saves the resulting
    embeddings as pickled arrays.

    This API call uses the default ssl_model.ckpt checkpoint without contrastive fine-tuning,
    to avoid potential data leakage in our evaluations.

"""

import os
from pathlib import Path
from dreams.api import dreams_embeddings

from utils import pickle_data

if __name__ == "__main__":

    MGF_cache_folder = Path("../../data/MGF_files") # Directory containing cached MGF spectra for each dataset/split 
    cache_folder = Path("../../results/nearest_neighbour/DreaMS_emb") # Directory where computed DreaMS embeddings will be stored
    if not os.path.exists(cache_folder): os.makedirs(cache_folder)

    datasets = ["NPLIB1", "massspecgym"] # Benchmark datasets evaluated in this study
    splits = ["scaffold", "random"] # Splitting strategies

    for dataset in datasets:

        for split in splits: 

            emb_folder = cache_folder / dataset / split
            if not os.path.exists(emb_folder): os.makedirs(emb_folder)

            # Get train embeddings
            train_emb_path = emb_folder / "train.pkl"

            if not os.path.exists(train_emb_path):
                    
                train_MGF_path = MGF_cache_folder / dataset / split / "train.mgf"
                emb = dreams_embeddings(train_MGF_path)
                pickle_data(emb, train_emb_path)
                print(f"Computed DreaMS embeddings for {dataset}/{split} train set: {emb.shape}")

            # Get test embeddings
            test_emb_path = emb_folder / "test.pkl"
            if not os.path.exists(test_emb_path):
                
                test_MGF_path = MGF_cache_folder / dataset / split / "test.mgf"
                emb = dreams_embeddings(test_MGF_path)
                pickle_data(emb, test_emb_path)
                print(f"Computed DreaMS embeddings for {dataset}/{split} test set: {emb.shape}")