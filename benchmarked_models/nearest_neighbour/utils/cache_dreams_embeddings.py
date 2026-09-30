"""
    Compute and cache DreaMS embeddings for train/test MGF splits.

    This script iterates over a set of datasets and split definitions, loads
    `train.mgf` and `test.mgf` files from a cache directory, computes DreaMS
    embeddings using `dreams.api.dreams_embeddings`, and saves the resulting
    embeddings as pickled arrays.

    This API call uses the default ssl_model.ckpt checkpoint without contrastive fine-tuning.

    Paths are resolved from this file's location, so the script can be run from any directory
    (conda env `dreams`):
        python utils/cache_dreams_embeddings.py                      # from nearest_neighbour/
        python -m utils.cache_dreams_embeddings                      # same, as a module

"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # nearest_neighbour/, so `utils` is importable
from utils.utils import pickle_data

ROOT = Path(__file__).resolve().parents[3]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["NPLIB1", "massspecgym"]) # Benchmark datasets evaluated in this study
    parser.add_argument("--splits", nargs="+", default=["scaffold", "random"]) # Splitting strategies
    parser.add_argument("--mgf_folder", type=Path, default=ROOT / "data" / "MGF_files") # Directory containing cached MGF spectra for each dataset/split
    parser.add_argument("--out_folder", type=Path, default=ROOT / "results" / "nearest_neighbour" / "DreaMS_emb") # Directory where computed DreaMS embeddings will be stored
    args = parser.parse_args()

    from dreams.api import dreams_embeddings

    MGF_cache_folder = args.mgf_folder
    cache_folder = args.out_folder
    if not os.path.exists(cache_folder): os.makedirs(cache_folder)

    for dataset in args.datasets:

        for split in args.splits:

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
