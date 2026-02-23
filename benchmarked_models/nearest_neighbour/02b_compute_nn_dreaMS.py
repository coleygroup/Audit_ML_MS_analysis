import os 
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from utils import load_pickle, load_json, pickle_data

def string_to_bits(string): 

    bits = np.array([int(c) for c in string])

    return bits

def get_info(data):

    FP_info, formula_info = {}, {}

    for r in tqdm(data):

        FP_info[str(r["id_"])] = string_to_bits(r["FPs"]["morgan4_4096"])
        formula_info[str(r["id_"])] = r["formula"]

    return FP_info, formula_info

if __name__ == "__main__":

    data_folder = Path("/data/rbg/users/klingmin/projects/MS_processing/data/")
    splits_folder = Path("/data/rbg/users/klingmin/projects/MS_processing/data_splits")
    cache_folder = Path("./cache/nearest_neighbour_sim")
    dreaMS_embs_folder = Path("./cache/DreaMS_emb")

    if not os.path.exists(cache_folder): os.makedirs(cache_folder)

    datasets = ["nist2023"]
    splits = ["scaffold_vanilla", "random"]

    # 1. Get the splits
    all_splits = {} 

    for dataset in datasets:

        all_splits[dataset] = {} 

        for split in splits: 

            current_filepath = splits_folder / dataset / "splits" / f"{split}.json"
            assert os.path.exists(current_filepath)

            split_ids = load_json(current_filepath)
            train, test = split_ids["train"], split_ids["test"]
            train = [t.replace(".pkl", "") for t in train]
            test = [t.replace(".pkl", "") for t in test]

            all_splits[dataset][split] = {"train": train,
                                          "test": test}

    # 2. Get the nearest neighbour now 
    for dataset in datasets:

        data, MS_info, FP_info, formula_info = None, None, None, None

        for split in splits: 

            output_path = cache_folder / f"{dataset}_{split}_dreaMS.pkl"
            if os.path.exists(output_path): continue 

            print(f"Processing {dataset}, {split} split now.")
            
            if data is None:
                
                data = load_pickle(data_folder / f"{dataset}" / f"{dataset}_w_mol_info_w_frag_CF_preds.pkl")
                FP_info, formula_info = get_info(data)
                print("Done loading data")

            train_ids, test_ids = all_splits[dataset][split]["train"], all_splits[dataset][split]["test"]

            train_MS = np.array(load_pickle(dreaMS_embs_folder / dataset / split / "train.pkl"))
            test_MS = np.array(load_pickle(dreaMS_embs_folder / dataset / split / "test.pkl"))

            train_FP = np.array([FP_info[id_] for id_ in train_ids])
            train_formula = np.array([formula_info[id_] for id_ in train_ids])

            computed_test_ids, top_train_ids = [],[]
            computed_test_FP, pred_FP = [],[]

            for test_idx, id_ in tqdm(enumerate(test_ids)):

                test_formula = formula_info[id_]
                test_FP = FP_info[id_]

                # Let us sieve out the train 
                sieved_idx = [idx for idx, f in enumerate(train_formula) if f == test_formula]
                if len(sieved_idx) == 0: continue

                # Get the prediction now
                sim = cosine_similarity([test_MS[test_idx]], train_MS[sieved_idx])
                train_idx = np.argmax(sim, axis = 1)[0]

                computed_test_ids.append(id_)
                top_train_ids.append(sieved_idx[train_idx])

                computed_test_FP.append(test_FP)
                pred_FP.append(train_FP[sieved_idx[train_idx]])
            
            pickle_data((test_ids, top_train_ids, computed_test_FP, pred_FP), output_path)