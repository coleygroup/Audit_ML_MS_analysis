import os
import copy
import argparse
import numpy as np 
from tqdm import tqdm 

import torch
import torch.nn.functional as F 

from modules import *
from dataloader import MSDataset

from utils import pickle_data, write_json, read_config, load_pickle

def to_binary(FP, threshold):

    FP = torch.sigmoid(FP).cpu().numpy()
    FP = (FP > threshold).astype(int)

    return FP 

@torch.no_grad()
def get_loss(FP_pred, FP):
    return F.binary_cross_entropy_with_logits(FP_pred, FP, reduce = False)

@torch.no_grad()
def batch_jaccard_index(FP_pred, FP):

    # Intersection = bitwise AND
    intersection = np.logical_and(FP, FP_pred).sum(axis=1)

    # Union = bitwise OR
    union = np.logical_or(FP, FP_pred).sum(axis=1)

    # Avoid division-by-zero by adding a small epsilon
    jaccard_scores = intersection / (union + 1e-9)

    return jaccard_scores

@torch.no_grad()
def forward(model_name, model, batch, device, include_adduct, include_CE, include_instrument):

    adduct, CE, instrument = None, None, None
    if include_adduct: adduct = batch["adduct"].to(device)
    if include_CE: CE = batch["CE"].to(device)
    if include_instrument: instrument = batch["instrument"].to(device)
    
    if model_name == "binned_MS_encoder":
        
        FP_pred, _ = model(batch["binned_MS"].to(device), adduct, CE, instrument)

    elif model_name == "MS_encoder":

        mz, intensities, mask = batch["mz"], batch["intensities"], batch["mask"]
        binned_ms = batch["binned_MS"]

        mz, intensities, mask = mz.to(device), intensities.to(device), mask.to(device)
        binned_ms = binned_ms.to(device)
        FP_pred, _ = model(mz, intensities, mask, binned_ms, adduct, CE, instrument)

    elif model_name == "formula_encoder":

        intensities, formula, mask = batch["intensities"], batch["formula"], batch["mask"]
        binned_ms = batch["binned_MS"]

        intensities, formula, mask = intensities.to(device), formula.to(device), mask.to(device)
        binned_ms = binned_ms.to(device)

        FP_pred, _ = model(intensities, formula, mask, binned_ms, adduct, CE, instrument)

    elif model_name == "frag_encoder":

        intensities, mask = batch["intensities"], batch["mask"]
        frags_tokens, frags_mask, frags_weight = batch["frags_tokens"], batch["frags_mask"], batch["frags_weight"]
        binned_ms = batch["binned_MS"]

        intensities, mask = intensities.to(device), mask.to(device)
        frags_tokens, frags_mask, frags_weight  = frags_tokens.to(device), frags_mask.to(device), frags_weight.to(device)  
        binned_ms = binned_ms.to(device)

        FP_pred, _ = model(intensities, mask, binned_ms, frags_tokens, frags_mask, frags_weight)

    else:
        raise Exception() 
    
    return FP_pred 

@torch.no_grad()
def predict(model, config, device, batch_size, threshold = 0.5):

    model_name = config["model"]["name"] 
    project = config["project"]

    get_CF, get_frags = False, False

    if model_name == "formula_encoder": get_CF = True 
    if model_name == "frag_encoder": get_frags = True 

    # Get the split file 
    if project == "FP_baselines_FT":

        sampling_strategy = config["sampling_strategy"]
        sampling_ratio = config["ratio"]

        if sampling_strategy in ["IF_val", "IF_test"]:
            
            model_mapping = {"binned_MS_encoder" : "binned", "MS_encoder" : "MS", "formula_encoder": "formula"}
            model_acryn = model_mapping[model_name]
            split_file = os.path.join(config["data"]["splits_folder"], config["data"]["dataset"], 
                                                        "splits_sampling", sampling_strategy, 
                                                        f"sampled_{sampling_strategy}_{model_acryn}_{sampling_ratio}.json")
        else:
            split_file = os.path.join(config["data"]["splits_folder"], config["data"]["dataset"], 
                                                        "splits_sampling", sampling_strategy, 
                                                        f"sampled_{sampling_strategy}_{sampling_ratio}.json")
            
        assert os.path.exists(split_file)
        
        # Get the config file of the original model 
        original_results_folder, _ = get_FT_checkpoint_path(config["model"]["results_dir"], config["data"]["dataset"], model_name)
        config = read_config(os.path.join(original_results_folder, "run.yaml")) # overwrite the config
        config["data"]["split_file"] = split_file # overwrite the config file 

    elif "results_w_sampling" in config["checkpoint"]:

        sampling_strategy = config["sampling_strategy"]
        sampling_ratio = config["ratio"]

        if sampling_strategy in ["IF_val", "IF_test"]:
            
            model_mapping = {"binned_MS_encoder" : "binned", "MS_encoder" : "MS", "formula_encoder": "formula"}
            model_acryn = model_mapping[model_name]
            split_file = os.path.join(config["data"]["splits_folder"], config["data"]["dataset"], 
                                                        "splits_sampling", sampling_strategy, 
                                                        f"sampled_{sampling_strategy}_{model_acryn}_{sampling_ratio}_combined.json")
        else:
            split_file = os.path.join(config["data"]["splits_folder"], config["data"]["dataset"], 
                                                        "splits_sampling", sampling_strategy, 
                                                        f"sampled_{sampling_strategy}_{sampling_ratio}_combined.json")
            
        assert os.path.exists(split_file)
        
        # Get the config file of the original model 
        config["data"]["split_file"] = split_file # overwrite the config file 

    else:
        config["data"]["split_file"] = os.path.join(config["data"]["splits_folder"], config["data"]["dataset"], "splits", config["data"]["split_file"])

    # Update the data directory 
    config["data"]["dir"] = os.path.join(config["data"]["data_folder"], config["data"]["dataset"], "frags_preds")
    config["data"]["adduct_file"] = os.path.join(config["data"]["data_folder"], config["data"]["dataset"], "all_adducts.pkl")
    config["data"]["instrument_file"] = os.path.join(config["data"]["data_folder"], config["data"]["dataset"], "all_instruments.pkl")

    # Check if we are getting the meta information 
    feats_params = config["model"]["feats_params"]
    include_adduct, include_CE, include_instrument = feats_params["include_adduct"], feats_params["include_CE"], feats_params["include_instrument"]

    dataset = MSDataset(dir = config["data"]["dir"],
                        split_file = config["data"]["split_file"],
                        adduct_file = config["data"]["adduct_file"],
                        instrument_file = config["data"]["instrument_file"],
                        batch_size = batch_size,
                        num_workers = 4,
                        max_da = config["data"]["max_da"], 
                        max_MS_peaks = config["data"]["max_MS_peaks"],
                        bin_resolution = config["data"]["bin_resolution"], 
                        FP_type = config["data"]["FP_type"],
                        intensity_type = config["data"]["intensity_type"], 
                        intensity_threshold = config["data"]["intensity_threshold"],
                        considered_atoms = config["data"]["considered_atoms"],
                        n_frag_candidates = config["data"]["n_frag_candidates"],
                        chemberta_model = config["data"]["chemberta_model"],
                        return_id_ = True, 
                        get_CF = get_CF,
                        get_frags = get_frags)

    data_loader = dataset.test_dataloader()
    
    # Run model predictions
    id_list, predictions, GT, losses, jaccard_scores = [], [], [], [], []
    total_loss, total_jaccard, total = 0,0, 0

    for batch in tqdm(data_loader):
        
        # Unpack the batch 
        id_ = batch["id_"]
        FP = batch["FP"]

        # Forward pass
        FP_pred = forward(model_name, model, batch, device, include_adduct, include_CE, include_instrument)
        FP_pred = FP_pred.cpu()

        # Get the loss 
        loss = get_loss(FP_pred, FP)
        loss = loss.mean(-1)
        jaccard = batch_jaccard_index(to_binary(FP_pred, threshold), FP.numpy())

        total_loss += loss.mean(-1).item() * FP_pred.size(0)
        total_jaccard += jaccard.sum()
        total += FP_pred.size(0)

        # Save the predctions 
        id_list.extend(id_)
        predictions.append(FP_pred)
        GT.append(FP)
        losses.extend(loss.numpy().tolist())
        jaccard_scores.extend(jaccard.tolist())

    # Format the predictions
    predictions = torch.cat(predictions, dim = 0).numpy().tolist()
    GT = torch.cat(GT, dim = 0).numpy().tolist()
    predictions = {id_list[i]: {"pred": predictions[i], "GT": GT[i], "loss": losses[i], "jaccard": jaccard_scores[i]} for i in range(len(id_list))}
    
    # Get the average loss 
    avg_loss = total_loss / total 

    # Get the average jaccard loss 
    avg_jaccard = total_jaccard / total

    return predictions, avg_loss, avg_jaccard

def get_checkpoint_path(folder):

    checkpoints = [f for f in os.listdir(folder) if f.endswith(".ckpt")]
    best_checkpoint, lowest_loss = "", 1e4

    for c in checkpoints:

        loss = float(c.replace("-v1", "").replace(".ckpt", "").split("=")[-1]) # hack 
        if loss < lowest_loss:
            lowest_loss = loss 
            best_checkpoint = c 
    
    return os.path.join(folder, best_checkpoint)

def get_FT_checkpoint_path(results_cache, dataset, model):
    model_mapping = {"binned_MS_encoder" : "binned_", "MS_encoder" : "MS_", "formula_encoder": "formula_"}
    results_cache = os.path.join(results_cache, f"{dataset}_sieved")
    checkpoint_folder = [f for f in os.listdir(results_cache) if model_mapping[model] in f and "scaffold_vanilla" in f and "scaffold_vanilla_sieved_test" not in f] # Hack
    assert len(checkpoint_folder) == 1 
    checkpoint_folder = os.path.join(results_cache, checkpoint_folder[0])

    checkpoints = [f for f in os.listdir(checkpoint_folder) if f.endswith(".ckpt")]
    best_checkpoint, lowest_loss = "", 1e4

    for c in checkpoints:

        loss = float(c.replace("-v1", "").replace(".ckpt", "").split("=")[-1]) # hack 
        if loss < lowest_loss:
            lowest_loss = loss 
            best_checkpoint = c 

    best_checkpoint = os.path.join(checkpoint_folder, best_checkpoint)

    return checkpoint_folder, best_checkpoint

def main(args):

    # Get the checkpoint and config
    checkpoint_dir = args.checkpoint 
    config = read_config(os.path.join(checkpoint_dir, "run.yaml"))
    
    # Update the config 
    config["checkpoint"] = args.checkpoint
    if config["project"] == "FP_baselines_FT" or "results_w_sampling" in config["checkpoint"]:
        config["sampling_strategy"] = checkpoint_dir.split("/")[-2]
        config["ratio"] = int(checkpoint_dir.split("/")[-1].split("_")[-1])

    # Load the model
    model_name = config["model"]["name"]  
    
    if model_name == "binned_MS_encoder":
        model = MSBinnedModel.load_from_checkpoint(get_checkpoint_path(checkpoint_dir))

    elif model_name == "MS_encoder":
        model = MSTransformerEncoder.load_from_checkpoint(get_checkpoint_path(checkpoint_dir))

    elif model_name == "formula_encoder":
        model = FormulaTransformerEncoder.load_from_checkpoint(get_checkpoint_path(checkpoint_dir))

    else:
        raise NotImplementedError()

    model.eval()
    model.to(args.device)

    # Get the predictions 
    predictions, loss, jaccard = predict(model, config, args.device, args.batch_size)
     
    # Write the predictions
    output_path = os.path.join(checkpoint_dir, "test_results.pkl")
    pickle_data(predictions, output_path)
    write_json({"loss": loss, "jaccard": jaccard}, os.path.join(checkpoint_dir, "test_performance.json"))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type = int, default = 512, help = "Batch size when running prediction.")
    parser.add_argument("--device", type = str, default = "cuda", help = "The device to use for prediction.")
    parser.add_argument("--checkpoint", type = str, help = "Path to a model checkpoint")
    args = parser.parse_args()

    # Manually add in (hack)
    all_folders = []
    folder = "./results_ablations/"
    
    # for dataset in os.listdir(folder):
    #     dataset_folder = os.path.join(folder, dataset)
    #     for checkpoint in os.listdir(dataset_folder):
    #         all_folders.append(os.path.join(dataset_folder, checkpoint))

    # # Manually add in (hack)
    # # folder = "./results_w_sampling/massspecgym"
    # # folder = "./results_w_sampling/nist2023"
    # # folder = "./FT_results/massspecgym"
    # # folder = "./FT_results/nist2023"

    # all_folders = []    
    
    for model in os.listdir(folder):
        model_folder = os.path.join(folder, model)
        for checkpoint in os.listdir(model_folder):
            if "subsampled" in checkpoint: continue 
            all_folders.append(os.path.join(model_folder, checkpoint))

    for f in all_folders:

        args.checkpoint = f
        check_test_performance = os.path.exists(os.path.join(f, "test_performance.json"))
        check_test_results = os.path.exists(os.path.join(f, "test_results.pkl"))

        if check_test_performance:
            assert check_test_results
            continue 

        print("Running prediction for: ", f)
        main(args)
        print("Prediction complete")
    
