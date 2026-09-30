import os
import argparse
import numpy as np 
from tqdm import tqdm 

import torch 
import torch.nn.functional as F 

from utils import read_config, pickle_data, write_json
from mist.data import datasets, splitter, featurizers

from model.mist_model import MistNet

def to_binary(FP, threshold):

    FP = FP.cpu().numpy()
    FP = (FP > threshold).astype(int)

    return FP 

@torch.no_grad()
def get_loss(FP_pred, FP):
    return F.binary_cross_entropy(FP_pred, FP, reduce = False)

@torch.no_grad()
def batch_jaccard_index(FP_pred, FP):

    # Intersection = bitwise AND
    intersection = np.logical_and(FP, FP_pred).sum(axis=1)

    # Union = bitwise OR
    union = np.logical_or(FP, FP_pred).sum(axis=1)

    # Avoid division-by-zero by adding a small epsilon
    jaccard_scores = intersection / (union + 1e-9)

    return jaccard_scores

def batch_cosine_sim(FP_pred, FP):
    FP_pred = np.asarray(FP_pred, dtype=np.float64)
    FP      = np.asarray(FP,      dtype=np.float64)
    num   = (FP_pred * FP).sum(axis=1)
    denom = np.linalg.norm(FP_pred, axis=1) * np.linalg.norm(FP, axis=1)
    return num / (denom + 1e-9)

def get_checkpoint_path(folder):

    checkpoints = [f for f in os.listdir(folder) if f.endswith(".ckpt")]
    best_checkpoint, lowest_loss = "", 1e4

    for c in checkpoints:

        loss = c.replace("-v1", "").replace(".ckpt", "").split("=")[-1] # hack 
        if loss == "last": continue
        loss = float(loss)
        if loss < lowest_loss:
            lowest_loss = loss 
            best_checkpoint = c 
    
    if best_checkpoint == "":
        best_checkpoint = [f for f in checkpoints if f.endswith("last.ckpt")][0]

    return os.path.join(folder, best_checkpoint)

def get_datamodule(config, split = "test"):

    # `split` selects which partition to score. Validation predictions are what `tune_threshold.py` needs to
    # pick a decision threshold without ever looking at the test split.
    assert split in ("val", "test"), split

    # Split data
    my_splitter = splitter.get_splitter(**config["dataset"])

    # Update the config now 
    config["dataset"]["spec_features"] = "peakformula_test"
    config["dataset"]["allow_none_smiles"] = False

    # Get featurizers
    paired_featurizer = featurizers.get_paired_featurizer(**config["dataset"])

    # Build dataset
    spectra_mol_pairs = datasets.get_paired_spectra(**config["dataset"])
    spectra_mol_pairs = list(zip(*spectra_mol_pairs))

    # Get the requested split 
    _, (_, val, test) = my_splitter.get_splits(spectra_mol_pairs)

    chosen = val if split == "val" else test
    chosen_dataset = datasets.SpectraMolDataset(spectra_mol_list=chosen, featurizer=paired_featurizer, **config["train_settings"])
    chosen_loader = datasets.SpecDataModule.get_paired_loader(chosen_dataset, shuffle=False)

    return chosen_loader

def batch_to_device(batch: dict, device) -> None:
    
    """batch_to_device.

    Convert batch tensors to same device as the model


    Args:
        batch (dict): Batch from data loader

    """
    # Port to cuda
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = batch[key].to(device)

def update_config(args, config):

    config["args"] = args.__dict__

    config["train_params"]["weight_decay"] = float(config["train_params"]["weight_decay"])
    config["model"]["params"]["fp_names"] = config["dataset"]["fp_names"] 
    config["model"]["params"]["magma_modulo"] = config["dataset"]["magma_modulo"]
    config["model"]["params"]["magma_aux_loss"] = config["dataset"]["magma_aux_loss"]

    config["model"]["params"]["learning_rate"] = config["train_params"]["learning_rate"] 
    config["model"]["params"]["weight_decay"] = config["train_params"]["weight_decay"]
    config["model"]["params"]["lr_decay_frac"] = config["train_params"]["lr_decay_frac"]
    config["model"]["params"]["scheduler"] = config["train_params"]["scheduler"]

    data_folder = config["dataset"]["data_folder"]
    dataset = config["dataset"]["dataset"]
    config["dataset"]["labels_file"] = os.path.join(data_folder, dataset, "labels.tsv")
    config["dataset"]["subform_folder"] = os.path.join(data_folder, dataset, "subformulae", "default_subformulae/")
    config["dataset"]["spec_folder"] = os.path.join(data_folder, dataset, "spec_folder")
    config["dataset"]["magma_folder"] = os.path.join(data_folder, dataset, "magma_outputs", "magma_tsv")
    config["dataset"]["split_file"] = os.path.join(data_folder, dataset, "splits", config["dataset"]["split_filename"])

    return config

@torch.no_grad()
def predict(model, config, device, threshold, split = "test"):

    # Get the dataset
    test_loader = get_datamodule(config, split = split)

    # Run model predictions
    id_list, predictions, GT, losses, jaccard_scores, cs_scores = [], [], [], [], [], []
    total_loss, total_jaccard, total_cs, total = 0, 0, 0, 0

    for spectra_batch in tqdm(test_loader):

        id_ = spectra_batch["names"]
        FP = spectra_batch["mols"][:, :].to(float)

        batch_to_device(spectra_batch, device)

        # Get the predicted fingerprints 
        FP_pred = model.encode_spectra(spectra_batch)[0].to(float).cpu()

        # Get the loss 
        loss = get_loss(FP_pred, FP)
        loss = loss.mean(-1)
        jaccard = batch_jaccard_index(to_binary(FP_pred, threshold), FP.numpy())
        cs = batch_cosine_sim(FP_pred.numpy(), FP.numpy())

        total_loss += loss.mean(-1).item() * FP_pred.size(0)
        total_jaccard += jaccard.sum()
        total_cs += cs.sum()
        total += FP_pred.size(0)

        # Save the predctions 
        id_list.extend(id_)
        predictions.append(FP_pred)
        GT.append(FP)
        losses.extend(loss.numpy().tolist())
        jaccard_scores.extend(jaccard.tolist())
        cs_scores.extend(cs.tolist())
        
    # Format the predictions
    predictions = torch.cat(predictions, dim = 0).numpy().tolist()
    GT = torch.cat(GT, dim = 0).numpy().tolist()
    predictions = {id_list[i]: {"pred": predictions[i], "GT": GT[i], "loss": losses[i], "jaccard": jaccard_scores[i]} for i in range(len(id_list))}
    
    # Get the average loss 
    avg_loss = total_loss / total 

    # Get the average jaccard loss 
    avg_jaccard = total_jaccard / total
    avg_cs = total_cs / total 

    return predictions, avg_loss, avg_jaccard, avg_cs

def main(args, threshold=0.5):

    # Get the checkpoint and config
    checkpoint_dir = args.checkpoint 
    config = read_config(os.path.join(checkpoint_dir, "run.yaml"))
    config = update_config(args, config)

    # Get the model 
    model = MistNet.load_from_checkpoint(get_checkpoint_path(checkpoint_dir))
    model.eval()
    model.to(args.device)

    # Get the predictions
    predictions, loss, jaccard, cs = predict(model, config, args.device, threshold=threshold, split=args.split)

    # Write the predictions. `threshold` only fixes the jaccard stored alongside each prediction; the raw
    # per-bit probabilities are kept, so tune_threshold.py can re-threshold without re-running the model.
    pickle_data(predictions, os.path.join(checkpoint_dir, f"{args.split}_results.pkl"))
    write_json({"loss": loss, "jaccard": jaccard, "cs": cs, "threshold": threshold},
               os.path.join(checkpoint_dir, f"{args.split}_performance.json"))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
 
    parser.add_argument("--batch_size", type = int, default = 512, help = "Batch size when running prediction.")
    parser.add_argument("--device", type = str, default = "cuda", help = "The device to use for prediction.")
    parser.add_argument("--checkpoint", type = str, help = "Path to a model checkpoint (a run folder)")
    parser.add_argument("--split", type = str, default = "test", choices = ["val", "test"],
                        help = "Which split to score. Use val to produce the predictions tune_threshold.py selects on.")
    parser.add_argument("--results_dir", type = str, default = "./results_previous/mist_seed_1/",
                        help = "Swept for run folders when --checkpoint is not given.")
    parser.add_argument("--filter", type = str, default = "",
                        help = "Only sweep run folders whose name contains this string.")
    parser.add_argument("--overwrite", action = "store_true", help = "Re-score runs that already have results.")
    args = parser.parse_args()

    if args.checkpoint:
        all_folders = [args.checkpoint]
    else:
        folder = args.results_dir
        all_folders = [os.path.join(folder, c) for c in os.listdir(folder) if args.filter in c]

    for f in all_folders:

        args.checkpoint = f
        done = (os.path.exists(os.path.join(f, f"{args.split}_performance.json"))
                and os.path.exists(os.path.join(f, f"{args.split}_results.pkl")))

        if done and not args.overwrite:
            print(f"Skipping (already has {args.split} predictions): ", f)
            continue 

        print(f"Running {args.split} prediction for: ", f)
        main(args, threshold=0.5)
        print("Prediction complete")