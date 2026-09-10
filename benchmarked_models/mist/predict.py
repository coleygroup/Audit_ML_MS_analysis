import os
import argparse
import numpy as np 
from tqdm import tqdm 

import torch 
import torch.nn.functional as F 

from utils import read_config, pickle_data, write_json
from utils import split_utils
from mist.data import datasets, featurizers
from config_utils import update_mist_config

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

def get_checkpoint_path(folder):

    if folder.endswith(".ckpt"):
        return folder

    checkpoints = [f for f in os.listdir(folder) if f.endswith(".ckpt")]
    if not checkpoints:
        raise FileNotFoundError(f"No .ckpt files found in {folder}")

    best_checkpoint, lowest_loss = "", float("inf")

    for c in checkpoints:
        # Monitor-based checkpoints are named "{epoch:03d}-{val_loss:.5f}.ckpt".
        # Names without a parseable trailing "=<float>" (notably "last.ckpt",
        # which is all the tuned configs emit under save_last_only) are not
        # rankable by loss and are handled by the fallback below.
        stem = c.replace("-v1", "").replace(".ckpt", "")
        try:
            loss = float(stem.split("=")[-1])
        except ValueError:
            continue
        if loss < lowest_loss:
            lowest_loss = loss
            best_checkpoint = c

    if not best_checkpoint:
        if "last.ckpt" in checkpoints:
            best_checkpoint = "last.ckpt"
        elif len(checkpoints) == 1:
            best_checkpoint = checkpoints[0]
        else:
            raise ValueError(
                f"Cannot choose a checkpoint in {folder}: none carry a parseable "
                f"monitor value and there is no last.ckpt. Found: {sorted(checkpoints)}. "
                f"Pass --checkpoint <path-to-.ckpt> explicitly."
            )

    return os.path.join(folder, best_checkpoint)

def get_loader(config, which = "test"):

    # Split data (repo-side splitter: matches split names as strings)
    my_splitter = split_utils.get_splitter(**config["dataset"])

    # Update the config now.
    # Test-time featurization is plain "peakformula" with the MAGMa auxiliary
    # target disabled. Upstream MIST exposes no "peakformula_test" key; that
    # name only existed in a local MIST fork, where it was a thin subclass
    # forcing magma_aux_loss=False. Setting the flag here keeps the same
    # behaviour without depending on a non-upstream featurizer registry entry.
    config["dataset"]["spec_features"] = "peakformula"
    config["dataset"]["magma_aux_loss"] = False
    config["dataset"]["allow_none_smiles"] = False

    # Get featurizers
    paired_featurizer = featurizers.get_paired_featurizer(**config["dataset"])

    # Build dataset
    spectra_mol_pairs = datasets.get_paired_spectra(**config["dataset"])
    spectra_mol_pairs = list(zip(*spectra_mol_pairs))

    # Get the requested split
    _, (_, val, test) = my_splitter.get_splits(spectra_mol_pairs)
    split_data = {"val": val, "test": test}[which]

    dataset = datasets.SpectraMolDataset(spectra_mol_list=split_data, featurizer=paired_featurizer, **config["train_settings"])
    loader = datasets.SpecDataModule.get_paired_loader(dataset, shuffle=False)

    return loader

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
    return update_mist_config(args, config)

@torch.no_grad()
def sweep_threshold(model, config, device, grid = np.arange(0.02, 0.99, 0.02)):

    """Pick the binarisation threshold that maximises Jaccard on the validation split."""

    all_pred, all_GT = [], []

    for spectra_batch in tqdm(get_loader(config, which = "val")):

        FP = spectra_batch["mols"][:, :].to(float)
        batch_to_device(spectra_batch, device)
        all_pred.append(model.encode_spectra(spectra_batch)[0].to(float).cpu())
        all_GT.append(FP)

    FP_pred = torch.cat(all_pred, dim = 0).numpy()
    FP = torch.cat(all_GT, dim = 0).numpy()

    best_threshold, best_jaccard = 0.5, -1.0
    for threshold in grid:
        jaccard = batch_jaccard_index((FP_pred > threshold).astype(int), FP).mean()
        if jaccard > best_jaccard:
            best_jaccard, best_threshold = float(jaccard), float(threshold)

    print(f"Validation-selected threshold: {best_threshold:.2f} (val jaccard {best_jaccard:.4f}, n={len(FP)})")
    return best_threshold, best_jaccard, len(FP)

@torch.no_grad()
def predict(model, config, device, threshold = 0.5):

    # Get the dataset
    test_loader = get_loader(config, which = "test")

    # Run model predictions
    id_list, predictions, GT, losses, jaccard_scores = [], [], [], [], []
    total_loss, total_jaccard, total_jaccard_half, total = 0,0,0,0

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
        jaccard_half = batch_jaccard_index(to_binary(FP_pred, 0.5), FP.numpy())

        total_loss += loss.mean(-1).item() * FP_pred.size(0)
        total_jaccard += jaccard.sum()
        total_jaccard_half += jaccard_half.sum()
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
    avg_jaccard = float(total_jaccard / total)
    avg_jaccard_half = float(total_jaccard_half / total)

    return predictions, float(avg_loss), avg_jaccard, avg_jaccard_half

def main(args):

    # Get the checkpoint and config
    checkpoint_dir = args.checkpoint
    if checkpoint_dir.endswith(".ckpt"):
        checkpoint_dir = os.path.dirname(checkpoint_dir)
    config = read_config(os.path.join(checkpoint_dir, "run.yaml"))
    config = update_config(args, config)

    # Get the model 
    model = MistNet.load_from_checkpoint(get_checkpoint_path(args.checkpoint))
    model.eval()
    model.to(args.device)

    # Get the predictions
    threshold, val_jaccard, n_val = sweep_threshold(model, config, args.device)
    predictions, loss, jaccard, jaccard_at_half = predict(model, config, args.device, threshold = threshold)

    # Write the predictions
    output_path = os.path.join(checkpoint_dir, "test_results.pkl")
    pickle_data(predictions, output_path)
    write_json({"loss": loss, "jaccard": jaccard, "threshold": threshold,
                "val_jaccard": val_jaccard, "n_val": n_val, "n_test": len(predictions),
                "jaccard_at_0.5": jaccard_at_half},
               os.path.join(checkpoint_dir, "test_performance.json"))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
 
    parser.add_argument("--batch_size", type = int, default = 512, help = "Batch size when running prediction.")
    parser.add_argument("--device", type = str, default = "cuda", help = "The device to use for prediction.")
    parser.add_argument("--checkpoint", type = str, help = "Path to a model checkpoint")
    args = parser.parse_args()
    if args.checkpoint is None:
        raise ValueError("--checkpoint must point to a checkpoint directory or .ckpt file")

    main(args)
