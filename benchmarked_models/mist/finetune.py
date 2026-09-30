import os
import copy
import yaml
import random
import logging
import argparse
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from mist.data import datasets, splitter, featurizers
from utils import read_config, load_pickle, pickle_data

# Refine the mist model in our own directory 
from model import mist_model

# Weights & Biases: export WANDB_API_KEY in your shell (or run `wandb login`) before training.

def update_config(args, config):

    config["args"] = args.__dict__

    config["model"]["params"] = {} 
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

    # Get the split file
    sampling_strategy = config["args"]["sampling_strategy"]
    sampling_ratio = config["args"]["ratio"]
    split_folder = os.path.join(data_folder, dataset, "splits_sampling", sampling_strategy)

    if sampling_strategy in ["IF_val", "IF_test"]:  # Change the split file if its based on IF
        split_file = os.path.join(split_folder, f"sampled_{sampling_strategy}_MIST_{sampling_ratio}.tsv")
        
    elif sampling_strategy == "CF":
        split_file = os.path.join(split_folder, f"sampled_{sampling_strategy}_MIST_{sampling_ratio}.tsv")

    else:
        split_file = os.path.join(split_folder, f"sampled_{sampling_strategy}_{sampling_ratio}.tsv")
    
    assert os.path.exists(split_file)
    config["dataset"]["split_file"] = split_file

    return config

def get_datamodule(config):

    # Split data
    my_splitter = splitter.get_splitter(**config["dataset"])

    # Get model class
    model_class = mist_model.MistNet
    config["model"]["name"] = model_class.__name__

    # Get featurizers
    paired_featurizer = featurizers.get_paired_featurizer(**config["dataset"])

    # Build dataset
    spectra_mol_pairs = datasets.get_paired_spectra(**config["dataset"])
    spectra_mol_pairs = list(zip(*spectra_mol_pairs))

    # Redefine splitter s.t. this splits three times and remove subsetting
    split_name, (train, val, test) = my_splitter.get_splits(spectra_mol_pairs)

    for name, _data in zip(["train", "val", "test"], [train, val, test]):
        logging.info(f"Split: {split_name}, Len of {name}: {len(_data)}")
    
    train_dataset = datasets.SpectraMolDataset(
        spectra_mol_list=train, featurizer=paired_featurizer, **config["train_settings"]
    )
    val_dataset = datasets.SpectraMolDataset(
        spectra_mol_list=val, featurizer=paired_featurizer, **config["train_settings"]
    )
    test_dataset = datasets.SpectraMolDataset(
        spectra_mol_list=test, featurizer=paired_featurizer, **config["train_settings"]
    )
    spec_dataloader_module = datasets.SpecDataModule(
        train_dataset, val_dataset, test_dataset, **config["train_settings"]
    ) # Note: this is already a pytorch lightning data module 
    
    return spec_dataloader_module

@rank_zero_only
def create_results_dir(results_dir):
    if not os.path.exists(results_dir): os.makedirs(results_dir)

@rank_zero_only
def write_config(wandb_logger, config):

    # Dump raw config now
    run_out_dir = wandb_logger.experiment.dir
    config_out_path = os.path.join(run_out_dir, "run.yaml")
    with open(config_out_path, "w") as f:
        yaml.dump(config, f)
    wandb_logger.experiment.save("run.yaml", policy = "now")

@rank_zero_only
def write_config_local(config, config_out_path):
    with open(config_out_path, "w") as f:
        yaml.dump(config, f)

def get_checkpoint_file(results_cache, dataset):

    results_cache = os.path.join(results_cache, f"{dataset}_sieved")
    checkpoint_folder = [f for f in os.listdir(results_cache) if "scaffold_vanilla" in f] # Hack
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

def finetune(config):

    # Set a random seed 
    seed_everything(config["seed"])

    # Update the results directory 
    results_dir = os.path.join(args.results_dir, config["dataset"]["dataset"], args.sampling_strategy)
    sampling_ratio = args.ratio
    expt_name = f"MIST_{sampling_ratio}"
    results_dir = os.path.join(results_dir, expt_name)
    create_results_dir(results_dir)

    # Get the checkpoint to finetune from
    results_cache = config["model"]["results_dir"]
    _, checkpoint = get_checkpoint_file(results_cache, config["dataset"]["dataset"])

    # Get the wandb logger 
    wandb_logger = None 
    if not config["args"]["debug"] and config["args"]["wandb"]:
        wandb_logger = WandbLogger(project = config["project"],
                                   config = config,
                                   group = config["args"]["config_file"].replace(".yaml", ""),
                                   entity = config["args"]["user"],
                                   name = expt_name,
                                   log_model = False)
        
        # Dump config
        raw_config = copy.deepcopy(config)
        del raw_config["args"]
        write_config(wandb_logger, raw_config)
    
    # Write the config here
    config_o = read_config(os.path.join(config["args"]["config_dir"], config["args"]["config_file"]))
    config_o["exp_name"] = expt_name
    write_config_local(config_o, os.path.join(results_dir, "run.yaml"))

    # Get the datamodule 
    datamodule = get_datamodule(config)

    # Create model
    model = mist_model.MistNet.load_from_checkpoint(checkpoint)

    # Get trainer and logger
    # config["trainer"]["check_val_every_n_epoch"] = None
    monitor = config["callbacks"]["val_monitor"]
    checkpoint_callback = ModelCheckpoint(monitor=monitor,
                                          dirpath = results_dir,
                                          filename = '{epoch:03d}-{val_bce_loss:.5f}', # Hack 
                                          every_n_train_steps = config["trainer"]["log_every_n_steps"], 
                                          save_top_k = 2, mode = "min")
    
    earlystop_callback = EarlyStopping(monitor=monitor, patience=config["callbacks"]["patience"])
    trainer = pl.Trainer(**config["trainer"], logger = wandb_logger, callbacks=[checkpoint_callback, earlystop_callback])

    # Start the training now
    trainer.fit(model, datamodule = datamodule)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type = str, default = "./all_configs", help = "Config directory")
    parser.add_argument("--config_file", type = str, default = "FT_config.yaml", help = "Config file")
    parser.add_argument("--sampling_strategy", type = str, default = "random", help = "Sampling strategy")
    parser.add_argument("--ratio", type = int, default = 5, help = "Amount of additional training data")
    parser.add_argument("--torch_hub_cache", type = str, default = "./cache", help = "Torch hub cache directory")
    parser.add_argument("--results_dir", type = str, default = "./FT_results", help = "Results output directory")
    parser.add_argument("--debug", action = "store_true", default = False, help = "Set debug mode")
    parser.add_argument("--disable_checkpoint", action = "store_true", default = False, help = "Disable checkpointing")
    parser.add_argument("--wandb", action = "store_true", default = True, help = "Enable wandb logging")
    parser.add_argument("--user", type = str, default = None, help = "wandb entity (default: your wandb default entity)")

    args = parser.parse_args()

    # Read in and update the config
    config = read_config(os.path.join(args.config_dir, args.config_file))
    config = update_config(args, config)

    # Finetune the model now 
    finetune(config)