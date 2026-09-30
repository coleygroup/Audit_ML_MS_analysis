import os
import math
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

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.warning")

# EMA Callback
class EMACallback(pl.Callback):
    """Maintain EMA weights and swap them in for validation/checkpointing."""

    def __init__(self, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def on_fit_start(self, trainer, pl_module):
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        for name, param in pl_module.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def _swap_to_ema(self, pl_module):
        self.backup = {}
        for name, param in pl_module.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def _swap_from_ema(self, pl_module):
        for name, param in pl_module.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def on_validation_start(self, trainer, pl_module):
        if self.shadow:
            self._swap_to_ema(pl_module)

    def on_validation_end(self, trainer, pl_module):
        if self.backup:
            self._swap_from_ema(pl_module)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if not self.shadow:
            return
        state_dict = checkpoint.get("state_dict", {})
        for name, value in self.shadow.items():
            if name in state_dict:
                state_dict[name] = value.detach().clone()

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

    # New parameters for cosine schedule and warmup.
    config["model"]["params"]["cosine_schedule"] = config["train_params"].get("cosine_schedule", False)
    config["model"]["params"]["cosine_eta_min"] = config["train_params"].get("cosine_eta_min", 0.0)
    config["model"]["params"]["warmup_frac"] = config["train_params"].get("warmup_frac", 0.0)
    config["model"]["params"]["max_epochs"] = config["trainer"]["max_epochs"]

    data_folder = config["dataset"]["data_folder"]
    dataset = config["dataset"]["dataset"]
    config["dataset"]["labels_file"] = os.path.join(data_folder, dataset, "labels.tsv")
    config["dataset"]["subform_folder"] = os.path.join(data_folder, dataset, "subformulae", "default_subformulae")
    config["dataset"]["spec_folder"] = os.path.join(data_folder, dataset, "spec_folder")
    config["dataset"]["magma_folder"] = os.path.join(data_folder, dataset, "magma_outputs", "magma_tsv")
    config["dataset"]["split_file"] = os.path.join(data_folder, dataset, "splits", config["dataset"]["split_filename"])

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
    
    if config["dataset"].get("train_with_val", False):
        logging.info("Merging validation split into training split")
        train = train + val

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

def get_exp_name(config):

    dataset_code = ""

    if "canopus" in config["dataset"]["dataset"]: dataset_code = "C"
    elif "massspecgym" in config["dataset"]["dataset"]: dataset_code = "MSG"
    elif "nist2023" in config["dataset"]["dataset"]: dataset_code = "NIST2023"
    else: raise Exception("Dataset not recognized - ", config["dataset"]["dataset"])

    model_code = "MIST"
    split_code = config["dataset"]["split_file"].split("/")[-1].replace(".tsv", "")

    if "w_meta" in config["args"]["config_file"]:
        name = f"{dataset_code}_{model_code}_meta_4096_{split_code}"
    elif "wo_meta" in config["args"]["config_file"]: 
        name = f"{dataset_code}_{model_code}_4096_{split_code}"

    elif "sieved" in config["args"]["config_file"]: 
        name = f"{dataset_code}_{model_code}_sieved_4096_{split_code}"
    elif "original_longer" in config["args"]["config_file"]:
        name = f"{dataset_code}_{model_code}_original_longer_4096_{split_code}"
    elif "original" in config["args"]["config_file"]:
        name = f"{dataset_code}_{model_code}_original_4096_original_{split_code}"
    elif "scaffold_ours" in config["args"]["config_file"]:
        name = f"{dataset_code}_{model_code}_scaffold_ours_4096_{split_code}"
    elif "scaffold_tuned_w_val" in config["args"]["config_file"]:
        name = f"{dataset_code}_{model_code}_scaffold_tuned_w_val_4096_{split_code}"
    elif "scaffold_tuned" in config["args"]["config_file"]:
        name = f"{dataset_code}_{model_code}_scaffold_tuned_wo_val_4096_{split_code}"
    elif "tuned" in config["args"]["config_file"]:
        name = f"{dataset_code}_{model_code}_tuned_4096_{split_code}"
    else:
        raise NotImplementedError() 
    
    return name 

def train(config):

    # Set a random seed 
    seed_everything(config["seed"])

    # Update the results directory 
    results_dir = os.path.join(config["args"]["results_dir"], "mist")
    expt_name = get_exp_name(config)

    results_dir = os.path.join(results_dir, expt_name)
    create_results_dir(results_dir)

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
    model = mist_model.MistNet(**config["model"]["params"])

    # Updated training steps calculation to account for the number of batches per epoch and the total number of epochs
    steps_per_epoch = math.ceil(len(datamodule.train) / datamodule.batch_size)
    model.total_training_steps = steps_per_epoch * config["trainer"]["max_epochs"]
    logging.info(
        "Training steps: %s/epoch * %s epochs = %s total steps",
        steps_per_epoch,
        config["trainer"]["max_epochs"],
        model.total_training_steps,
    )

    # Get trainer and logger
    monitor = config["callbacks"]["val_monitor"]
    # Updated the checkpointing logic to allow for disabling validation and saving the last checkpoint only
    disable_validation = config["callbacks"].get("disable_validation", False)
    save_last = config["callbacks"].get("save_last", False)
    save_last_only = config["callbacks"].get("save_last_only", False)

    if save_last_only:
        checkpoint_callback = ModelCheckpoint(
            dirpath=results_dir,
            filename="{epoch:03d}",
            save_last=True,
            save_top_k=0,
            every_n_epochs=1,
        )
    
    else:
        checkpoint_callback = ModelCheckpoint(monitor=monitor,
                                              dirpath = results_dir,
                                              filename = '{epoch:03d}-{val_loss:.5f}', # Hack 
                                              every_n_train_steps = config["trainer"]["log_every_n_steps"], 
                                              save_top_k = 2, mode = "min")

    # Updated the callbacks logic to include EarlyStopping and EMACallback based on the configuration
    callbacks = [checkpoint_callback]
    if not config["train_params"].get("cosine_schedule", False) and not disable_validation:
        callbacks.append(EarlyStopping(monitor=monitor, patience=config["callbacks"]["patience"]))
    if config["train_params"].get("ema", False):
        callbacks.append(EMACallback(decay=config["train_params"].get("ema_decay", 0.999)))

    trainer_kwargs = dict(config["trainer"])
    if disable_validation:
        trainer_kwargs["limit_val_batches"] = 0
        trainer_kwargs["num_sanity_val_steps"] = 0

    trainer = pl.Trainer(**config["trainer"], logger = wandb_logger, callbacks=callbacks)

    # Start the training now
    trainer.fit(model, datamodule = datamodule)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type = str, default = "./all_configs", help = "Config directory")
    parser.add_argument("--config_file", type = str, default = "w_meta_config.yaml", help = "Config file")
    parser.add_argument("--torch_hub_cache", type = str, default = "./cache", help = "Torch hub cache directory")
    parser.add_argument("--results_dir", type = str, default = "./results_longer", help = "Results output directory")
    parser.add_argument("--debug", action = "store_true", default = False, help = "Set debug mode")
    parser.add_argument("--disable_checkpoint", action = "store_true", default = False, help = "Disable checkpointing")
    parser.add_argument("--wandb", action = "store_true", default = True, help = "Enable wandb logging")
    parser.add_argument("--user", type = str, default = None, help = "wandb entity (default: your wandb default entity)")

    args = parser.parse_args()

    # Set torch hub cache
    torch.hub.set_dir(args.torch_hub_cache)

    # Read in and update the config
    config = read_config(os.path.join(args.config_dir, args.config_file))
    config = update_config(args, config)

    # Run the trainer now 
    train(config)
