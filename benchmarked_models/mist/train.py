import os
import yaml
import random
import logging
import math
import argparse
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from mist.data import datasets, splitter, featurizers
from utils import read_config, load_pickle, pickle_data
from config_utils import get_mist_exp_name, update_mist_config

# Refine the mist model in our own directory 
from model import mist_model

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.warning")

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
def write_config_local(config, config_out_path):
    with open(config_out_path, "w") as f:
        yaml.dump(config, f)

def update_config(args, config):
    return update_mist_config(args, config)

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
    return get_mist_exp_name(config)

def train(config):

    # Set a random seed 
    seed_everything(config["seed"])

    # Update the results directory 
    results_dir = os.path.join(config["args"]["results_dir"], "mist")
    expt_name = get_exp_name(config)

    results_dir = os.path.join(results_dir, expt_name)
    create_results_dir(results_dir)

    # Write the config here
    config_o = read_config(os.path.join(config["args"]["config_dir"], config["args"]["config_file"]))
    config_o["exp_name"] = expt_name
    write_config_local(config_o, os.path.join(results_dir, "run.yaml"))

    # Get the datamodule 
    datamodule = get_datamodule(config)

    # Create model
    model = mist_model.MistNet(**config["model"]["params"])
    steps_per_epoch = math.ceil(len(datamodule.train) / datamodule.batch_size)
    model.total_training_steps = steps_per_epoch * config["trainer"]["max_epochs"]
    logging.info(
        "Training steps: %s/epoch * %s epochs = %s total steps",
        steps_per_epoch,
        config["trainer"]["max_epochs"],
        model.total_training_steps,
    )

    # Get trainer and logger
    callback_config = config.get("callbacks", {})
    monitor = callback_config.get("val_monitor", "val_loss")
    disable_validation = callback_config.get("disable_validation", False)
    save_last = callback_config.get("save_last", False)
    save_last_only = callback_config.get("save_last_only", False)

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
                                              filename = '{epoch:03d}-{val_loss:.5f}',
                                              every_n_train_steps = config["trainer"]["log_every_n_steps"],
                                              save_top_k = 2, mode = "min",
                                              save_last=save_last)
    
    callbacks = [checkpoint_callback]
    if not config["train_params"].get("cosine_schedule", False) and not disable_validation:
        callbacks.append(EarlyStopping(monitor=monitor, patience=config["callbacks"]["patience"]))
    if config["train_params"].get("ema", False):
        callbacks.append(EMACallback(decay=config["train_params"].get("ema_decay", 0.999)))

    trainer_kwargs = dict(config["trainer"])
    if disable_validation:
        trainer_kwargs["limit_val_batches"] = 0
        trainer_kwargs["num_sanity_val_steps"] = 0

    trainer = pl.Trainer(**trainer_kwargs, logger=False, callbacks=callbacks)

    # Start the training now
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=config["args"].get("resume_checkpoint"),
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type = str, default = "./all_configs", help = "Config directory")
    parser.add_argument("--config_file", type = str, default = "w_meta_config.yaml", help = "Config file")
    parser.add_argument("--torch_hub_cache", type = str, default = "./cache", help = "Torch hub cache directory")
    parser.add_argument("--results_dir", type = str, default = "./results", help = "Results output directory")
    parser.add_argument("--debug", action = "store_true", default = False, help = "Set debug mode")
    parser.add_argument("--disable_checkpoint", action = "store_true", default = False, help = "Disable checkpointing")
    parser.add_argument("--wandb", action = "store_true", default = False, help = "Deprecated; logging is disabled")
    parser.add_argument("--user", type = str, default = "serenakhoolm", help = "Set the user")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Resume training from this checkpoint")

    args = parser.parse_args()

    # Set torch hub cache
    torch.hub.set_dir(args.torch_hub_cache)

    # Read in and update the config
    config = read_config(os.path.join(args.config_dir, args.config_file))
    config = update_config(args, config)

    # Run the trainer now 
    train(config)
