import os
import yaml
import copy
import math
import argparse
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from dataloader import MSDataset
from utils import read_config, load_pickle
from modules import MSBinnedModel, MSTransformerEncoder, FormulaTransformerEncoder

# Weights & Biases: export WANDB_API_KEY in your shell (or run `wandb login`) before training.

def update_config(args, config):
    
    config["args"] = args.__dict__
    devices = config["trainer"].get("devices", 1)

    if config["args"]["debug"]:
        torch.autograd.set_detect_anomaly(True)
        devices = 1
        config["trainer"].update(devices=devices)
        config["data"]["num_workers"] = 0

    config["trainer"]["val_check_interval"] = config["trainer"]["log_every_n_steps"] - 1

    if devices > 1:
        config.setdefault("trainer", {}).update(strategy = DDPStrategy(find_unused_parameters=False))

    if args.disable_checkpoint:
        config["trainer"]["enable_checkpointing"] = False

    # Update the data directory 
    config["data"]["dir"] = os.path.join(config["data"]["data_folder"], config["data"]["dataset"], "frags_preds")

    sampling_strategy = config["args"]["sampling_strategy"]
    sampling_ratio = config["args"]["ratio"]

    if sampling_strategy in ["IF_val", "IF_test"]:  # Change the split file if its based on IF

        model_name = config["model"]["name"]
        model_mapping = {"binned_MS_encoder" : "binned", "MS_encoder": "MS", "formula_encoder": "formula"}
        model_name = model_mapping[model_name]

        split_file = os.path.join(config["data"]["splits_folder"], config["data"]["dataset"], 
                                                    "splits_sampling", sampling_strategy, 
                                                    f"sampled_{sampling_strategy}_{model_name}_{sampling_ratio}.json")
    else:
        split_file = os.path.join(config["data"]["splits_folder"], config["data"]["dataset"], 
                                                    "splits_sampling", sampling_strategy, 
                                                    f"sampled_{sampling_strategy}_{sampling_ratio}.json")
    
    assert os.path.exists(split_file)
    config["data"]["split_file"] = split_file

    config["data"]["adduct_file"] = os.path.join(config["data"]["data_folder"], config["data"]["dataset"], "all_adducts.pkl")
    config["data"]["instrument_file"] = os.path.join(config["data"]["data_folder"], config["data"]["dataset"], "all_instruments.pkl")
    del config["data"]["data_folder"]
    del config["data"]["splits_folder"]

    # Update getting the CF, fragments for each individual model 
    if config["model"]["name"] == "formula_encoder": config["data"]["get_CF"] = True 

    return config

def get_checkpoint_file(results_cache, dataset, model):
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

@rank_zero_only
def create_results_dir(results_dir):
    if not os.path.exists(results_dir): os.makedirs(results_dir)

def finetune(config):  

    # Set a random seed 
    seed_everything(config["seed"])

    # Update the results directory 
    results_dir = os.path.join(args.results_dir, config["data"]["dataset"], args.sampling_strategy)
    dataset_name = config["data"]["dataset"]
    model = config["model"]["name"]
    sampling_ratio = args.ratio
    expt_name = f"{model}_{sampling_ratio}"
    results_dir = os.path.join(results_dir, expt_name)
    create_results_dir(results_dir)

    # Get the checkpoint to finetune from
    results_cache = config["model"]["results_dir"]
    checkpoint_dir, checkpoint = get_checkpoint_file(results_cache, dataset_name, model)
    params = read_config(os.path.join(checkpoint_dir, "run.yaml"))

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

    # Set output folder
    config["trainer"].update(default_root_dir = results_dir)

    # Get dataset (hack to update)
    del config["data"]["dataset"]
    config["data"]["FP_type"] = params["data"]["FP_type"]
    config["data"]["bin_resolution"] = params["data"]["bin_resolution"]
    config["data"]["considered_atoms"] = params["data"]["considered_atoms"]
    config["data"]["intensity_threshold"] = params["data"]["intensity_threshold"]
    config["data"]["intensity_type"] = params["data"]["intensity_type"]
    config["data"]["max_MS_peaks"] = params["data"]["max_MS_peaks"]
    config["data"]["max_da"] = params["data"]["max_da"]

    dataset = MSDataset(**config["data"])
    dataset.prepare_data()
    dataset.setup()

    # Get trainer and logger
    config["trainer"]["check_val_every_n_epoch"] = None
    monitor = config["callbacks"]["monitor"]
    checkpoint_callback = ModelCheckpoint(monitor=monitor,
                                          dirpath = results_dir,
                                          filename = '{epoch:03d}-{val_FP_loss:.5f}', # Hack
                                          every_n_train_steps = config["trainer"]["log_every_n_steps"], 
                                          save_top_k = 2, mode = "min")
    earlystop_callback = EarlyStopping(monitor=monitor, patience=config["callbacks"]["patience"])
    trainer = pl.Trainer(**config["trainer"], logger = wandb_logger, callbacks=[checkpoint_callback, earlystop_callback])

    # Get the model
    model_name = config["model"]["name"]
    if model_name == "binned_MS_encoder":
        model = MSBinnedModel.load_from_checkpoint(checkpoint)

    elif model_name == "MS_encoder":
        model = MSTransformerEncoder.load_from_checkpoint(checkpoint)

    elif model_name == "formula_encoder":
        model = FormulaTransformerEncoder.load_from_checkpoint(checkpoint)

    else:
        raise Exception(f"{model_name} not supported.")

    # Resume training 
    trainer.fit(model, datamodule = dataset) 

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

    # Set torch hub cache
    torch.hub.set_dir(args.torch_hub_cache)

    # Read in the config
    config = read_config(os.path.join(args.config_dir, args.config_file))
    config = update_config(args, config)

    # Run the trainer now 
    finetune(config)


