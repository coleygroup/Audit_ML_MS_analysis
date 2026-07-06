import os
from pathlib import Path

import pandas as pd


def _maybe_set(config_section, key, value):
    if key not in config_section or config_section[key] in (None, ""):
        config_section[key] = value


def _expand_config_paths(dataset_config):
    path_keys = [
        "data_folder",
        "labels_file",
        "subform_folder",
        "spec_folder",
        "magma_folder",
        "spec_hdf5",
        "subform_hdf5",
        "magma_hdf5",
        "split_file",
        "prepared_split_dir",
        "forward_labels",
        "forward_aug_folder",
    ]
    for key in path_keys:
        if key in dataset_config and isinstance(dataset_config[key], str):
            dataset_config[key] = os.path.expanduser(os.path.expandvars(dataset_config[key]))


def normalize_split_if_needed(dataset_config, prepared_root=None):
    split_file = dataset_config.get("split_file")
    if not split_file:
        return dataset_config

    split_path = Path(split_file)
    if split_path.suffix.lower() != ".tsv" or not split_path.exists():
        return dataset_config

    split_df = pd.read_csv(split_path, sep="\t", nrows=1)
    name_col = dataset_config.get("split_name_col")
    split_col = dataset_config.get("split_value_col")
    if name_col is None:
        name_col = "name" if "name" in split_df.columns else "spec"
    if split_col is None:
        split_col = "split" if "split" in split_df.columns else "Fold_0"

    if name_col == "name" and split_col == "split":
        return dataset_config

    if prepared_root is None:
        prepared_root = Path(os.environ.get("ML_MS_ANALYSIS_PREPARED_SPLITS", "/tmp/ml_ms_analysis_mist_splits"))
    prepared_root = Path(prepared_root)
    prepared_root.mkdir(parents=True, exist_ok=True)

    full_split_df = pd.read_csv(split_path, sep="\t")
    normalized = full_split_df[[name_col, split_col]].rename(
        columns={name_col: "name", split_col: "split"}
    )
    out_file = prepared_root / f"{split_path.stem}_normalized.tsv"
    normalized.to_csv(out_file, sep="\t", index=False)
    dataset_config["split_file_original"] = str(split_path)
    dataset_config["split_file"] = str(out_file)
    dataset_config["split_name_col"] = "name"
    dataset_config["split_value_col"] = "split"
    return dataset_config


def update_mist_config(args, config):
    config["args"] = args.__dict__

    dataset_config = config["dataset"]
    _expand_config_paths(dataset_config)
    train_params = config["train_params"]
    model_params = config["model"]["params"]

    train_params["weight_decay"] = float(train_params["weight_decay"])
    model_params["fp_names"] = dataset_config["fp_names"]
    model_params["magma_modulo"] = dataset_config["magma_modulo"]
    model_params["magma_aux_loss"] = dataset_config["magma_aux_loss"]
    model_params["learning_rate"] = train_params["learning_rate"]
    model_params["weight_decay"] = train_params["weight_decay"]
    model_params["lr_decay_frac"] = train_params["lr_decay_frac"]
    model_params["scheduler"] = train_params["scheduler"]
    model_params["cosine_schedule"] = train_params.get("cosine_schedule", False)
    model_params["cosine_eta_min"] = train_params.get("cosine_eta_min", 0.0)
    model_params["warmup_frac"] = train_params.get("warmup_frac", 0.0)
    model_params["max_epochs"] = config["trainer"]["max_epochs"]
    if hasattr(args, "batch_size") and args.batch_size is not None:
        config["train_settings"]["batch_size"] = args.batch_size

    data_folder = dataset_config.get("data_folder")
    dataset = dataset_config.get("dataset")
    if data_folder and dataset:
        dataset_root = Path(data_folder) / dataset
        _maybe_set(dataset_config, "labels_file", str(dataset_root / "labels.tsv"))
        if not dataset_config.get("subform_hdf5"):
            _maybe_set(
                dataset_config,
                "subform_folder",
                str(dataset_root / "subformulae" / "default_subformulae"),
            )
        if not dataset_config.get("spec_hdf5"):
            _maybe_set(dataset_config, "spec_folder", str(dataset_root / "spec_folder"))
        if not dataset_config.get("magma_hdf5"):
            _maybe_set(dataset_config, "magma_folder", str(dataset_root / "magma_outputs" / "magma_tsv"))
        if "split_filename" in dataset_config:
            _maybe_set(
                dataset_config,
                "split_file",
                str(dataset_root / "splits" / dataset_config["split_filename"]),
            )

    prepared_root = dataset_config.get("prepared_split_dir")
    dataset_config = normalize_split_if_needed(dataset_config, prepared_root)
    return config


def get_mist_exp_name(config):
    if config.get("exp_name"):
        return config["exp_name"]

    dataset_name = config["dataset"]["dataset"]
    if "canopus" in dataset_name:
        dataset_code = "C"
    elif "massspecgym" in dataset_name:
        dataset_code = "MSG"
    elif "nist2023" in dataset_name:
        dataset_code = "NIST2023"
    else:
        raise ValueError(f"Dataset not recognized: {dataset_name}")

    split_file = config["dataset"].get("split_file") or config["dataset"].get("split_filename", "split")
    split_code = Path(split_file).name.replace(".tsv", "")
    model_code = "MIST"
    config_file = config["args"].get("config_file", "")
    if "w_meta" in config_file:
        suffix = "meta_4096"
    elif "sieved" in config_file:
        suffix = "sieved_4096"
    else:
        suffix = "4096"
    return f"{dataset_code}_{model_code}_{suffix}_{split_code}"
