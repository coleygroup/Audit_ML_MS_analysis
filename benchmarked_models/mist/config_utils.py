import os
from pathlib import Path


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
    """Deprecated no-op kept for backwards compatibility.

    Split-file normalization used to be done by writing a rewritten TSV to a
    shared scratch directory. That approach could not work: the rewritten file
    was read back by MIST with dtype inference, which undid the string cast it
    existed to apply. It also derived the output name from the split file stem
    alone, so ``NPLIB1/splits/random.tsv`` and
    ``massspecgym/splits/random.tsv`` both mapped to ``random_normalized.tsv``
    in one directory and could silently cross-wire concurrent runs.

    Both concerns now live in
    :mod:`benchmarked_models.mist.utils.split_utils`, which reads the split
    file with ``dtype=str`` and renames columns in memory. Nothing is written
    to disk, so ``ML_MS_ANALYSIS_PREPARED_SPLITS`` is no longer consulted.
    """
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
