#!/usr/bin/env python3
"""Prepare Google Drive MIST data and runtime configs for MSG/NPLIB1 reruns."""

import argparse
import csv
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pandas as pd
import yaml


DATASETS = {
    "massspecgym": {
        "label": "MassSpecGym",
        "exp_prefix": "MSG_MIST_4096",
        "base_config": "msg_random_mist_config.yaml",
        "files": {
            "labels.tsv": "1slWczJpu-caPRzEoJFOZeC7C8HqL05Us",
            "spec_files.tar.gz": "1DoZvLSgodBdrT9lJyx36csXWpksgX4Dm",
            "subformulae.tar.gz": "18AvuHbGkNwB3A96e4VlDqbDNCjl2lHGS",
            "magma_outputs.tar.gz": "1itQYk8CYChKWRRhoaf2yNxtVv62lFqsc",
        },
        "splits": {
            "random": "1AEnCVmkEODhFpmWb2qOfTNPdwoMSAnZ_",
            "scaffold": "1nhSOOyX9S4k93CEesSg_W2VPJYimyLO2",
        },
        "original_base_config": "msg_random_original_mist_config.yaml",
        "original_exp_prefix": "MSG_ORIGINAL_MIST_4096",
    },
    "NPLIB1": {
        "label": "NPLIB1",
        "exp_prefix": "NPLIB1_MIST_4096",
        "base_config": "nplib1_random_mist_config.yaml",
        "files": {
            "labels.tsv": "1IVFWgaOgv1amRr5HGzeQKa7Pp3smMTDm",
            "spec_files.tar.gz": "1e0ZMgZPxsznbSYSM9PZUobZ_maKN6rGU",
            "subformulae.tar.gz": "1DTcbFTDJ5XFHu3JpEgIlkHIBitpJSkhu",
            "magma_outputs.tar.gz": "1Ik0aUWHXic_cy58WZJGIKKBsQJLZuEAw",
        },
        "splits": {
            "random": "1gniU09a5D_y3kfY03WQa2qDLsSd9tbpL",
            "scaffold": "1jr3wDlToTLYHb1n9Vm-iR4Jih0SBFHnW",
        },
        "original_base_config": "nplib1_random_original_mist_config.yaml",
        "original_exp_prefix": "NPLIB1_ORIGINAL_MIST_4096",
    },
}


def run(cmd, cwd=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, cwd=cwd)


def looks_like_html(path):
    if not path.exists() or path.stat().st_size == 0:
        return True
    with path.open("rb") as handle:
        head = handle.read(512).lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html")


def download_drive_file(file_id, out_file, cookie_file, force=False):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists() and out_file.stat().st_size > 0 and not force and not looks_like_html(out_file):
        return

    tmp_file = out_file.with_suffix(out_file.suffix + ".part")
    if tmp_file.exists():
        tmp_file.unlink()

    urls = [
        f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t",
        f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t",
    ]
    last_error = None
    for url in urls:
        try:
            run(
                [
                    "curl",
                    "-L",
                    "--fail",
                    "--retry",
                    "5",
                    "--retry-delay",
                    "5",
                    "--cookie",
                    cookie_file,
                    "--cookie-jar",
                    cookie_file,
                    "--output",
                    tmp_file,
                    url,
                ]
            )
            if not looks_like_html(tmp_file):
                tmp_file.replace(out_file)
                return
        except subprocess.CalledProcessError as exc:
            last_error = exc
        if tmp_file.exists():
            tmp_file.unlink()
    raise RuntimeError(f"Failed to download Drive file {file_id} to {out_file}") from last_error


def extract_tarball(tarball, dataset_root):
    marker = dataset_root / f".extracted_{tarball.name}"
    if marker.exists():
        return
    with tarfile.open(tarball, "r:gz") as tar:
        tar.extractall(dataset_root)
    marker.write_text("ok\n")


def first_existing_dir(dataset_root, names):
    for name in names:
        path = dataset_root / name
        if path.exists():
            return path
    return None


def find_spec_folder(dataset_root):
    return first_existing_dir(dataset_root, ["spec_files", "spec_folder", "specs"])


def find_subform_folder(dataset_root):
    candidates = [
        dataset_root / "subformulae" / "default_subformulae",
        dataset_root / "subformulae" / "subformulae_default",
        dataset_root / "default_subformulae",
        dataset_root / "subformulae",
    ]
    for path in candidates:
        if path.exists() and any(path.glob("*.json")):
            return path
    for path in dataset_root.rglob("*"):
        if path.is_dir() and path.name in {"default_subformulae", "subformulae_default"}:
            return path
    return None


def find_magma_folder(dataset_root):
    candidates = [
        dataset_root / "magma_outputs" / "magma_tsv",
        dataset_root / "magma_outputs",
        dataset_root / "magma_tsv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def split_counts(split_file):
    counts = {}
    with split_file.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            counts[row["split"]] = counts.get(row["split"], 0) + 1
    return counts


def validate_split(labels_file, split_file):
    labels = pd.read_csv(labels_file, sep="\t", dtype=str)
    splits = pd.read_csv(split_file, sep="\t", dtype=str)
    label_ids = set(labels["spec"].astype(str))
    split_ids = set(splits["name"].astype(str))
    split_sets = {
        split: set(splits.loc[splits["split"].eq(split), "name"].astype(str))
        for split in ["train", "val", "test"]
    }
    return {
        "labels": int(len(labels)),
        "unique_labels": int(len(label_ids)),
        "split_rows": int(len(splits)),
        "unique_split_ids": int(len(split_ids)),
        "split_counts": {k: int(v) for k, v in splits["split"].value_counts().to_dict().items()},
        "missing_split_ids_in_labels": int(len(split_ids - label_ids)),
        "labels_not_in_split": int(len(label_ids - split_ids)),
        "train_val_overlap": int(len(split_sets["train"] & split_sets["val"])),
        "train_test_overlap": int(len(split_sets["train"] & split_sets["test"])),
        "val_test_overlap": int(len(split_sets["val"] & split_sets["test"])),
    }


def write_config(
    repo_root,
    run_root,
    dataset_key,
    split_name,
    dataset_root,
    config_dir,
    base_config_name=None,
    exp_prefix=None,
    output_suffix="mist_config",
):
    spec = DATASETS[dataset_key]
    base_config = (
        repo_root
        / "benchmarked_models"
        / "mist"
        / "all_configs"
        / (base_config_name or spec["base_config"])
    )
    config = yaml.safe_load(base_config.read_text())
    labels_file = dataset_root / "labels.tsv"
    split_file = dataset_root / "splits" / f"{split_name}.tsv"
    spec_folder = find_spec_folder(dataset_root)
    subform_folder = find_subform_folder(dataset_root)
    magma_folder = find_magma_folder(dataset_root)
    if spec_folder is None:
        raise FileNotFoundError(f"Could not locate extracted spec folder under {dataset_root}")
    if subform_folder is None:
        raise FileNotFoundError(f"Could not locate extracted subformula JSON folder under {dataset_root}")

    config["exp_name"] = f"{exp_prefix or spec['exp_prefix']}_{split_name}"
    config["dataset"]["dataset"] = dataset_key
    config["dataset"]["labels_file"] = str(labels_file)
    config["dataset"]["spec_folder"] = str(spec_folder)
    config["dataset"]["subform_folder"] = str(subform_folder)
    config["dataset"]["split_file"] = str(split_file)
    config["dataset"]["storage_mode"] = "google_drive_mist_outputs"
    config["dataset"].pop("data_folder", None)
    config["dataset"].pop("split_filename", None)
    config["dataset"].pop("spec_hdf5", None)
    config["dataset"].pop("subform_hdf5", None)
    config["dataset"].pop("magma_hdf5", None)
    if magma_folder is not None:
        config["dataset"]["magma_folder"] = str(magma_folder)
    config["train_settings"]["persistent_workers"] = bool(config["train_settings"].get("num_workers", 0))

    out_file = config_dir / f"{dataset_key.lower()}_{split_name}_{output_suffix}.yaml"
    out_file.write_text(yaml.safe_dump(config, sort_keys=False))
    return out_file


def prepare_dataset(args, dataset_key):
    spec = DATASETS[dataset_key]
    dataset_root = args.data_root / dataset_key
    splits_dir = dataset_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    cookie_file = args.data_root / ".gdrive_cookies.txt"

    for name, file_id in spec["files"].items():
        out_file = dataset_root / name
        download_drive_file(file_id, out_file, cookie_file, force=args.force_download)
        if name.endswith(".tar.gz"):
            extract_tarball(out_file, dataset_root)

    for split_name, file_id in spec["splits"].items():
        download_drive_file(
            file_id,
            splits_dir / f"{split_name}.tsv",
            cookie_file,
            force=args.force_download,
        )

    config_files = []
    split_reports = {}
    for split_name in spec["splits"]:
        split_file = splits_dir / f"{split_name}.tsv"
        split_reports[split_name] = validate_split(dataset_root / "labels.tsv", split_file)
        config_files.append(
            write_config(args.repo_root, args.run_root, dataset_key, split_name, dataset_root, args.config_dir)
        )
        config_files.append(
            write_config(
                args.repo_root,
                args.run_root,
                dataset_key,
                split_name,
                dataset_root,
                args.config_dir,
                base_config_name=spec["original_base_config"],
                exp_prefix=spec["original_exp_prefix"],
                output_suffix="original_mist_config",
            )
        )

    return {
        "dataset": dataset_key,
        "root": str(dataset_root),
        "labels_file": str(dataset_root / "labels.tsv"),
        "spec_folder": str(find_spec_folder(dataset_root)),
        "subform_folder": str(find_subform_folder(dataset_root)),
        "magma_folder": str(find_magma_folder(dataset_root)),
        "splits": split_reports,
        "configs": [str(path) for path in config_files],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("/home/runzhong/mist_repro_20260629"))
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    args.repo_root = args.repo_root or (args.run_root / "code" / "ML_MS_analysis")
    args.data_root = args.data_root or (args.run_root / "raw" / "google_drive_mist_outputs")
    args.config_dir = args.config_dir or (args.run_root / "manifests" / "runtime_configs")
    args.manifest = args.manifest or (args.run_root / "manifests" / "google_drive_mist_outputs_manifest.json")
    args.data_root.mkdir(parents=True, exist_ok=True)
    args.config_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_root": str(args.run_root),
        "data_root": str(args.data_root),
        "storage_mode": "google_drive_mist_outputs",
        "datasets": {},
    }
    for dataset_key in ["massspecgym", "NPLIB1"]:
        manifest["datasets"][dataset_key] = prepare_dataset(args, dataset_key)

    args.manifest.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {args.manifest}")


if __name__ == "__main__":
    main()
