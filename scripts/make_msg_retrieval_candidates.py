#!/usr/bin/env python3
"""Build MassSpecGym retrieval candidate TSVs from official JSON files.

The MassSpecGym Hugging Face candidate JSONs are keyed by query molecule SMILES
and contain candidate SMILES lists. This script converts those JSONs to the TSV
format consumed by ``benchmarked_models/evaluation/evaluate_mist_retrieval.py``:

    spec, smiles, inchikey, ionization, instrument, precursor, collision_energies

It is intentionally self-contained for this reproduction. It depends on pandas,
RDKit, and tqdm, but not on ``ms_pred``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import shutil
import signal
import urllib.request
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import inchi
from rdkit import rdBase
from tqdm import tqdm


DEFAULT_DATASET_DIR = Path("data/spec_datasets/msg_all")
DEFAULT_CANDIDATE_URLS = {
    "formula": (
        "https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/resolve/main/"
        "data/molecules/MassSpecGym1.5_retrieval_candidates_formula.json"
    ),
    "mass": (
        "https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/resolve/main/"
        "data/molecules/MassSpecGym1.5_retrieval_candidates_mass.json"
    ),
}
OUTPUT_COLUMNS = [
    "spec",
    "smiles",
    "inchikey",
    "ionization",
    "instrument",
    "precursor",
    "collision_energies",
]


class StandardizationTimeout(RuntimeError):
    """Raised when RDKit/InChI standardization exceeds the configured timeout."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download MassSpecGym retrieval candidate JSONs and convert them to "
            "candidate TSVs for this reproduction."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Dataset directory containing labels.tsv and splits/. Defaults to msg_all.",
    )
    parser.add_argument(
        "--labels-file",
        type=Path,
        default=None,
        help="Labels TSV. Defaults to <dataset-dir>/labels.tsv.",
    )
    parser.add_argument(
        "--split-file",
        default="split.tsv",
        help=(
            "Split file name under <dataset-dir>/splits, or an absolute path. "
            "Use --no-split-file to select IDs from MGF files or all labels."
        ),
    )
    parser.add_argument(
        "--no-split-file",
        action="store_true",
        help="Do not read a split file; use --mgf-root/--mgf-splits or all labels.",
    )
    parser.add_argument(
        "--subset",
        default="test",
        help="Split subset to export. Defaults to test.",
    )
    parser.add_argument(
        "--mgf-root",
        type=Path,
        default=None,
        help=(
            "Optional MGF split root, e.g. data/MGF_files. When set, IDs are "
            "collected from <mgf-root>/<mgf-dataset>/<split>/<subset>.mgf."
        ),
    )
    parser.add_argument("--mgf-dataset", default="massspecgym")
    parser.add_argument(
        "--mgf-splits",
        nargs="+",
        default=None,
        help="MGF split names to collect IDs from, e.g. random scaffold.",
    )
    parser.add_argument(
        "--candidate-types",
        nargs="+",
        default=["formula", "mass"],
        choices=sorted(DEFAULT_CANDIDATE_URLS),
        help="Candidate JSON types to process.",
    )
    parser.add_argument("--formula-url", default=DEFAULT_CANDIDATE_URLS["formula"])
    parser.add_argument("--mass-url", default=DEFAULT_CANDIDATE_URLS["mass"])
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Where JSON files are stored. Defaults to <dataset-dir>/retrieval/hf_raw.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where candidate TSVs are written. Defaults to <dataset-dir>/retrieval.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=256,
        help="Maximum candidates per spectrum, including the true candidate.",
    )
    parser.add_argument(
        "--expected-spec-count",
        type=int,
        default=None,
        help="Optional exact unique spec count expected in each output TSV.",
    )
    parser.add_argument(
        "--standardize-timeout",
        type=float,
        default=30.0,
        help="Seconds allowed for one SMILES standardization. Set <=0 to disable.",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--compare-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory with existing TSVs of the same output names. "
            "Generated files are compared by unordered per-spec InChIKey sets."
        ),
    )
    parser.add_argument(
        "--compare-formula-tsv",
        type=Path,
        default=None,
        help="Optional existing formula TSV to compare against.",
    )
    parser.add_argument(
        "--compare-mass-tsv",
        type=Path,
        default=None,
        help="Optional existing mass TSV to compare against.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use existing JSON files in --raw-dir instead of downloading.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing output TSVs.",
    )
    return parser.parse_args()


def normalize_hf_url(url: str) -> str:
    return url.replace("/blob/", "/resolve/")


def download_file(url: str, out_path: Path, overwrite: bool = False) -> None:
    if out_path.exists() and not overwrite:
        print(f"Using existing JSON: {out_path}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".download")
    url = normalize_hf_url(url)
    print(f"Downloading {url} -> {out_path}")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp_path.replace(out_path)


def resolve_split_path(dataset_dir: Path, split_file: str) -> Path:
    split_path = Path(split_file)
    if split_path.is_absolute():
        return split_path
    return dataset_dir / "splits" / split_path


def get_split_columns(split_df: pd.DataFrame) -> tuple[str, str]:
    name_col = "spec" if "spec" in split_df.columns else "name"
    if name_col not in split_df.columns:
        raise ValueError("Split file must contain either a 'spec' or 'name' column.")
    if "split" in split_df.columns:
        return name_col, "split"
    fold_cols = [col for col in split_df.columns if col != name_col]
    if len(fold_cols) != 1:
        raise ValueError("Split file must contain one fold column besides spec/name.")
    return name_col, fold_cols[0]


def iter_mgf_ids(path: Path) -> Iterator[str]:
    with path.open(errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line.upper().startswith("ID_="):
                yield line.split("=", 1)[1]


def collect_mgf_subset_ids(
    mgf_root: Path,
    dataset: str,
    splits: Iterable[str],
    subset: str,
) -> set[str]:
    ids: set[str] = set()
    for split in splits:
        mgf_path = mgf_root / dataset / split / f"{subset}.mgf"
        if not mgf_path.exists():
            raise FileNotFoundError(mgf_path)
        ids.update(iter_mgf_ids(mgf_path))
    return ids


def rdkit_canonical_smiles(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def inchikey_2d(value: str | None) -> str | None:
    if not value:
        return None
    return str(value).replace("ikey ", "").strip().split("-")[0]


def stereo_group_key(value: str | None) -> str | None:
    """Return the 2D InChIKey block used by retrieval hit-rate scoring."""
    return inchikey_2d(value)


@contextlib.contextmanager
def standardization_alarm(seconds: float) -> Iterator[None]:
    if seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return

    def _handler(signum, frame):
        raise StandardizationTimeout()

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def inchi_roundtrip_smiles(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    inchi_text = inchi.MolToInchi(mol)
    if not inchi_text:
        return None
    roundtrip_mol = inchi.MolFromInchi(inchi_text)
    if roundtrip_mol is None:
        return None
    return Chem.MolToSmiles(roundtrip_mol, isomericSmiles=True)


def inchikey_from_smiles(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    key = inchi.MolToInchiKey(mol)
    return key.replace("ikey ", "").strip() if key else None


def standardize_candidate(
    smi: str,
    cache: dict[str, tuple[str, str] | None],
    timeout: float,
) -> tuple[str, str] | None:
    smi = str(smi)
    if smi in cache:
        return cache[smi]

    try:
        with standardization_alarm(timeout):
            norm_smi = inchi_roundtrip_smiles(smi)
    except StandardizationTimeout:
        cache[smi] = None
        return None

    if norm_smi is None:
        cache[smi] = None
        return None
    norm_ikey = inchikey_from_smiles(norm_smi)
    if not norm_ikey:
        cache[smi] = None
        return None

    cache[smi] = (norm_smi, norm_ikey)
    return cache[smi]


def standardize_candidate_worker(args: tuple[str, float]) -> tuple[str, tuple[str, str] | None]:
    smi, timeout = args
    rdBase.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.*")
    return smi, standardize_candidate(smi, {}, timeout)


def normalize_true_structure(
    smiles: str,
    inchikey: str,
    cache: dict[tuple[str, str], tuple[str, str]],
    timeout: float,
) -> tuple[str, str]:
    key = (str(smiles), str(inchikey))
    if key in cache:
        return cache[key]

    try:
        with standardization_alarm(timeout):
            norm_smi = inchi_roundtrip_smiles(smiles)
    except StandardizationTimeout:
        norm_smi = None

    clean_inchikey = str(inchikey).replace("ikey ", "").strip()
    if norm_smi is None:
        out = (str(smiles), clean_inchikey)
        cache[key] = out
        return out

    norm_ikey = inchikey_from_smiles(norm_smi)
    out = (norm_smi, norm_ikey or clean_inchikey)
    cache[key] = out
    return out


def load_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if "spec" not in labels.columns and "name" in labels.columns:
        labels = labels.rename(columns={"name": "spec"})
    if "collision_energies" not in labels.columns:
        labels["collision_energies"] = ""
    if "precursor" not in labels.columns and "precursor_MZ" in labels.columns:
        labels = labels.rename(columns={"precursor_MZ": "precursor"})
    if "instrument" not in labels.columns:
        labels["instrument"] = "Orbitrap"
    return labels


def select_subset_labels(
    labels: pd.DataFrame,
    dataset_dir: Path,
    split_file: str,
    subset: str,
    no_split_file: bool,
    mgf_root: Path | None,
    mgf_dataset: str,
    mgf_splits: list[str] | None,
) -> pd.DataFrame:
    selected_ids: set[str] | None = None
    if mgf_root is not None:
        if not mgf_splits:
            raise ValueError("--mgf-splits is required when --mgf-root is set")
        selected_ids = collect_mgf_subset_ids(mgf_root, mgf_dataset, mgf_splits, subset)
    elif not no_split_file:
        split_path = resolve_split_path(dataset_dir, split_file)
        split_df = pd.read_csv(split_path, sep="\t", dtype=str, keep_default_na=False)
        name_col, split_col = get_split_columns(split_df)
        selected_ids = set(split_df.loc[split_df[split_col] == subset, name_col].astype(str))

    if selected_ids is not None:
        labels = labels[labels["spec"].astype(str).isin(selected_ids)].copy()
    else:
        labels = labels.copy()

    labels = labels.sort_values("spec").reset_index(drop=True)
    if len(labels) == 0:
        raise ValueError("No labels selected for candidate export.")
    missing_ids = selected_ids - set(labels["spec"].astype(str)) if selected_ids is not None else set()
    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} selected IDs are absent from labels. "
            f"Examples: {sorted(missing_ids)[:10]}"
        )
    return labels


def load_subset_labels(args: argparse.Namespace) -> pd.DataFrame:
    labels_path = args.labels_file or args.dataset_dir / "labels.tsv"
    labels = load_labels(labels_path)
    labels = select_subset_labels(
        labels=labels,
        dataset_dir=args.dataset_dir,
        split_file=args.split_file,
        subset=args.subset,
        no_split_file=args.no_split_file,
        mgf_root=args.mgf_root,
        mgf_dataset=args.mgf_dataset,
        mgf_splits=args.mgf_splits,
    )

    required = {"spec", "smiles", "inchikey", "ionization", "precursor", "collision_energies"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"{labels_path} is missing required columns: {missing}")

    query_keys = labels["smiles"].map(rdkit_canonical_smiles)
    if query_keys.isna().any():
        bad = labels.loc[query_keys.isna(), "spec"].head(10).tolist()
        raise ValueError(f"Invalid label SMILES. Examples: {bad}")
    labels["query_key"] = query_keys

    norm_cache: dict[tuple[str, str], tuple[str, str]] = {}
    query_norm_map: dict[str, tuple[str, str]] = {}
    query_label_table = labels[["query_key", "smiles", "inchikey"]].drop_duplicates("query_key")
    for idx, row in enumerate(query_label_table.itertuples(index=False), start=1):
        query_norm_map[row.query_key] = normalize_true_structure(
            row.smiles,
            row.inchikey,
            norm_cache,
            args.standardize_timeout,
        )
        if idx % 100 == 0 or idx == len(query_label_table):
            print(f"Normalized {idx}/{len(query_label_table)} unique true label structures", flush=True)

    labels["true_smiles"] = labels["query_key"].map(lambda key: query_norm_map[key][0])
    labels["true_inchikey"] = labels["query_key"].map(lambda key: query_norm_map[key][1])
    if labels["true_smiles"].eq("").any() or labels["true_inchikey"].eq("").any():
        bad = labels.loc[
            labels["true_smiles"].eq("") | labels["true_inchikey"].eq(""),
            "spec",
        ].head(10).tolist()
        raise ValueError(f"Could not normalize true label structures. Examples: {bad}")

    return labels


def build_query_candidate_cache(
    labels: pd.DataFrame,
    candidate_map: dict[str, list[str]],
    max_candidates: int | None,
    timeout: float,
    num_workers: int,
) -> tuple[dict[str, list[tuple[str, str]]], Counter[str]]:
    query_table = labels[["query_key", "true_smiles", "true_inchikey"]].drop_duplicates()
    missing_queries = sorted(set(query_table["query_key"]) - set(candidate_map))
    if missing_queries:
        raise ValueError(
            f"{len(missing_queries)} query SMILES keys are absent from the JSON. "
            f"Examples: {missing_queries[:5]}"
        )

    raw_candidate_smiles = sorted(
        {
            str(cand_smi)
            for query_key in query_table["query_key"]
            for cand_smi in candidate_map[query_key]
        }
    )
    print(
        f"Standardizing {len(raw_candidate_smiles)} unique raw candidate SMILES "
        f"with {max(1, num_workers)} worker(s)",
        flush=True,
    )
    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            norm_cache = dict(
                tqdm(
                    pool.imap_unordered(
                        standardize_candidate_worker,
                        ((smi, timeout) for smi in raw_candidate_smiles),
                        chunksize=100,
                    ),
                    total=len(raw_candidate_smiles),
                    desc="Standardizing candidate SMILES",
                )
            )
    else:
        norm_cache: dict[str, tuple[str, str] | None] = {}
        for smi in tqdm(raw_candidate_smiles, desc="Standardizing candidate SMILES"):
            norm_cache[smi] = standardize_candidate(smi, norm_cache, timeout)

    query_candidates: dict[str, list[tuple[str, str]]] = {}
    stats: Counter[str] = Counter()
    for idx, row in enumerate(query_table.itertuples(index=False), start=1):
        true_group = stereo_group_key(row.true_inchikey)
        deduped: dict[str, str] = {}

        for cand_smi in candidate_map[row.query_key]:
            norm = norm_cache.get(str(cand_smi))
            if norm is None:
                stats["invalid_or_timeout_candidates"] += 1
                continue
            cand_norm_smi, cand_norm_ikey = norm
            if stereo_group_key(cand_norm_ikey) == true_group:
                stats["raw_candidates_matching_true_group"] += 1
                continue
            deduped.setdefault(cand_norm_ikey, cand_norm_smi)

        decoys = [(smi, ikey) for ikey, smi in deduped.items()]
        if max_candidates is not None:
            decoys = decoys[: max_candidates - 1]
        query_candidates[row.query_key] = [(row.true_smiles, row.true_inchikey)] + decoys
        stats[f"candidate_count_{len(query_candidates[row.query_key])}"] += 1

        if idx % 100 == 0 or idx == len(query_table):
            print(f"Standardized {idx}/{len(query_table)} unique query candidate lists")

    return query_candidates, stats


def write_candidate_tsv(
    labels: pd.DataFrame,
    query_candidates: dict[str, list[tuple[str, str]]],
    out_path: Path,
    overwrite: bool,
) -> None:
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists. Pass --overwrite to replace it.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    rows = 0
    specs = set()
    with tmp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in labels.itertuples(index=False):
            base = {
                "spec": row.spec,
                "ionization": row.ionization,
                "instrument": row.instrument,
                "precursor": row.precursor,
                "collision_energies": row.collision_energies,
            }
            for cand_smi, cand_ikey in query_candidates[row.query_key]:
                writer.writerow({**base, "smiles": cand_smi, "inchikey": cand_ikey})
                rows += 1
            specs.add(row.spec)
    tmp_path.replace(out_path)
    print(f"Wrote {out_path}: rows={rows} unique_specs={len(specs)}")


def validate_output(labels: pd.DataFrame, out_path: Path, expected_spec_count: int | None) -> None:
    true_group = {
        row.spec: stereo_group_key(row.true_inchikey)
        for row in labels[["spec", "true_inchikey"]].itertuples(index=False)
    }
    expected_specs = set(true_group)
    present = {spec: False for spec in expected_specs}
    specs = set()
    rows = 0

    for chunk in pd.read_csv(
        out_path,
        sep="\t",
        usecols=["spec", "inchikey"],
        dtype=str,
        keep_default_na=False,
        chunksize=500000,
    ):
        rows += len(chunk)
        chunk_specs = set(chunk["spec"].astype(str))
        specs.update(chunk_specs)
        chunk["target_group"] = chunk["spec"].map(true_group)
        chunk["candidate_group"] = chunk["inchikey"].map(stereo_group_key)
        for spec in chunk.loc[chunk["target_group"] == chunk["candidate_group"], "spec"].unique():
            present[str(spec)] = True

    missing_specs = sorted(expected_specs - specs)
    extra_specs = sorted(specs - expected_specs)
    missing_true = sorted(spec for spec, found in present.items() if not found)
    if expected_spec_count is not None and len(specs) != expected_spec_count:
        raise ValueError(
            f"{out_path}: expected {expected_spec_count} unique specs, found {len(specs)}."
        )
    if missing_specs or extra_specs or missing_true:
        raise ValueError(
            f"{out_path}: missing_specs={len(missing_specs)} extra_specs={len(extra_specs)} "
            f"missing_true_candidate={len(missing_true)}; "
            f"missing_true_examples={missing_true[:10]}"
        )
    print(
        f"Validated {out_path}: rows={rows} unique_specs={len(specs)} "
        "missing_true_candidate=0"
    )


def candidate_sets(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=["spec", "inchikey"],
        dtype=str,
        keep_default_na=False,
        chunksize=500000,
    ):
        for spec_id, group in chunk.groupby("spec", sort=False):
            values = out.setdefault(str(spec_id), set())
            values.update(str(value).strip() for value in group["inchikey"] if str(value).strip())
    return out


def compare_candidate_tsvs(reference_path: Path, generated_path: Path) -> None:
    reference = candidate_sets(reference_path)
    generated = candidate_sets(generated_path)
    missing_specs = sorted(set(reference) - set(generated))
    extra_specs = sorted(set(generated) - set(reference))
    mismatched = []
    for spec_id in sorted(set(reference) & set(generated)):
        if reference[spec_id] != generated[spec_id]:
            mismatched.append(
                (
                    spec_id,
                    len(reference[spec_id] - generated[spec_id]),
                    len(generated[spec_id] - reference[spec_id]),
                )
            )

    if missing_specs or extra_specs or mismatched:
        raise ValueError(
            f"{generated_path} does not match {reference_path}: "
            f"missing_specs={len(missing_specs)} extra_specs={len(extra_specs)} "
            f"mismatched_specs={len(mismatched)}; "
            f"missing_examples={missing_specs[:5]} extra_examples={extra_specs[:5]} "
            f"mismatch_examples={mismatched[:5]}"
        )
    print(
        f"Matched {generated_path} to {reference_path}: "
        f"specs={len(generated)} candidate sets identical ignoring row order"
    )


def output_name(subset: str, candidate_type: str, max_candidates: int | None) -> str:
    if max_candidates is None:
        return f"cands_df_{subset}_{candidate_type}.tsv"
    return f"cands_df_{subset}_{candidate_type}_{max_candidates}.tsv"


def main() -> None:
    args = parse_args()
    rdBase.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.*")

    raw_dir = args.raw_dir or args.dataset_dir / "retrieval" / "hf_raw"
    out_dir = args.out_dir or args.dataset_dir / "retrieval"
    urls = {"formula": args.formula_url, "mass": args.mass_url}

    for candidate_type in args.candidate_types:
        raw_path = raw_dir / f"MassSpecGym1.5_retrieval_candidates_{candidate_type}.json"
        if not args.skip_download:
            download_file(urls[candidate_type], raw_path, overwrite=args.overwrite)

    labels = load_subset_labels(args)
    if args.expected_spec_count is not None and len(labels) != args.expected_spec_count:
        raise ValueError(
            f"Expected {args.expected_spec_count} labels in subset={args.subset}, found {len(labels)}."
        )
    print(
        f"Loaded {len(labels)} {args.subset} labels; "
        f"{labels['query_key'].nunique()} unique query molecules"
    )

    for candidate_type in args.candidate_types:
        raw_path = raw_dir / f"MassSpecGym1.5_retrieval_candidates_{candidate_type}.json"
        print(f"Loading {raw_path}")
        with raw_path.open() as handle:
            candidate_map = json.load(handle)

        query_candidates, stats = build_query_candidate_cache(
            labels=labels,
            candidate_map=candidate_map,
            max_candidates=args.max_candidates,
            timeout=args.standardize_timeout,
            num_workers=args.num_workers,
        )
        out_path = out_dir / output_name(args.subset, candidate_type, args.max_candidates)
        write_candidate_tsv(labels, query_candidates, out_path, overwrite=args.overwrite)
        validate_output(labels, out_path, expected_spec_count=args.expected_spec_count)
        compare_path = None
        if candidate_type == "formula" and args.compare_formula_tsv is not None:
            compare_path = args.compare_formula_tsv
        elif candidate_type == "mass" and args.compare_mass_tsv is not None:
            compare_path = args.compare_mass_tsv
        elif args.compare_dir is not None:
            compare_path = args.compare_dir / out_path.name
        if compare_path is not None:
            compare_candidate_tsvs(compare_path, out_path)
        print(f"{candidate_type} summary: {dict(stats)}")


if __name__ == "__main__":
    main()
