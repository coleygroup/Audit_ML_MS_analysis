#!/usr/bin/env python3
"""Extract same-formula PubChem candidates from a formula-keyed HDF5 file."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Iterable

import h5py
import pandas as pd


def iter_mgf_ids(path: Path) -> Iterable[str]:
    with path.open(errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line.upper().startswith("ID_="):
                yield line.split("=", 1)[1].strip()


def load_formulas(args: argparse.Namespace) -> list[str]:
    if args.formulas_file is not None:
        return sorted(
            {
                line.strip()
                for line in args.formulas_file.read_text().splitlines()
                if line.strip()
            }
        )

    metadata = pd.read_csv(args.metadata, sep="\t", dtype={"id": str})
    if "id" not in metadata.columns or args.formula_col not in metadata.columns:
        raise ValueError(
            f"{args.metadata} must contain id and {args.formula_col!r} columns"
        )
    id_to_formula = metadata.set_index("id")[args.formula_col].astype(str).to_dict()

    ids: set[str] = set()
    for split in args.splits:
        mgf_file = args.mgf_root / split / "test.mgf"
        ids.update(iter_mgf_ids(mgf_file))
    missing = sorted(ids - set(id_to_formula))
    if missing:
        raise ValueError(f"{len(missing)} MGF test ids are missing from {args.metadata}")
    return sorted({id_to_formula[spec_id] for spec_id in ids if id_to_formula[spec_id]})


def decode_hdf5_value(value) -> list:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        if len(value) == 1:
            value = value[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value)
    raise TypeError(f"Unsupported HDF5 value type: {type(value)!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata/NPLIB1_metadata.tsv"))
    parser.add_argument("--mgf-root", type=Path, default=Path("data/MGF_files/NPLIB1"))
    parser.add_argument("--splits", nargs="+", default=["random", "scaffold"])
    parser.add_argument("--formula-col", default="formula")
    parser.add_argument("--formulas-file", type=Path, default=None)
    parser.add_argument(
        "--candidates-output",
        type=Path,
        default=Path("results/comparison/nplib1_pubchem_hdf5_candidates.tsv.gz"),
    )
    parser.add_argument(
        "--counts-output",
        type=Path,
        default=Path("results/comparison/nplib1_pubchem_hdf5_candidate_counts.csv"),
    )
    args = parser.parse_args()

    formulas = load_formulas(args)
    rows = []
    args.candidates_output.parent.mkdir(parents=True, exist_ok=True)
    args.counts_output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.hdf5, "r") as h5, gzip.open(args.candidates_output, "wt") as out:
        for formula in formulas:
            if formula not in h5:
                rows.append({"formula": formula, "exists": 0, "n_candidates": 0})
                continue
            candidates = decode_hdf5_value(h5[formula][()])
            rows.append(
                {
                    "formula": formula,
                    "exists": 1,
                    "n_candidates": len(candidates),
                }
            )
            if candidates:
                out.write(f"{formula}\t{json.dumps(candidates)}\n")

    pd.DataFrame(rows).to_csv(args.counts_output, index=False)
    print(f"formulas={len(rows)}")
    print(f"with_candidates={sum(row['n_candidates'] > 0 for row in rows)}")
    print(f"total_candidates={sum(row['n_candidates'] for row in rows)}")
    print(f"wrote {args.candidates_output}")
    print(f"wrote {args.counts_output}")


if __name__ == "__main__":
    main()
