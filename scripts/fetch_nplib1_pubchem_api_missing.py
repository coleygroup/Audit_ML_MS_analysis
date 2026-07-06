#!/usr/bin/env python3
"""Fetch PubChem same-formula candidates missing from the local HDF5 map."""

from __future__ import annotations

import argparse
import gzip
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
    "fastformula/{formula}/property/CanonicalSMILES,IsomericSMILES,InChIKey/JSON"
)


def load_existing_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") == "error":
                continue
            out[record["formula"]] = record
    return out


def append_cache(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def fetch_formula(formula: str, timeout: int) -> dict:
    url = PUBCHEM_URL.format(formula=urllib.parse.quote(formula, safe=""))
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ML-MS-analysis-repro/1.0 (PubChem formula completion)"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    candidates = []
    for item in payload.get("PropertyTable", {}).get("Properties", []):
        smiles = (
            item.get("ConnectivitySMILES")
            or item.get("CanonicalSMILES")
            or item.get("IsomericSMILES")
        )
        inchikey = item.get("InChIKey")
        if smiles and inchikey:
            candidates.append([smiles, inchikey])
    return {"formula": formula, "status": "ok", "candidates": candidates}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--counts",
        type=Path,
        default=Path("results/comparison/nplib1_pubchem_hdf5_candidate_counts.csv"),
        help="CSV produced from the local PubChem HDF5 extract.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("results/comparison/nplib1_pubchem_api_missing_cache.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/comparison/nplib1_pubchem_api_missing_candidates.tsv.gz"),
    )
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    counts = pd.read_csv(args.counts)
    formulas = counts.loc[counts["n_candidates"].eq(0), "formula"].astype(str).tolist()
    cache = load_existing_cache(args.cache)

    for index, formula in enumerate(formulas, start=1):
        if formula in cache:
            continue
        record = None
        for attempt in range(1, args.retries + 1):
            try:
                record = fetch_formula(formula, args.timeout)
                break
            except urllib.error.HTTPError as exc:
                if exc.code in {404, 400}:
                    record = {
                        "formula": formula,
                        "status": f"http_{exc.code}",
                        "candidates": [],
                    }
                    break
                wait = args.sleep * attempt * 4
                last_error = f"http_{exc.code}: {exc.reason}"
            except Exception as exc:  # noqa: BLE001 - keep cache resumable on network errors.
                wait = args.sleep * attempt * 4
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < args.retries:
                time.sleep(wait)
        if record is None:
            record = {
                "formula": formula,
                "status": "error",
                "error": last_error,
                "candidates": [],
            }
        append_cache(args.cache, record)
        cache[formula] = record
        print(
            f"{index}/{len(formulas)} {formula} {record['status']} "
            f"n={len(record['candidates'])}",
            flush=True,
        )
        time.sleep(args.sleep)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt") as handle:
        for formula in formulas:
            record = cache[formula]
            if record.get("candidates"):
                handle.write(f"{formula}\t{json.dumps(record['candidates'])}\n")
    fetched = [cache[formula] for formula in formulas]
    print(
        "summary "
        f"formulas={len(formulas)} "
        f"with_candidates={sum(bool(r.get('candidates')) for r in fetched)} "
        f"total_candidates={sum(len(r.get('candidates', [])) for r in fetched)} "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
