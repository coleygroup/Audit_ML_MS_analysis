import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from benchmarked_models.common.benchmark_utils import fingerprint_to_bits


def bin_peaks(peaks, precursor_mz=None, bin_resolution=0.25, max_da=2000.0):
    n_bins = int(math.ceil(max_da / bin_resolution))
    out = np.zeros((n_bins,), dtype=np.float32)
    inv = 1.0 / bin_resolution
    for mz, intensity in peaks:
        bin_idx = math.floor(float(mz) * inv)
        if 0 <= bin_idx < n_bins:
            out[bin_idx] += float(intensity)
    if precursor_mz is not None:
        bin_idx = math.floor(float(precursor_mz) * inv)
        if 0 <= bin_idx < n_bins:
            out[bin_idx] += 100.0
    return out


def _parse_metadata(line):
    if "=" not in line:
        return None, None
    key, value = line.rstrip("\n").split("=", 1)
    return key.upper(), value.strip()


def iter_mgf_records(path: Path):
    metadata = None
    peaks = []
    with open(path, errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            upper = line.upper()
            if upper == "BEGIN IONS":
                metadata = {}
                peaks = []
                continue
            if upper == "END IONS":
                if metadata is not None:
                    yield metadata, peaks
                metadata = None
                peaks = []
                continue
            if metadata is None:
                continue
            key, value = _parse_metadata(line)
            if key is not None:
                metadata[key] = value
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    peaks.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass


def load_mgf_records(
    path: Path,
    bin_resolution=0.25,
    max_da=2000.0,
    metadata_by_id: Optional[Dict[str, Dict[str, str]]] = None,
):
    ms_info = {}
    fp_info = {}
    formula_info = {}
    inchikey_info = {}
    smiles_info = {}
    for metadata, peaks in iter_mgf_records(path):
        spec_id = metadata.get("ID_") or metadata.get("TITLE") or metadata.get("NAME")
        if not spec_id:
            continue
        sidecar = metadata_by_id.get(spec_id, {}) if metadata_by_id else {}
        precursor = metadata.get("PRECURSOR_MZ") or metadata.get("PEPMASS")
        if precursor and " " in precursor:
            precursor = precursor.split()[0]
        ms_info[spec_id] = bin_peaks(
            peaks,
            precursor_mz=float(precursor) if precursor else None,
            bin_resolution=bin_resolution,
            max_da=max_da,
        )
        fp = metadata.get("FP") or sidecar.get("fp") or sidecar.get("fingerprint")
        if fp:
            fp_info[spec_id] = fingerprint_to_bits(fp)
        formula_info[spec_id] = (
            metadata.get("FORMULA")
            or metadata.get("CHEMICAL_FORMULA")
            or sidecar.get("formula")
        )
        inchikey_info[spec_id] = metadata.get("INCHIKEY") or sidecar.get("inchikey")
        smiles_info[spec_id] = metadata.get("SMILES") or sidecar.get("smiles")
    return ms_info, fp_info, formula_info, inchikey_info, smiles_info


def load_metadata_sidecar(path: Optional[Path]):
    if path is None:
        return None
    import pandas as pd

    df = pd.read_csv(path, sep=None, engine="python")
    id_col = next((c for c in ["spec", "name", "id", "id_"] if c in df.columns), None)
    if id_col is None:
        raise ValueError(f"No id column found in metadata sidecar {path}")
    out = {}
    for _, row in df.iterrows():
        out[str(row[id_col])] = {str(k).lower(): v for k, v in row.items()}
    return out
