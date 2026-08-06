"""Recover NPLIB1 test spectra present in the MGF export but missing from the MIST export.

The NPLIB1 MGF export used by the nearest-neighbour and DreaMS pipelines contains
23 random and 5 scaffold test spectra (27 unique) that the MIST data export omits.
Those spectra have no ``.ms`` file and no subformula JSON, and they appear in
neither ``labels.tsv`` nor ``splits/*.tsv``. Without them MIST scores 2,721 random
and 2,684 scaffold test spectra while every other method scores 2,744 and 2,689,
so the rows are not directly comparable.

Nothing is actually missing from the underlying data: the peaks are in the MGF and
the structures are in the metadata TSV. This script rebuilds the MIST-side inputs:

1. finds MGF test ids absent from the MIST split file,
2. writes MIST-format ``.ms`` files from the MGF peaks plus metadata fields,
3. runs MIST's own ``assign_subformulae.py`` to generate their subformula JSONs,
4. writes an extended data root (labels, splits, spectra, subformulae) that merges
   the originals with the recovered entries.

The recovered spectra are added as ``test`` entries only. They are absent from the
original split file entirely, so they were never in the training or validation
folds and no leakage is introduced. Re-running ``predict.py`` against the extended
root is sufficient; no retraining is required.

Example:

    python scripts/recover_nplib1_missing_test_spectra.py \\
      --mist-data-root /path/to/scratch/mist_repro/raw/google_drive_mist_outputs/NPLIB1 \\
      --mgf-root data/MGF_files/NPLIB1 \\
      --metadata data/metadata/NPLIB1_metadata.tsv \\
      --output-root /path/to/scratch/mist_repro/nplib1_extended
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

SPLITS = ("random", "scaffold")
LABEL_COLUMNS = [
    "dataset",
    "spec",
    "name",
    "ionization",
    "formula",
    "smiles",
    "inchikey",
    "instrument",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mist-data-root", type=Path, required=True,
                        help="NPLIB1 dir of the MIST export (labels.tsv, spec_files/, subformulae/, splits/).")
    parser.add_argument("--mgf-root", type=Path, required=True,
                        help="NPLIB1 MGF dir containing <split>/test.mgf.")
    parser.add_argument("--metadata", type=Path, required=True,
                        help="NPLIB1 metadata TSV supplying SMILES/formula/precursor for recovered spectra.")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Where to write the extended data root.")
    parser.add_argument("--feature-id", default="ID_",
                        help="MGF metadata key holding the spectrum id (default: ID_).")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Workers for subformula assignment.")
    parser.add_argument("--skip-subformulae", action="store_true",
                        help="Assume subformula JSONs already exist under <output-root>/_recovered/subformulae.")
    return parser.parse_args()


def iter_mgf_records(path: Path):
    metadata, peaks = None, []
    with open(path, errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            upper = line.upper()
            if upper == "BEGIN IONS":
                metadata, peaks = {}, []
                continue
            if upper == "END IONS":
                if metadata is not None:
                    yield metadata, peaks
                metadata, peaks = None, []
                continue
            if metadata is None:
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                metadata[key.upper()] = value.strip()
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    peaks.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass


def spec_id_of(metadata, feature_id):
    for key in (feature_id.upper(), "ID_", "TITLE", "NAME"):
        value = metadata.get(key)
        if value:
            return value.strip()
    return ""


def write_ms_file(path: Path, spec_id, row, peaks, mgf_meta):
    formula = row.get("formula_corrected") or row.get("formula") or ""
    ionization = row.get("ionization") or "[M+H]+"
    parentmass = row.get("precursor_mz_final") or row.get("precursor_mz") or ""
    if not parentmass:
        raw = mgf_meta.get("PRECURSOR_MZ") or mgf_meta.get("PEPMASS") or ""
        parentmass = raw.split()[0] if raw else ""
    inchikey = row.get("inchikey_full") or row.get("inchikey") or "None"
    lines = [
        f">compound {spec_id}",
        f">formula {formula}",
        f">parentmass {parentmass}",
        f">ionization {ionization}",
        ">InChi None",
        f">InChiKey {inchikey}",
        f"#smiles {row['smiles']}",
        f"#instrumentation {row.get('instrument') or ''}",
        "",
        ">ms2peaks",
    ]
    lines += [f"{mz} {inten}" for mz, inten in peaks]
    path.write_text("\n".join(lines) + "\n")
    return {
        "dataset": "NPLIB1",
        "spec": spec_id,
        "name": spec_id,
        "ionization": ionization,
        "formula": formula,
        "smiles": row["smiles"],
        "inchikey": inchikey,
        "instrument": row.get("instrument") or "",
    }


def link_tree(src_dir: Path, dst_dir: Path, pattern: str):
    """Symlink every file matching pattern from src into dst (flat)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in src_dir.rglob(pattern):
        if not path.is_file():
            continue
        target = dst_dir / path.name
        if not target.exists():
            os.symlink(path.resolve(), target)


def main():
    args = parse_args()
    mist_root = args.mist_data_root
    out_root = args.output_root
    staging = out_root / "_recovered"
    staging_specs = staging / "spec_files"
    staging_specs.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.metadata, sep="\t", dtype=str).fillna("")
    meta["spec"] = meta["spec"].str.strip()
    meta_by_id = {r["spec"]: r for _, r in meta.iterrows()}

    label_rows = []
    recovered_by_split = {}

    for split in SPLITS:
        split_df = pd.read_csv(mist_root / "splits" / f"{split}.tsv", sep="\t", dtype=str)
        known = set(split_df["name"].str.strip())
        mgf_path = args.mgf_root / split / "test.mgf"

        missing = []
        for mgf_meta, peaks in iter_mgf_records(mgf_path):
            spec_id = spec_id_of(mgf_meta, args.feature_id)
            if not spec_id or spec_id in known:
                continue
            missing.append(spec_id)
            target = staging_specs / f"{spec_id}.ms"
            if target.exists():
                continue
            row = meta_by_id.get(spec_id)
            if row is None or not row.get("smiles"):
                print(f"  WARNING {split}/{spec_id}: no metadata or SMILES, cannot recover")
                continue
            label_rows.append(write_ms_file(target, spec_id, row, peaks, mgf_meta))

        recovered_by_split[split] = sorted(set(missing))
        print(f"{split}: {len(recovered_by_split[split])} MGF-only test spectra")

    if not label_rows and not args.skip_subformulae:
        print("Nothing to recover; the MIST export already covers the MGF test set.")
        return

    recovered_labels = pd.DataFrame(label_rows, columns=LABEL_COLUMNS).drop_duplicates(subset=["spec"])
    recovered_labels.to_csv(staging / "labels.tsv", sep="\t", index=False)
    print(f"wrote {len(recovered_labels)} recovered spectra to {staging_specs}")

    subform_out = staging / "subformulae"
    if not args.skip_subformulae:
        try:
            import mist.subformulae.assign_subformulae as assign_mod
            script = Path(assign_mod.__file__)
        except Exception as exc:  # pragma: no cover - depends on install layout
            raise SystemExit(
                "Could not locate mist.subformulae.assign_subformulae; ensure MIST vFRIGID "
                f"is importable (see REPRODUCING_RESULTS.md section 1). Original error: {exc}"
            )
        cmd = [
            sys.executable, str(script),
            "--spec-files", str(staging_specs),
            "--labels-file", str(staging / "labels.tsv"),
            "--output-dir", str(subform_out),
            "--num-workers", str(args.num_workers),
        ]
        print("running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    produced = list(subform_out.rglob("*.json"))
    print(f"subformula assignments available: {len(produced)}/{len(recovered_labels)}")
    if len(produced) < len(recovered_labels):
        print("  NOTE: assignment did not cover every recovered spectrum; "
              "those spectra will remain unscored.")

    # Extended root: originals plus recovered entries.
    ext_specs = out_root / "spec_files"
    ext_subform = out_root / "subformulae" / "default_subformulae"
    link_tree(mist_root / "spec_files", ext_specs, "*.ms")
    link_tree(mist_root / "subformulae" / "default_subformulae", ext_subform, "*.json")
    link_tree(staging_specs, ext_specs, "*.ms")
    link_tree(subform_out, ext_subform, "*.json")

    base_labels = pd.read_csv(mist_root / "labels.tsv", sep="\t", dtype=str)
    for column in base_labels.columns:
        if column not in recovered_labels.columns:
            recovered_labels[column] = ""
    merged = pd.concat(
        [base_labels, recovered_labels[base_labels.columns]], ignore_index=True
    ).drop_duplicates(subset=["spec"])
    merged.to_csv(out_root / "labels.tsv", sep="\t", index=False)

    (out_root / "splits").mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        split_df = pd.read_csv(mist_root / "splits" / f"{split}.tsv", sep="\t", dtype=str)
        ids = recovered_by_split[split]
        added = pd.DataFrame({"name": ids, "split": ["test"] * len(ids)})
        out = pd.concat([split_df, added], ignore_index=True).drop_duplicates(subset=["name"])
        out.to_csv(out_root / "splits" / f"{split}.tsv", sep="\t", index=False)
        print(f"{split}: test entries {int((split_df['split'] == 'test').sum())} -> "
              f"{int((out['split'] == 'test').sum())}")

    print(f"\nExtended data root: {out_root}")
    print("Point the dataset paths of your run.yaml / config at:")
    print(f"  labels_file:    {out_root / 'labels.tsv'}")
    print(f"  spec_folder:    {ext_specs}")
    print(f"  subform_folder: {ext_subform}")
    print(f"  split_file:     {out_root / 'splits' / '<split>.tsv'}")


if __name__ == "__main__":
    main()
