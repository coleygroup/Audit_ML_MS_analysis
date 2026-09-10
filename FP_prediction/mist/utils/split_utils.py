"""Repository-side split handling for MIST split files.

Upstream MIST's ``PresetSpectraSplitter`` reads the split TSV with a bare
``pd.read_csv``, so pandas infers column dtypes. NPLIB1 spectrum names are
purely numeric strings (``4745``, ``15289``, ...) and become ``int64``, while
the spectrum names produced by ``Spectra.get_spec_name()`` are always ``str``.
The resulting lookup never matches and every split comes back empty. The
failure is silent on MassSpecGym, whose spectrum names are non-numeric, so it
only bites on NPLIB1.

This module keeps the fix on the repository side instead of patching the
upstream package: it subclasses the upstream splitter and re-reads the split
file with ``dtype=str``. Column renaming (for split files that use
``spec``/``Fold_0`` instead of ``name``/``split``) is handled in memory here
too, so no normalized copy is ever written to disk.
"""

from pathlib import Path

import pandas as pd

from mist.data import splitter as _upstream_splitter


DEFAULT_NAME_COLS = ("name", "spec")
DEFAULT_SPLIT_COLS = ("split", "Fold_0")


def _pick_column(columns, configured, candidates, split_file):
    if configured is not None:
        if configured not in columns:
            raise ValueError(
                f"{split_file}: configured column {configured!r} not found; "
                f"available columns are {list(columns)}"
            )
        return configured
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(
        f"{split_file}: expected one of {list(candidates)} among columns "
        f"{list(columns)}"
    )


def read_split_frame(split_file, name_col=None, split_col=None):
    """Read a split TSV as strings, normalized to ``name``/``split`` columns."""
    split_file = Path(split_file)
    # dtype=str is the whole point: it must be applied at read time, because a
    # str cast applied before writing a TSV is undone by dtype inference when
    # the file is read back.
    split_df = pd.read_csv(split_file, sep="\t", dtype=str)
    name_col = _pick_column(split_df.columns, name_col, DEFAULT_NAME_COLS, split_file)
    split_col = _pick_column(split_df.columns, split_col, DEFAULT_SPLIT_COLS, split_file)
    normalized = split_df[[name_col, split_col]].rename(
        columns={name_col: "name", split_col: "split"}
    )
    normalized["name"] = normalized["name"].astype(str).str.strip()
    normalized["split"] = normalized["split"].astype(str).str.strip()
    return normalized


class StrKeyPresetSpectraSplitter(_upstream_splitter.PresetSpectraSplitter):
    """``PresetSpectraSplitter`` that matches split names as strings.

    ``PresetSpectraSplitter.__init__`` is deliberately not called: it performs
    the dtype-inferring read this class exists to replace, and it hard-codes
    ``name``/``split`` column names, so it raises on split files that use
    ``spec``/``Fold_0``. Only ``get_splits`` is inherited.
    """

    def __init__(self, split_file: str = None, **kwargs):
        _upstream_splitter.SpectraSplitter.__init__(self, **kwargs)
        if split_file is None:
            raise ValueError("Preset splitter requires split_file arg.")
        self.split_file = split_file
        self.split_name = Path(split_file).stem
        self.split_df = read_split_frame(
            split_file,
            name_col=kwargs.get("split_name_col"),
            split_col=kwargs.get("split_value_col"),
        )
        self.name_to_fold = dict(zip(self.split_df["name"], self.split_df["split"]))


def get_splitter(**kwargs):
    """Drop-in replacement for ``mist.data.splitter.get_splitter``."""
    return StrKeyPresetSpectraSplitter(**kwargs)
