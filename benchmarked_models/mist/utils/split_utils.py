"""Repository-side split handling for MIST.

MIST's ``PresetSpectraSplitter`` reads the split TSV with a bare ``pd.read_csv``,
so pandas infers column dtypes. NPLIB1 spectrum names are purely numeric strings
(``4745``, ``15289``, ...) and become ``int64``, while the spectrum names produced
by ``Spectra.get_spec_name()`` are always ``str``. The resulting lookup never
matches and every split comes back empty. The failure is silent on MassSpecGym,
whose spectrum names are non-numeric, so it only bites on NPLIB1.

This module keeps the fix on the repository side instead of patching the upstream
package: it subclasses the upstream splitter and re-reads the split file with
``dtype=str``.
"""

from pathlib import Path

import pandas as pd

from mist.data import splitter as _upstream_splitter


class StrKeyPresetSpectraSplitter(_upstream_splitter.PresetSpectraSplitter):
    """``PresetSpectraSplitter`` that matches split names as strings.

    ``PresetSpectraSplitter.__init__`` is deliberately not called: it performs the
    dtype-inferring read this class exists to replace. Only ``get_splits`` is
    inherited.
    """

    def __init__(self, split_file: str = None, **kwargs):
        _upstream_splitter.SpectraSplitter.__init__(self, **kwargs)
        if split_file is None:
            raise ValueError("Preset splitter requires split_file arg.")

        self.split_file = split_file
        self.split_name = Path(split_file).stem
        # dtype=str is the whole point: it must be applied at read time, because a
        # str cast applied before writing a TSV is undone by dtype inference when
        # the file is read back.
        self.split_df = pd.read_csv(self.split_file, sep="\t", dtype=str)
        self.split_df["name"] = self.split_df["name"].str.strip()
        self.split_df["split"] = self.split_df["split"].str.strip()
        self.name_to_fold = dict(zip(self.split_df["name"], self.split_df["split"]))


def get_splitter(**kwargs):
    """Drop-in replacement for ``mist.data.splitter.get_splitter``."""
    return StrKeyPresetSpectraSplitter(**kwargs)
