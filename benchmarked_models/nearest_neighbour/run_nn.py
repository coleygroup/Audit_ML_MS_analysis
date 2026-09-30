"""
Nearest-neighbour baselines for fingerprint prediction from MS/MS spectra: the single entry point.

For every test spectrum, training spectra are retrieved and the fingerprint of the retrieved compound(s) is the
prediction. Nothing is learned; retrieval is always against the training split, the validation split is used only to
select the hyperparameters of the sub-formula rule, and the test split is never used for any choice.

Settings (--setting; several may be given; default "all"):

    <sim>_no_formula_filter      formula unknown: the whole training split is searched with <sim>
    <sim>_with_formula_filter    formula known: only the training spectra with the test spectrum's formula are searched;
                                 when there are none, the whole training split (= <sim>_no_formula_filter)
    <sim>_subformula_match       as <sim>_with_formula_filter, but spectra without a same-formula training spectrum get
                                 the sub-formula rule instead
    subformula_only              the sub-formula rule for every test spectrum (no spectral similarity)

    <sim> is one of
        cosine            matchms CosineGreedy
        modified_cosine   matchms ModifiedCosine
        neutral_loss      matchms NeutralLossesCosine
        dreams            cosine of DreaMS embeddings (utils/cache_dreams_embeddings.py, conda env `dreams`)
        binned            cosine of 0.25 Da binned spectra with the precursor peak

    "all" runs every setting (16); a mode name (no_formula_filter, with_formula_filter, subformula_match) runs its five
    similarities. DreaMS settings are skipped with a warning when "all" / a mode name selects them and no cached
    embeddings exist for the dataset / split.

    The sub-formula rule ranks training spectra by the cosine between bags of sub-formula fragment / neutral-loss
    tokens plus alpha x a heteroatom-weighted formula similarity and predicts the softmax(T)-weighted average of the
    top-k fingerprints, thresholded at 0.5; (w, alpha, T, k) are selected on the validation split
    (utils/subformula_fallback.py). Peak-matching similarities are checked against matchms before every full search
    (utils/peak_similarity.py).

Datasets: the built-in NPLIB1 and massspecgym (data/processed_data/<name>.pkl, data/splits/<name>/<split>.json), any
other dataset stored in the same layout (--dataset <name>), or a dataset anywhere on disk (--dataset <name>
--data_file records.pkl --split_dir dir/ with dir/<split>.json). Records: see utils/nn_lib.load_dataset; splits are
json {"train": [ids], "val": [ids], "test": [ids]}.

Metrics, per setting: Jaccard of the binary prediction (threshold 0.5) and cosine similarity of the raw prediction (no
threshold), on the full test split and on the formula-matched / unmatched subsets, next to MIST when a MIST run exists.

Usage (conda env `mist`: numpy, scipy, numba, matchms, torch, rdkit):
    python run_nn.py --dataset NPLIB1 --split scaffold                          # every setting
    python run_nn.py --all                                                      # built-in datasets x scaffold / random
    python run_nn.py --dataset massspecgym --split random --setting dreams_subformula_match cosine_no_formula_filter
    python run_nn.py --dataset NPLIB1 massspecgym --split scaffold --setting with_formula_filter
    python run_nn.py --dataset mydata --data_file /path/records.pkl --split_dir /path/splits --split fold0
    python run_nn.py --all --audit                                              # split properties (coverage, leakage, oracle)
    python run_nn.py --summarize                                                # tables -> results/nearest_neighbour/summary_*.md

Outputs in results/nearest_neighbour/<dataset>_<split>/ (see utils/nn_driver.py), tables in results/nearest_neighbour/
(summary_nn_<fp>.md: every setting; summary_<mode>_<fp>.md: per mode with subsets; audit_metrics_<fp>.md), and for
--fingerprint mist the predictions in the layout of the trained models under results/FP_prediction/nearest_neighbour/.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import nn_lib as nn
from utils import nn_driver as drv
from utils import audit_metrics

SPECTRAL_MODES = ("no_formula_filter", "with_formula_filter", "subformula_match")
SETTINGS = {f"{sim}_{mode}": (sim, mode) for mode in SPECTRAL_MODES for sim in drv.SIMILARITIES}
SETTINGS["subformula_only"] = ("subformula", "subformula_only")


def expand_settings(names):
    """[(name, sim, mode, explicit)] in run order (each mode reuses the caches of the previous one)."""
    chosen = {}
    for name in names:
        if name == "all":
            chosen.update({s: False for s in SETTINGS if s not in chosen})
        elif name in SPECTRAL_MODES:
            chosen.update({f"{sim}_{name}": False for sim in drv.SIMILARITIES if f"{sim}_{name}" not in chosen})
        else:
            chosen[name] = True
    return [(s, *SETTINGS[s], chosen[s]) for s in SETTINGS if s in chosen]


def dreams_available(dataset, split):
    return all((nn.DREAMS_FOLDER / dataset / split / f"{p}.pkl").exists() and (nn.MGF_FOLDER / dataset / split / f"{p}.mgf").exists()
               for p in ("train", "test"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", help=f"dataset name(s); built-in: {', '.join(nn.DATASETS)}")
    ap.add_argument("--split", nargs="+", help=f"split name(s), i.e. <split>.json; built-in: {', '.join(nn.SPLITS)}")
    ap.add_argument("--all", action="store_true", help="every built-in dataset x built-in split")
    ap.add_argument("--data_file", help="records of a dataset outside data/processed_data (one --dataset only)")
    ap.add_argument("--split_dir", help="folder with <split>.json for --data_file (default: data/splits/<dataset>)")
    ap.add_argument("--setting", nargs="+", default=["all"],
                    choices=["all", *SPECTRAL_MODES, *SETTINGS], metavar="SETTING",
                    help="all | a mode name | one or more of: " + ", ".join(SETTINGS))
    ap.add_argument("--fingerprint", choices=("mist", "dataset"), default="mist",
                    help="mist: RDKit Morgan r=2 / 4096 from SMILES (MIST's target); dataset: the stored morgan4_4096 bits")
    ap.add_argument("--tolerance", type=float, default=drv.DEFAULTS["tolerance"], help="peak match tolerance in Da (matchms default 0.1)")
    ap.add_argument("--mz_power", type=float, default=drv.DEFAULTS["mz_power"])
    ap.add_argument("--intensity_power", type=float, default=drv.DEFAULTS["intensity_power"])
    ap.add_argument("--max_peaks", type=int, default=drv.DEFAULTS["max_peaks"],
                    help="keep the N most intense peaks per spectrum for peak matching (0 = all)")
    ap.add_argument("--topk", type=int, default=drv.DEFAULTS["topk"], help="neighbours stored per test spectrum")
    ap.add_argument("--n_check", type=int, default=300, help="random pairs checked against matchms before a full search (0 = skip)")
    ap.add_argument("--n_threads", type=int, default=None, help="numba threads (default: all cores)")
    ap.add_argument("--n_workers", type=int, default=32, help="processes for sub-formula tokenisation")
    ap.add_argument("--audit", action="store_true", help="also compute split properties (coverage, leakage, fingerprint oracle)")
    ap.add_argument("--skip_oracle", action="store_true", help="with --audit: skip the O(test x train) oracle")
    ap.add_argument("--summarize", action="store_true", help="only (re)write the summary tables from existing results")
    a = ap.parse_args()

    if a.summarize:
        for mode in drv.MODES:
            drv.summarize(mode, a.fingerprint)
        drv.summarize_all(a.fingerprint)
        audit_metrics.summarize(a.fingerprint)
        return

    if a.all:
        jobs = [(d, s) for d in nn.DATASETS for s in nn.SPLITS]
    elif a.dataset and a.split:
        jobs = [(d, s) for d in a.dataset for s in a.split]
    else:
        ap.error("give --dataset and --split, --all, or --summarize")
    if a.data_file:
        if not a.dataset or len(a.dataset) != 1:
            ap.error("--data_file needs exactly one --dataset name")
        nn.register_dataset(a.dataset[0], a.data_file, a.split_dir or nn.SPLITS_FOLDER / a.dataset[0])
    for d, s in jobs:
        if not nn.data_path(d).exists():
            ap.error(f"no records for dataset {d!r} at {nn.data_path(d)} (use --data_file)")
        if not nn.split_path(d, s).exists():
            ap.error(f"no split {s!r} for dataset {d!r} at {nn.split_path(d, s)} (use --split_dir)")
    if a.n_threads:
        import numba
        numba.set_num_threads(a.n_threads)

    params = {"tolerance": a.tolerance, "mz_power": a.mz_power, "intensity_power": a.intensity_power,
              "max_peaks": a.max_peaks or None, "topk": a.topk}
    settings = expand_settings(a.setting)
    for d, s in jobs:
        if a.audit:
            audit_metrics.run(d, s, a.fingerprint, a.skip_oracle)
        for name, sim, mode, explicit in settings:
            if sim == "dreams" and not dreams_available(d, s):
                msg = (f"no DreaMS embeddings for {d}/{s} under {nn.DREAMS_FOLDER / d / s} (run "
                       f"utils/cache_dreams_embeddings.py in env `dreams`)")
                if explicit:
                    sys.exit(msg)
                print(f"skipping {name}: {msg}")
                continue
            drv.run(sim, mode, d, s, a.fingerprint, params, a.n_check, a.n_workers)
    for mode in drv.MODES:
        drv.summarize(mode, a.fingerprint)
    drv.summarize_all(a.fingerprint)
    if a.audit:
        audit_metrics.summarize(a.fingerprint)


if __name__ == "__main__":
    main()
