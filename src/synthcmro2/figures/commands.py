"""Commands of the figure part (``synthcmro2 figures | pls | benchmark | crop-atlas``).

figures     Figures 2-4 and S2: figure files, one CSV per statistics table, and a provenance record (input
            checksums, software versions, configuration) in ``paths.output_dir``
pls         mean-centered PLS of calculation versus rest (Figure 3b input; needs pyls)
benchmark   model errors versus the test-retest error of analytical CMRO2 (Table S2)
public-tables  shareable tables (group voxel maps, participant statistics) from the individual tables
crop-atlas  crop an atlas from the full 4-mm MNI grid to the training grid

Figures and benchmark run either on the individual tables of ``synthcmro2 model export`` or on the public
tables (group voxel maps), depending on ``config/figures.yaml``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import numpy as np

from .atlas import crop_to_mask_grid
from .benchmark import noise_benchmark
from .config import Config, load_config
from . import exclusion
from .data import check_metrics_match_voxels, is_group_table, load_participant_metrics, load_voxel_values
from .manuscript import FigureOutput, SurfaceContext, make_figure2, make_figure3, make_figure4, make_figure_s2
from .plotting import apply_style, save_figure
from .pls import run_pls
from .surfaces import voxel_parcel_ids

logger = logging.getLogger("synthcmro2.figures")

FIGURES = {"2": make_figure2, "3": make_figure3, "4": make_figure4, "S2": make_figure_s2}
PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "seaborn",
    "pyyaml",
    "nilearn",
    "nibabel",
]


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def provenance(cfg: Config, config_file: Path) -> dict:
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "config_file": str(config_file),
        "config": config_file.read_text(encoding="utf8"),
        "inputs": {str(p): sha256(p) for p in (cfg.participant_metrics, cfg.voxel_values, cfg.retest_voxel_values,
                                               cfg.parcel_icc, cfg.parcel_sd, cfg.age_calibration)
                   if p is not None and p.exists()},  # fmt: skip
    }


def write_outputs(name: str, output: FigureOutput, cfg: Config, dpi: float | None = None) -> None:
    """Save the figure and its tables; ``dpi`` (CLI) overrides plotting.dpi (null = Matplotlib default)."""
    out_dir = cfg.output_dir
    dpi = dpi if dpi is not None else cfg.plotting.get("dpi")
    paths = save_figure(output.figure, out_dir / "figures", name, cfg.plotting["formats"], dpi)
    for path in paths:
        logger.info("wrote %s", path)
    for table_name, table in output.tables.items():
        path = out_dir / "stats" / f"{table_name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=table.index.name is not None or table.index.nlevels > 1)
        logger.info("wrote %s", path)


def _setup_logging(log_file: Path, verbose: bool) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.WARNING,  # third-party libraries: warnings only
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, mode="w")],
        force=True,
    )
    logging.getLogger("synthcmro2").setLevel(logging.DEBUG if verbose else logging.INFO)


def _load_tables(cfg: Config):
    metrics = load_participant_metrics(cfg.participant_metrics)
    voxels = load_voxel_values(cfg.voxel_values)
    check_metrics_match_voxels(metrics, voxels, metrics_only=tuple(filter(None, [cfg.datasets.get("calibration")])))
    if cfg.exclude_parcels:
        logger.info("excluding Glasser parcels %s (both hemispheres)", ", ".join(cfg.exclude_parcels))
        if is_group_table(voxels):
            logger.warning("group voxel table: the parcels are removed from the group maps, but the participant "
                           "metrics cannot be recomputed without the individual voxel values")  # fmt: skip
            vox = exclusion.excluded_voxels(voxel_parcel_ids(cfg.surface), exclusion.labels_for(cfg))
            return metrics, exclusion.drop_voxels(voxels, vox)
        metrics, voxels = exclusion.apply(metrics, voxels, voxel_parcel_ids(cfg.surface), exclusion.labels_for(cfg))
    return metrics, voxels


def figures_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="synthcmro2 figures", description="Figures 2-4 and S2 with their statistics.")
    parser.add_argument("--config", type=Path, default=Path("config/figures.yaml"))
    parser.add_argument("--tag", help="export tag used in the config paths ({tag}); default: paths.tag")
    parser.add_argument("--exclude-parcels", nargs="+", metavar="NAME",
                        help="Glasser parcels to remove in both hemispheres (e.g. TE1m TE2a TE1p TE2p); overrides "
                        "analysis.exclude_parcels; outputs go to <output_dir>_excl-<names>")  # fmt: skip
    parser.add_argument("--figures", nargs="+", choices=sorted(FIGURES), help="default: all (S2 if configured)")
    parser.add_argument("--skip-surfaces", action="store_true")
    parser.add_argument("--dpi", type=float, help="resolution of raster outputs (PNG); default: plotting.dpi (300)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.tag, args.exclude_parcels)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(cfg.output_dir / "run.log", args.verbose)

    figures = args.figures or [f for f in sorted(FIGURES) if f != "S2" or cfg.has_retest or cfg.has_parcel_icc]
    if "S2" not in figures:
        logger.info("Figure S2 skipped: neither paths.retest_voxel_values nor paths.parcel_icc found")
    metrics, voxels = _load_tables(cfg)

    apply_style(cfg.plotting)
    ctx = None if args.skip_surfaces else SurfaceContext.load(cfg)
    for number in figures:
        name = f"Figure{number}"
        logger.info("building %s", name)
        np.random.seed(int(cfg.analysis["random_seed"]))  # seaborn's strip-plot jitter: identical figures per run
        write_outputs(name, FIGURES[number](metrics, voxels, cfg, ctx), cfg, args.dpi)
    with (cfg.output_dir / "provenance.json").open("w", encoding="utf8") as fh:
        json.dump(provenance(cfg, args.config.resolve()), fh, indent=2)
    return 0


def pls_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="synthcmro2 pls", description="Mean-centered PLS of calculation versus rest.")
    parser.add_argument("--config", type=Path, default=Path("config/figures.yaml"))
    parser.add_argument("--tag", help="export tag used in the config paths ({tag}); default: paths.tag")
    parser.add_argument("--out", type=Path, help="output folder (default: paths.pls_dir, where `figures` reads it)")
    parser.add_argument("--seed", type=int, default=None, help="default: analysis.random_seed")
    parser.add_argument("--exclude-parcels", nargs="+", metavar="NAME",
                        help="Glasser parcels to remove in both hemispheres (e.g. TE1m TE2a TE1p TE2p); overrides "
                        "analysis.exclude_parcels; outputs go to <output_dir>_excl-<names>")  # fmt: skip
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.tag, args.exclude_parcels)
    out = args.out or cfg.pls_dir
    _setup_logging(out / "run.log", False)
    seed = cfg.analysis["random_seed"] if args.seed is None else args.seed
    voxels = load_voxel_values(cfg.voxel_values)
    if is_group_table(voxels):
        parser.error("PLS needs the individual voxel table of `synthcmro2 model export`; the public data hold the "
                     "PLS results instead (paths.pls_dir)")  # fmt: skip
    if cfg.exclude_parcels:
        vox = exclusion.excluded_voxels(voxel_parcel_ids(cfg.surface), exclusion.labels_for(cfg))
        logger.info("excluding %s: %d voxels removed from the PLS input", ", ".join(cfg.exclude_parcels), vox.size)
        voxels = exclusion.drop_voxels(voxels, vox)
    prov = run_pls(cfg, voxels, out, seed)
    for tag, rec in prov["analyses"].items():
        logger.info("%s: LV1 p = %.4f -> %s", tag, rec["p_lv1"], out / rec["file"])
    return 0


def benchmark_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="synthcmro2 benchmark", description="Errors vs test-retest error (Table S2).")
    parser.add_argument("--config", type=Path, default=Path("config/figures.yaml"))
    parser.add_argument("--tag", help="export tag used in the config paths ({tag}); default: paths.tag")
    parser.add_argument("--exclude-parcels", nargs="+", metavar="NAME",
                        help="Glasser parcels to remove in both hemispheres (e.g. TE1m TE2a TE1p TE2p); overrides "
                        "analysis.exclude_parcels; outputs go to <output_dir>_excl-<names>")  # fmt: skip
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.tag, args.exclude_parcels)
    _setup_logging(cfg.output_dir / "benchmark.log", False)
    metrics, voxels = _load_tables(cfg)
    import pandas as pd

    parcel_sd = pd.read_csv(cfg.parcel_sd) if cfg.parcel_sd is not None and cfg.parcel_sd.exists() else None
    table = noise_benchmark(metrics, voxels, voxel_parcel_ids(cfg.surface), cfg, parcel_sd)
    path = cfg.output_dir / "stats" / "noise_benchmark.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    cols = ["figure", "dataset", "model", "level", "n_participants", "mae_median", "retest_within_session",
            "retest_between_day", "mae_over_retest_between"]  # fmt: skip
    logger.info("wrote %s\n%s", path, table[cols].round(2).to_string(index=False))
    return 0


def public_tables_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="synthcmro2 public-tables",
                                     description="Shareable tables from the individual tables (see public.py).")  # fmt: skip
    parser.add_argument("--config", type=Path, default=Path("config/figures.yaml"),
                        help="configuration that reads the INDIVIDUAL tables and has a `public` section")  # fmt: skip
    parser.add_argument("--tag", help="export tag used in the config paths ({tag}); default: paths.tag")
    parser.add_argument("--exclude-parcels", nargs="+", metavar="NAME",
                        help="Glasser parcels removed before the tables are written (as for the figures)")
    parser.add_argument("--out", type=Path, required=True, help="output folder, e.g. data/public")
    args = parser.parse_args(argv)
    from .public import write_public_tables

    cfg = load_config(args.config, args.tag, args.exclude_parcels)
    _setup_logging(cfg.output_dir / "public_tables.log", False)
    metrics, voxels = _load_tables(cfg)
    if is_group_table(voxels):
        parser.error("public-tables needs the individual voxel table (paths.voxel_values)")
    for path in write_public_tables(metrics, voxels, cfg, args.out):
        logger.info("wrote %s", path)
    return 0


def crop_atlas_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="synthcmro2 crop-atlas", description="Crop an atlas to the training grid.")
    parser.add_argument("image")
    parser.add_argument("reference_mask")
    parser.add_argument("output")
    args = parser.parse_args(argv)
    before, after = crop_to_mask_grid(args.image, args.reference_mask, args.output)
    print(f"{args.image} {before} -> {args.output} {after}")
    return 0
