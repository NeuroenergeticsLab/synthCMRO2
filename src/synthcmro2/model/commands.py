"""Command line: ``synthcmro2 model --config config/model.yaml {check,train,select,predict,export}``.

check    validate the configuration and that every input file exists (no PyTorch needed)
train    cross-validated training of one or more models (refuses to overwrite checkpoints)
select   evaluate the saved folds on their validation participants; write cv_selection.csv
predict  apply each model's selected fold to the test datasets; write NIfTI maps and test_metrics.csv
apply    synthesize CMRO2 for one participant from new input maps (no analytical CMRO2 needed)
export   build the figure tables (synthcmro2_stats_df, cmro2_vox_df) from the predicted maps (no PyTorch needed)

Each command writes a log and a provenance record (software versions, configuration, input checksums of
the checkpoints used) to <results_dir>/<command>/.
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

import pandas as pd

from .config import Config, load_config
from .data import check_grid, grid, load_mask, load_masked, map_path
from .folds import cohort_folds

logger = logging.getLogger("synthcmro2.model")
PACKAGES = ["torch", "monai", "pytorch-msssim", "numpy", "scipy", "scikit-learn", "nibabel", "pandas", "pyyaml"]


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def provenance(config_file: Path, argv: list[str], files: list[Path]) -> dict:
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    gpu = None
    if "torch" in sys.modules:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": argv,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu": gpu,
        "packages": versions,
        "config_file": str(config_file),
        "config": config_file.read_text(encoding="utf8"),
        "files": {str(p): sha256(p) for p in files if p.exists()},
    }


def check(cfg: Config) -> list[str]:
    """Return a list of problems (missing files, maps off the mask grid); empty if the inputs are complete.

    Maps whose grid origin differs from the mask's within ``grid.tolerance_mm`` are listed (INFO): they are
    read voxel by voxel, without resampling.
    """
    import nibabel as nib

    problems = []
    if not cfg.gm_mask.exists():
        return [f"missing gray matter mask {cfg.gm_mask}"]
    mask = load_mask(cfg.gm_mask)
    n_folds, seed = int(cfg.training["n_folds"]), int(cfg.training["seed"])
    for cohort in cfg.cohorts.values():
        try:
            cohort_folds(cohort, n_folds, seed)
        except RuntimeError as err:
            problems.append(str(err))
    needed: set[tuple[Path, str, str]] = set()
    for model in cfg.models.values():
        for sid in cfg.cohorts[model.cohort].used_ids:
            for key in (*model.inputs, "CMRO2"):
                needed.add((map_path(cfg, cfg.training_dir, sid, key), "training", sid))
    for cohort in cfg.cohorts.values():
        logger.info("cohort %s: %d training participants used (%d excluded)", cohort.name, len(cohort.used_ids),
                    len(cohort.train_ids) - len(cohort.used_ids))  # fmt: skip
    for ds in cfg.datasets.values():
        logger.info("dataset %s: %d participants, models %s%s", ds.key, len(ds.ids), ", ".join(ds.models),
                    " (out of fold)" if ds.out_of_fold else "")  # fmt: skip
        for mkey in ds.models:
            for sid in ds.ids:
                for key in (*cfg.models[mkey].inputs, "CMRO2"):
                    needed.add((map_path(cfg, ds.data_dir, sid, key, ds.condition), ds.key, sid))
        if ds.export_models and ds.participants_info and not ds.participants_info.exists():
            problems.append(f"{ds.key}: missing participants table {ds.participants_info}")
    ref, offset_ids = grid(cfg), {}
    targets = {map_path(cfg, cfg.training_dir, sid, "CMRO2") for c in cfg.cohorts.values() for sid in c.used_ids}
    for path, where, sid in sorted(needed):
        if not path.exists():
            problems.append(f"{where}: missing {path}")
            continue
        try:
            img = nib.load(str(path))
            if check_grid(img, path, mask.shape, ref) > 1e-6:
                offset_ids.setdefault(where, set()).add(sid)
            if path in targets:  # a training target without signal would teach the model to output zeros
                n_pos = int((load_masked(path, mask, ref)[mask] > 0).sum())
                if n_pos < 3:
                    problems.append(f"{where}: {sid} reference CMRO2 has {n_pos} positive gray matter voxels; "
                                    "add it to the cohort's exclude_ids")  # fmt: skip
        except ValueError as err:
            problems.append(f"{where}: {err}")
    for where, ids in sorted(offset_ids.items()):
        logger.info("%s: grid origin off the mask's by <= %.2g mm (read without resampling) for %d participant(s): %s",
                    where, ref[1], len(ids), ", ".join(sorted(ids)))  # fmt: skip
    return problems


def parse_args(argv):
    p = argparse.ArgumentParser(prog="synthcmro2 model", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("--config", type=Path, default=Path("config/model.yaml"))
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    t = sub.add_parser("train")
    t.add_argument("--models", nargs="+", required=True, help="model keys from the config, e.g. cbf_t2s")
    t.add_argument("--folds", nargs="+", type=int)
    t.add_argument("--epochs", type=int, help="override (for smoke tests)")
    t.add_argument("--overwrite", action="store_true")
    t.add_argument("--device", default="cuda:0")
    s = sub.add_parser("select")
    s.add_argument("--models", nargs="+")
    s.add_argument("--out", type=Path, help="default: paths.selection_dir")
    s.add_argument("--device", default="cpu")
    r = sub.add_parser("predict")
    r.add_argument("--datasets", nargs="+")
    r.add_argument("--models", nargs="+")
    r.add_argument("--fold", nargs="+", metavar="MODEL=FOLD", default=[], help="override the selected fold")
    r.add_argument("--device", default="cpu")
    r.add_argument(
        "--resume", action="store_true", help="skip dataset/model pairs already complete in test_metrics.csv"
    )
    a = sub.add_parser("apply", help="synthetic CMRO2 from new input maps (e.g. CBF and T2*)")
    a.add_argument("--model", default="cbf_t2s", help="model key from the config (default: cbf_t2s)")
    a.add_argument("--input", nargs="+", required=True, metavar="KEY=PATH",
                   help="input maps on the 4-mm gray matter mask grid, e.g. CBF=sub-01_cbf.nii.gz T2S=sub-01_t2s.nii.gz")
    a.add_argument("--out", type=Path, required=True, help="output NIfTI (synthetic CMRO2, µmol/100 g/min)")
    a.add_argument("--checkpoint", type=Path, help="generator .pth (default: the selected fold under paths.models_dir)")
    a.add_argument("--device", default="cpu")
    e = sub.add_parser("export")
    e.add_argument("--datasets", nargs="+")
    e.add_argument("--models", nargs="+", help="restrict to these models (default: export_models of each dataset)")
    e.add_argument("--tag", default=datetime.now().strftime("%Y%m%d"), help="file name suffix (default: today)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    cfg = load_config(args.config)
    out = cfg.results_dir / args.command
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out / "run.log", mode="a")],
    )
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    if args.command == "apply":
        if args.model not in cfg.models:
            raise SystemExit(f"unknown model {args.model!r}; available: {', '.join(cfg.models)}")
        import torch

        from .apply import apply_model

        device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
        inputs = dict(item.split("=", 1) for item in args.input)
        apply_model(cfg, cfg.models[args.model], {k: Path(v) for k, v in inputs.items()}, args.out, args.checkpoint,
                    device)  # fmt: skip
        return 0
    for key in getattr(args, "models", None) or []:
        if key not in cfg.models:
            raise SystemExit(f"unknown model {key!r}; available: {', '.join(cfg.models)}")
    for key in getattr(args, "datasets", None) or []:
        if key not in cfg.datasets:
            raise SystemExit(f"unknown dataset {key!r}; available: {', '.join(cfg.datasets)}")

    used: list[Path] = [cfg.gm_mask]
    if args.command == "check":
        problems = check(cfg)
        for line in problems:
            logger.error(line)
        logger.info("check: %s", "OK" if not problems else f"{len(problems)} problems")
        return 1 if problems else 0
    if args.command == "export":
        from .export import export_tables

        stats_df, vox_df = export_tables(cfg, load_mask(cfg.gm_mask), args.datasets, args.models)
        for name, table in (("synthcmro2_stats_df", stats_df), ("cmro2_vox_df", vox_df)):
            path = out / f"{name}_{args.tag}.csv"
            table.to_csv(path, index=False)
            logger.info("wrote %s (%d rows)", path, len(table))
        logger.info("\n%s", stats_df.groupby(["dataset", "condition", "model"]).size().to_string())
        (out / "provenance.json").write_text(json.dumps(provenance(args.config.resolve(), argv, used), indent=2))
        return 0

    import torch

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    mask = load_mask(cfg.gm_mask)
    if args.command == "train":
        from .train import train_model

        for key in args.models:
            done = train_model(cfg, cfg.models[key], mask, device, args.folds, args.epochs, args.overwrite)
            used += list(done.values())
    elif args.command == "select":
        from .validate import select

        cv = select(cfg, mask, device, args.models, args.out)
        logger.info("\n%s", cv.round(3).to_string(index=False))
        used += [
            cfg.checkpoint("generator", cfg.models[k], f)
            for k in args.models or cfg.models
            for f in range(1, int(cfg.training["n_folds"]) + 1)
        ]
    else:
        from .predict import predict

        folds = {k: int(v) for k, v in (item.split("=") for item in args.fold)}
        path = out / "test_metrics.csv"  # updated after every dataset/model pair
        table = predict(cfg, mask, device, args.datasets, args.models, folds, metrics_file=path, resume=args.resume)
        logger.info("wrote %s", path)
        if table.empty:
            table = pd.DataFrame(columns=["model", "fold"])
        used += [
            cfg.checkpoint("generator", cfg.models[m], f)
            for m, f in table[["model", "fold"]].drop_duplicates().to_numpy()
        ]
    (out / "provenance.json").write_text(json.dumps(provenance(args.config.resolve(), argv, used), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
