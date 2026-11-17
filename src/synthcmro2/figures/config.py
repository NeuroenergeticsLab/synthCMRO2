"""Load and validate the figure configuration (config/figures.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SurfacePaths:
    """Optional atlas and surface files needed for the cortical surface panels."""

    gm_mask: Path | None = None
    glasser_atlas: Path | None = None
    glasser_label_gii: Path | None = None
    inflated_surf_gii: Path | None = None

    @property
    def available(self) -> bool:
        """True when every surface input is set and exists on disk."""
        paths = (self.gm_mask, self.glasser_atlas, self.glasser_label_gii, self.inflated_surf_gii)
        return all(p is not None and p.exists() for p in paths)


@dataclass(frozen=True)
class Config:
    """Typed view of the YAML configuration."""

    participant_metrics: Path
    voxel_values: Path
    output_dir: Path
    surface: SurfacePaths
    datasets: dict[str, str]
    dataset_labels: dict[str, str]
    models: dict[str, str]
    minimal_model: str
    alternative_input_models: list[str]
    alternative_input_baseline: str
    metrics: dict[str, str]
    analysis: dict[str, Any]
    pls: dict[str, Any]
    pls_dir: Path
    gm_mask: Path | None
    brain_background: Path | None
    plotting: dict[str, Any] = field(default_factory=dict)
    retest_voxel_values: Path | None = None  # voxel table of the test-retest scans (Figure S2; optional)
    parcel_icc: Path | None = None  # precomputed parcel ICCs (Figure S2 without the test-retest voxel table)
    parcel_sd: Path | None = None  # between-participant SD of parcel medians (Table S2 parcel level, group data)
    age_calibration: Path | None = None  # fitted age calibration (Figure 4b without training-set ages)
    public: dict[str, Any] = field(default_factory=dict)  # rules of `synthcmro2 public-tables`
    reliability: dict[str, Any] = field(default_factory=dict)
    exclude_parcels: tuple[str, ...] = ()  # Glasser parcels removed in both hemispheres (sensitivity analysis)

    def pls_file(self, tag: str) -> Path:
        """Path of the stored PLS result for a model tag (or the analytical tag)."""
        return self.pls_dir / self.pls["file_pattern"].format(tag=tag)

    @property
    def can_render_slices(self) -> bool:
        return self.gm_mask is not None and self.gm_mask.exists()

    @property
    def has_retest(self) -> bool:
        """True when the test-retest voxel table (Figure S2) is configured and exists."""
        return self.retest_voxel_values is not None and self.retest_voxel_values.exists()

    @property
    def has_parcel_icc(self) -> bool:
        """True when precomputed parcel ICCs (Figure S2 from public data) are configured and exist."""
        return self.parcel_icc is not None and self.parcel_icc.exists()

    @property
    def model_order(self) -> list[str]:
        """Model identifiers in figure order."""
        return list(self.models)

    def model_label(self, model: str) -> str:
        return self.models.get(model, model)


def _resolve_path(root: Path, value: str | None, tag: str = "") -> Path | None:
    if not value:
        return None
    if "{tag}" in value:
        if not tag:
            raise ValueError(f"path {value!r} uses {{tag}}, but paths.tag is not set (or pass --tag)")
        value = value.replace("{tag}", tag)
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def load_config(config_file: str | Path, tag: str | None = None, exclude_parcels=None) -> Config:
    """Read the YAML file and resolve relative paths against the project root.

    The project root is the parent of the directory containing the config file
    (that is, ``<root>/config/figures.yaml``). ``{tag}`` in a path is replaced by ``tag`` or, if not given,
    by ``paths.tag`` (the tag of ``synthcmro2 model export --tag``). ``exclude_parcels`` overrides
    ``analysis.exclude_parcels``; with parcels to exclude, ``output_dir`` and ``pls_dir`` get the suffix
    ``_excl-<names>`` so the main results are not overwritten.
    """
    config_file = Path(config_file).resolve()
    root = config_file.parent.parent
    with config_file.open(encoding="utf8") as fh:
        raw = yaml.safe_load(fh)

    paths = raw["paths"]
    tag = str(tag if tag is not None else paths.get("tag") or "")

    def resolve(value: str | None) -> Path | None:
        return _resolve_path(root, value, tag)

    if exclude_parcels is None:
        exclude_parcels = (raw.get("analysis") or {}).get("exclude_parcels") or []
    exclude_parcels = tuple(str(n) for n in exclude_parcels)
    excl = "_excl-" + "-".join(exclude_parcels) if exclude_parcels else ""

    gm_mask = resolve(paths.get("gm_mask"))
    surface_raw = paths.get("surface") or {}
    surface = SurfacePaths(gm_mask=gm_mask, **{k: resolve(v) for k, v in surface_raw.items()})

    config = Config(
        participant_metrics=resolve(paths["participant_metrics"]),
        voxel_values=resolve(paths["voxel_values"]),
        output_dir=resolve(paths["output_dir"] + excl),
        surface=surface,
        datasets=raw["datasets"],
        dataset_labels=raw["dataset_labels"],
        models=raw["models"],
        minimal_model=raw["minimal_model"],
        alternative_input_models=raw["alternative_input_models"],
        alternative_input_baseline=raw["alternative_input_baseline"],
        metrics=raw["metrics"],
        analysis=raw["analysis"],
        pls=raw["pls"],
        pls_dir=resolve(paths["pls_dir"] + excl),
        gm_mask=gm_mask,
        brain_background=resolve(paths.get("brain_background")),
        plotting=raw["plotting"],
        retest_voxel_values=resolve(paths.get("retest_voxel_values")),
        parcel_icc=resolve(paths.get("parcel_icc")),
        parcel_sd=resolve(paths.get("parcel_sd")),
        age_calibration=resolve(paths.get("age_calibration")),
        public=raw.get("public") or {},
        reliability=raw.get("reliability") or {},
        exclude_parcels=exclude_parcels,
    )
    _check_config(config)
    return config


def _check_config(config: Config) -> None:
    required_roles = {"test", "task", "fmri", "replication", "reproducibility"}
    missing = required_roles - set(config.datasets)
    if missing:
        raise ValueError(f"config 'datasets' is missing roles: {sorted(missing)}")
    if config.minimal_model not in config.models:
        raise ValueError(f"minimal_model '{config.minimal_model}' is not listed under 'models'")
    unknown_pls = set(config.pls["model_tags"]) - set(config.models)
    if unknown_pls:
        raise ValueError(f"pls.model_tags not listed under 'models': {sorted(unknown_pls)}")
    unknown = set(config.alternative_input_models) - set(config.models)
    if unknown:
        raise ValueError(f"alternative_input_models not listed under 'models': {sorted(unknown)}")
