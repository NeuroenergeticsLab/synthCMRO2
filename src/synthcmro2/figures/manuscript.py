"""Compose manuscript Figures 2-4 and the statistics reported with them.

Each ``make_figure*`` function returns a :class:`FigureOutput` holding the
matplotlib figure, the statistics tables behind every number on it, and the
group maps needed for the optional surface panels. Writing to disk is left to
the caller (``cli.py``).

Figure 1 (methods schematic) is not data-driven and is not generated here.
Figure 3b uses the stored PLS results (see ``pls.py``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from . import plotting as P
from . import reliability as R
from .config import Config
from .data import group_map, is_group_table, load_voxel_values, paired_by_subject, select
from .exclusion import expand, labels_for
from .pls import load_pls_result, match_observations, orientation_sign, overlap, plot_bsr_slice
from .stats import (
    Agreement,
    age_bias_calibration,
    age_slopes,
    age_slopes_adjusted,
    agreement,
    apply_age_calibration,
    correlation,
    fdr_bh,
    games_howell,
    paired_difference_summary,
    pairwise_wilcoxon,
    significance_stars,
)
from .surfaces import (
    Surface,
    load_surface,
    parcel_coverage,
    parcel_medians,
    plot_surface,
    shared_norm,
    truncated_cmap,
    vertex_values,
    voxel_parcel_ids,
)

logger = logging.getLogger(__name__)

FIG_WIDTH = 7.2  # inches, two-column width
FIG4_WIDTH = 6.2  # Figure 4 is narrower: two columns of single scatter plots


@dataclass
class FigureOutput:
    figure: Figure
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    group_maps: dict[str, pd.Series] = field(default_factory=dict)


CBAR_SIZE_IN = (1.3, 0.09)  # one physical size for every map colorbar (width, height in inches)


@dataclass(frozen=True)
class SurfaceContext:
    """Everything needed to draw group maps on the cortical surface (loaded once)."""

    surface: Surface
    parcel_ids: np.ndarray
    coverage: pd.DataFrame  # per parcel: voxels in atlas and mask, coverage, shown or not
    cmap: object  # matplotlib Colormap
    percentiles: tuple[float, float]
    ambient: float
    units: str

    @classmethod
    def load(cls, cfg: Config) -> SurfaceContext | None:
        if not cfg.surface.available:
            logger.warning("surface inputs not configured or missing: maps are omitted")
            return None
        s = cfg.plotting["surface"]
        coverage = parcel_coverage(cfg.surface)
        coverage["shown"] = (coverage["coverage"] >= s["min_coverage"]) & (
            coverage["n_mask_voxels"] >= s["min_voxels"]
        )
        logger.info(
            "surface maps: %d of %d parcels shown in gray (coverage < %.2f or < %d mask voxels)",
            (~coverage["shown"]).sum(),
            len(coverage),
            s["min_coverage"],
            s["min_voxels"],
        )
        return cls(
            load_surface(cfg.surface),
            voxel_parcel_ids(cfg.surface),
            coverage,
            truncated_cmap(s["cmap"], *s.get("cmap_range", (0.0, 1.0))),
            tuple(s["percentiles"]),
            s.get("ambient", 0.6),
            cfg.plotting["units"],
        )

    def parcels(self, values: pd.Series) -> np.ndarray:
        """Parcel medians; parcels the mask barely covers are NaN (drawn gray)."""
        out = np.array(parcel_medians(values, self.parcel_ids), dtype=float)  # writable copy
        out[~self.coverage["shown"].to_numpy()] = np.nan
        return out

    def norm_from(self, analytical: pd.Series):
        """Color limits from the analytical map; reused for the synthetic maps it is compared with."""
        return shared_norm(self.parcels(analytical), self.percentiles)

    def draw(self, ax, values: pd.Series, norm, view: str = "lateral") -> None:
        plot_surface(
            ax,
            self.surface,
            vertex_values(self.parcels(values), self.surface.labels),
            cmap=self.cmap,
            norm=norm,
            view=view,
            ambient=self.ambient,
        )

    def vcolorbar(self, sub, ax, norm) -> None:
        """Vertical colorbar right of ``ax``, as long as the horizontal ones (CBAR_SIZE_IN)."""
        sm = plt.cm.ScalarMappable(norm=norm, cmap=self.cmap)
        fs = plt.rcParams["font.size"]
        cb = sub.colorbar(sm, ax=ax, location="right", shrink=0.55, aspect=14, pad=0.03, extend="both")
        cb.ax.tick_params(labelsize=fs - 2, length=2, pad=1)
        cb.set_label(f"CMRO₂ ({self.units})", fontsize=fs - 2, labelpad=2)

    def hcolorbar(self, sub, norm, x_center: float = 0.5, y: float = 0.5) -> None:
        """Horizontal colorbar of fixed physical size, centered at (x_center, y) of ``sub``.

        The same size in every figure makes clear that the colorbar is a map
        scale, not an extension of a scatter plot axis.
        """
        width_in, height_in = CBAR_SIZE_IN
        sub_w = sub.bbox.width / sub.figure.dpi
        sub_h = sub.bbox.height / sub.figure.dpi
        w, h = width_in / sub_w, height_in / sub_h
        cax = sub.add_axes([x_center - w / 2, y - h / 2, w, h])
        sm = plt.cm.ScalarMappable(norm=norm, cmap=self.cmap)
        cb = sub.colorbar(sm, cax=cax, orientation="horizontal", extend="both")
        fs = plt.rcParams["font.size"]
        cb.ax.tick_params(labelsize=fs - 2, length=2, pad=1)
        cb.set_label(f"CMRO₂ ({self.units})", fontsize=fs - 1, labelpad=2)


def _map_column(sub, ctx: SurfaceContext, analytical: pd.Series, title: str, note: str = "", medial: bool = True):
    """Analytical group map (left hemisphere, lateral view above medial view) in a narrow column, colorbar below.

    Placed to the left of the scatter plots, so the figure reads from the
    analytical reference to the synthetic maps (scatter insets) without an
    extra row. Maps and colorbar are positioned explicitly in the column.
    Returns the color normalization shared with the insets.
    """
    norm = ctx.norm_from(analytical)
    fs = plt.rcParams["font.size"]
    if medial:
        ax_map = sub.add_axes([0.02, 0.60, 0.96, 0.22])
        ax_med = sub.add_axes([0.02, 0.36, 0.96, 0.22])
        ctx.draw(ax_med, analytical, norm, view="medial")
        sub.map_medial_axes = ax_med
        y_title, y_cbar, y_note = 0.83, 0.30, 0.20
    else:
        ax_map = sub.add_axes([0.02, 0.44, 0.96, 0.22])
        y_title, y_cbar, y_note = 0.67, 0.40, 0.30
    sub.map_axes = ax_map  # lettered by the caller
    ctx.draw(ax_map, analytical, norm)
    ctx.hcolorbar(sub, norm, x_center=0.5, y=y_cbar)
    sub.text(0.5, y_title, title, ha="center", va="bottom", fontsize=fs)
    if note:
        sub.text(0.5, y_note, note, ha="center", va="top", fontsize=fs - 1, color="0.35")
    return norm


def _map_inset(ax, ctx: SurfaceContext, values: pd.Series, norm, where: str = "lower right") -> None:
    """Group map as an inset of a scatter plot (lower right, or upper left)."""
    bounds = {
        "lower right": [0.5, 0.02, 0.5, 0.38],  # synthetic, Fig. 2 (no label)
        "lower right labeled": [0.5, 0.09, 0.5, 0.34],  # room for a label below
        "upper left labeled": [0.0, 0.57, 0.5, 0.34],  # room for a label above
    }[where]
    ctx.draw(ax.inset_axes(bounds), values, norm)


def _scatter_with_maps(ax, ctx: SurfaceContext, gmap: pd.DataFrame):
    """Analytical (upper left) and synthetic (lower right) maps as insets of one scatter plot."""
    norm = ctx.norm_from(gmap["CMRO2"])
    _map_inset(ax, ctx, gmap["CMRO2"], norm, where="upper left labeled")
    _map_inset(ax, ctx, gmap["synthCMRO2"], norm, where="lower right labeled")
    label = {"fontsize": 6, "transform": ax.transAxes, "ha": "center", "path_effects": P._HALO}
    ax.text(0.25, 0.92, "analytical", va="bottom", **label)  # header above the surface
    ax.text(0.75, 0.08, "synthetic", va="top", **label)  # header below the surface
    return norm


def _group_agreement(
    voxels: pd.DataFrame, cfg: Config, dataset: str, model: str
) -> tuple[pd.DataFrame, dict]:
    """Group map plus agreement statistics for one dataset and model."""
    a = cfg.analysis
    gmap = group_map(voxels, dataset, model, a["positive_values_only"], a["group_summary"])
    res = agreement(
        gmap["CMRO2"],
        gmap["synthCMRO2"],
        a["n_bootstrap"],
        a["ci_level"],
        rng=np.random.default_rng(a["random_seed"]),
    )
    row = {
        "dataset": dataset,
        "model": model,
        "level": "group",
        "summary": f"voxel-wise {a['group_summary']} across participants",
        "n_subjects_max": int(gmap["n_subjects"].max()),
        **res.as_dict(),
    }
    return gmap, row


def _scatter_kwargs(cfg: Config) -> dict:
    p = cfg.plotting
    return {
        "units": p["units"],
        "point_size": p["scatter_point_size"],
        "point_color": p["scatter_color"],
        "line_color": p["line_color"],
    }


# --------------------------------------------------------------------------
# Figure 2: input reduction at rest
# --------------------------------------------------------------------------


def make_figure2(
    metrics: pd.DataFrame, voxels: pd.DataFrame, cfg: Config, ctx: SurfaceContext | None = None
) -> FigureOutput:
    test, fmri = cfg.datasets["test"], cfg.datasets["fmri"]
    model_dataset = [(m, fmri if m == "fMRI" else test) for m in cfg.model_order]

    # (a) on top, full width: analytical map column + 2 x 3 scatter grid; (b) below
    fig = plt.figure(figsize=(FIG_WIDTH, 6.6), layout="constrained")
    left, right = fig.subfigures(2, 1, height_ratios=[2.1, 1])
    if ctx:
        map_col, grid = left.subfigures(1, 2, width_ratios=[0.8, 3])
    else:
        map_col, grid = None, left
    scatter_axes = grid.subplots(2, 3).ravel()
    violin_axes = right.subplots(1, 3)

    # (a) group level
    rows, maps = [], {}
    limits = cfg.plotting["axis_limits"]["test"]
    for i, (ax, (model, dataset)) in enumerate(zip(scatter_axes, model_dataset, strict=True)):
        gmap, row = _group_agreement(voxels, cfg, dataset, model)
        rows.append(row)
        maps[f"synthetic_{dataset}_{model}"] = gmap["synthCMRO2"]
        if model == cfg.model_order[0]:
            maps[f"analytical_{test}"] = gmap["CMRO2"]
        P.agreement_scatter(
            ax,
            gmap["CMRO2"],
            gmap["synthCMRO2"],
            _to_agreement(row),
            limits,
            title=f"{cfg.model_label(model)} model",
            show_ylabel=i % 3 == 0,
            show_xlabel=i >= 3,
            **_scatter_kwargs(cfg),
        )
    if ctx:
        norm = _map_column(
            map_col,
            ctx,
            maps[f"analytical_{test}"],
            "Analytical CMRO₂",
            note="Insets:\nsynthetic CMRO₂,\nsame color scale",
            medial=cfg.plotting["surface"].get("medial_view", True),
        )
        for ax, (model, dataset) in zip(scatter_axes, model_dataset, strict=True):
            _map_inset(ax, ctx, maps[f"synthetic_{dataset}_{model}"], norm)
    letter = P.Letters()
    if ctx:
        letter(map_col.map_axes)
    for ax in scatter_axes:
        letter(ax)
    P.panel_label(left, "", "Group level: test set at rest")

    # (b) individual level
    individual = pd.concat(
        [
            select(metrics, dataset=test, model=[m for m in cfg.model_order if m != "fMRI"]),
            select(metrics, dataset=fmri, model="fMRI"),
        ]
    )
    for ax, (metric, label) in zip(violin_axes, cfg.metrics.items(), strict=True):
        P.metric_violins(
            ax,
            individual,
            metric,
            cfg.model_order,
            [cfg.model_label(m) for m in cfg.model_order],
            label,
        )
    for ax in violin_axes:  # one letter per individual-level metric
        letter(ax)
    P.panel_label(right, "", "Individual level")

    tables = {
        "fig2a_group_agreement": pd.DataFrame(rows),
        "fig2b_model_comparisons": _model_comparisons(metrics, cfg),
        "fig2b_alternative_inputs": _alternative_inputs(metrics, cfg),
        "fig2b_individual_summary": _individual_summary(individual, cfg),
    }
    return FigureOutput(fig, tables, maps)


def _to_agreement(row: dict) -> Agreement:
    return Agreement(**{k: row[k] for k in Agreement.__dataclass_fields__})


def _model_comparisons(metrics: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Pairwise Wilcoxon tests between all test-set models, per metric (FDR within metric)."""
    models = [m for m in cfg.model_order if m != "fMRI"]
    out = []
    for metric in cfg.metrics:
        wide = paired_by_subject(metrics, metric, "model", models, dataset=cfg.datasets["test"])
        out.append(pairwise_wilcoxon(wide).assign(metric=metric))
    return pd.concat(out, ignore_index=True)


def _alternative_inputs(metrics: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Single-input T1-weighted and fMRI models versus the CBF model (paired by subject)."""
    base = cfg.alternative_input_baseline
    datasets = [cfg.datasets["test"], cfg.datasets["fmri"]]
    rows = []
    for alt in cfg.alternative_input_models:
        for metric in cfg.metrics:
            wide = paired_by_subject(metrics, metric, "model", [alt, base], dataset=datasets)
            rows.append(
                {
                    "model": alt,
                    "baseline": base,
                    "metric": metric,
                    **paired_difference_summary(wide[alt], wide[base], cfg.analysis["sd_ddof"]),
                }
            )
    return pd.DataFrame(rows)


def _individual_summary(individual: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return (
        individual.groupby(["dataset", "condition", "model"], observed=True)[list(cfg.metrics)]
        .agg(["count", "mean", "std", "median"])
        .round(4)
    )


# --------------------------------------------------------------------------
# Figure 3: task sensitivity
# --------------------------------------------------------------------------


TASK_MODELS = ["reference", "wo CBV", "CBF & T2*", "CBF"]


def make_figure3(
    metrics: pd.DataFrame, voxels: pd.DataFrame, cfg: Config, ctx: SurfaceContext | None = None
) -> FigureOutput:
    test, task = cfg.datasets["test"], cfg.datasets["task"]
    colors = cfg.plotting["condition_colors"]
    task_subjects = select(metrics, dataset=task)["subj_id"].unique().tolist()
    data = select(metrics, dataset=[test, task], model=TASK_MODELS, subj_id=task_subjects)

    comparisons = _rest_vs_task(data, cfg)

    fig = plt.figure(figsize=(FIG_WIDTH * 0.8, 6.0), layout="constrained")
    left, right = fig.subfigures(1, 2, width_ratios=[1, 1.25])
    violin_axes = left.subplots(3, 1, sharex=True)
    sub_b, sub_c = right.subfigures(2, 1, height_ratios=[1, 1.3])
    if ctx:
        scatter_c, cbar_c = sub_c.subfigures(2, 1, height_ratios=[1, 0.12])
    else:
        scatter_c, cbar_c = sub_c, None
    ax_c = scatter_c.subplots()

    # (a) individual level, rest versus calculation
    for i, (ax, (metric, label)) in enumerate(zip(violin_axes, cfg.metrics.items(), strict=True)):
        p_fdr = comparisons.query("metric == @metric").set_index("model").loc[TASK_MODELS, "p_fdr"]
        P.split_violins(
            ax,
            data,
            metric,
            "model",
            TASK_MODELS,
            "condition",
            ["rest", "calc"],
            [colors["light"], colors["dark"]],
            label,
            tick_labels=[cfg.model_label(m) for m in TASK_MODELS],
            p_values=p_fdr.to_numpy(),
            star_color=cfg.plotting["significance_color"],
        )
        if i < len(cfg.metrics) - 1:
            ax.tick_params(labelbottom=False)
    P.condition_legend(
        violin_axes[0],
        ["rest", "calculation"],
        [colors["light"], colors["dark"]],
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=2,
    )
    letter = P.Letters()
    for i, ax in enumerate(violin_axes):  # one letter per metric; stacked axes: letters left of the tick labels
        letter(ax, (-31, 6) if i == 0 else (-31, -2))
    P.panel_label(left, "", "Individual level")

    # (b) group level, PLS task patterns: synthetic versus analytical
    pls_table, pls_maps = pls_task_patterns(voxels, cfg, task_subjects)
    for ax in _draw_pls_panel(sub_b, pls_table, pls_maps, cfg):
        letter(ax)
    P.panel_label(sub_b, "", "Group level: task patterns")

    # (c) group level, minimal-input model during calculation
    gmap, row = _group_agreement(voxels, cfg, task, cfg.minimal_model)
    P.agreement_scatter(
        ax_c,
        gmap["CMRO2"],
        gmap["synthCMRO2"],
        _to_agreement(row),
        cfg.plotting["axis_limits"]["task"],
        title="" if ctx else f"{cfg.model_label(cfg.minimal_model)} model, calculation",
        stats_in_title=ctx is not None,
        **_scatter_kwargs(cfg),
    )
    if ctx:
        norm = _scatter_with_maps(ax_c, ctx, gmap)
        ctx.hcolorbar(cbar_c, norm, x_center=0.5, y=0.75)
    letter(ax_c)
    P.panel_label(sub_c, "", f"Group level: calculation, {cfg.model_label(cfg.minimal_model)} model")

    tables = {
        "fig3a_rest_vs_calc": comparisons,
        "fig3b_pls_overlap": pls_table,
        "fig3c_group_agreement": pd.DataFrame([row]),
    }
    maps = {f"synthetic_{task}_{cfg.minimal_model}": gmap["synthCMRO2"], f"analytical_{task}": gmap["CMRO2"]}
    return FigureOutput(fig, tables, maps)


def pls_task_patterns(
    voxels: pd.DataFrame, cfg: Config, subjects: list[str]
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Overlap of reliable calculation-versus-rest PLS patterns, synthetic versus analytical.

    Returns the statistics table (one row per model) and the oriented bootstrap
    ratio maps (positive = calculation > rest), including the analytical one.
    """
    p = cfg.pls
    datasets = [cfg.datasets["test"], cfg.datasets["task"]]
    first_model = next(iter(p["model_tags"]))

    def oriented(result, column, model):
        obs = match_observations(result, voxels, column, model, datasets, subjects)
        sign = orientation_sign(result, obs["condition"], p["task"], p["baseline"])
        scores = pd.Series(
            sign * result.brain_scores, index=pd.MultiIndex.from_frame(obs[["subj_id", "condition"]])
        )
        bsr = sign * result.bsr
        n_mask = len(voxel_parcel_ids(cfg.surface)) if cfg.surface.glasser_atlas is not None else None
        if cfg.exclude_parcels or (n_mask is not None and bsr.size != n_mask):
            # PLS ran without excluded voxels (sensitivity analysis, or public data that already exclude them):
            # back to all mask voxels (0 = not reliable)
            kept = np.sort(voxels["vox_id"].unique())
            if bsr.size != kept.size:
                raise ValueError(f"{result.name}: {bsr.size} PLS voxels for {kept.size} kept voxels; rerun `pls`")
            bsr = expand(bsr, kept, n_mask if n_mask is not None else int(kept.max()) + 1)
        return bsr, scores

    analytical = load_pls_result(cfg.pls_file(p["analytical_tag"]), p["latent_variable"])
    bsr_a, scores_a = oriented(analytical, "CMRO2", first_model)
    maps = {"analytical": bsr_a}
    rows = []
    for model, tag in p["model_tags"].items():
        result = load_pls_result(cfg.pls_file(tag), p["latent_variable"])
        bsr_m, scores_m = oriented(result, "synthCMRO2", model)
        maps[model] = bsr_m
        rows.append(
            {
                "model": model,
                "file_tag": tag,
                "p_lv_synthetic": result.p_value,
                "p_lv_analytical": analytical.p_value,
                "n_perm": result.n_perm,
                "n_boot": result.n_boot,
                "bsr_threshold": p["bsr_threshold"],
                **overlap(bsr_m, bsr_a, p["bsr_threshold"]),
                "brain_score_r": float(np.corrcoef(scores_m, scores_a.reindex(scores_m.index))[0, 1]),
            }
        )
    return pd.DataFrame(rows), maps


def _draw_pls_panel(sub: plt.Figure, table: pd.DataFrame, maps: dict, cfg: Config) -> list:
    """Axial BSR slices per model (needs the gray matter mask); otherwise a Dice bar chart. Returns the axes."""
    p = cfg.pls
    thr = p["bsr_threshold"]
    models = list(p["model_tags"])
    dice = table.set_index("model")["dice"]
    if not cfg.can_render_slices:
        logger.warning("gm_mask not configured: Figure 3b shows Dice values instead of brain slices")
        ax = sub.subplots()
        ax.bar(range(len(models)), dice.loc[models], color="0.5")
        ax.set_xticks(range(len(models)), [cfg.model_label(m) for m in models], rotation=30, ha="right")
        ax.set(ylim=(0, 1), ylabel=f"Dice, |BSR| > {thr:g}")
        for i, m in enumerate(models):
            ax.text(i, dice[m] + 0.02, f"{dice[m]:.2f}", ha="center", va="bottom")
        return [ax]

    axes = sub.subplots(2, int(np.ceil(len(models) / 2))).ravel()
    reliable_a = np.abs(maps["analytical"]) > thr
    for ax, model in zip(axes, models, strict=True):
        both = (np.abs(maps[model]) > thr) & reliable_a
        overlay = plot_bsr_slice(
            ax,
            maps[model],
            both,
            cfg.gm_mask,
            p["slice_z"],
            thr,
            p["bsr_limit"],
            background=cfg.brain_background,
        )
        ax.set_title(cfg.model_label(model), fontsize=plt.rcParams["font.size"])
        ax.text(0.5, -0.02, f"Dice = {dice[model]:.2f}", transform=ax.transAxes, ha="center", va="top")
    sub.colorbar(overlay, ax=axes, shrink=0.6, label="BSR (calculation > rest)")
    return list(axes)


def _rest_vs_task(data: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Paired Wilcoxon (calculation minus rest) per model; FDR across models within each metric."""
    rows = []
    for metric in cfg.metrics:
        for model in TASK_MODELS:
            wide = paired_by_subject(data, metric, "condition", ["calc", "rest"], model=model)
            rows.append(
                {
                    "metric": metric,
                    "model": model,
                    **paired_difference_summary(wide["calc"], wide["rest"], cfg.analysis["sd_ddof"]),
                }
            )
    out = pd.DataFrame(rows)
    out["p_fdr"] = out.groupby("metric")["p"].transform(lambda p: fdr_bh(p.to_numpy()))
    return out


# --------------------------------------------------------------------------
# Figure 4: generalizability and age
# --------------------------------------------------------------------------


def make_figure4(
    metrics: pd.DataFrame, voxels: pd.DataFrame, cfg: Config, ctx: SurfaceContext | None = None
) -> FigureOutput:
    test = cfg.datasets["test"]
    new_sets = [cfg.datasets["replication"], cfg.datasets["reproducibility"]]
    limit_keys = ["replication", "reproducibility"]
    model = cfg.minimal_model
    colors = cfg.plotting["condition_colors"]

    fig = plt.figure(figsize=(FIG4_WIDTH, 7.6), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.7, 1])
    cols = top.subfigures(1, 2, wspace=0.12)

    group_rows, gh_tables, maps = [], [], {}
    letter = P.Letters()
    for col, dataset, lim_key in zip(cols, new_sets, limit_keys, strict=True):
        sub_scatter, sub_violins = col.subfigures(2, 1, height_ratios=[2, 1])
        ax_scatter = sub_scatter.subplots()
        violin_axes = sub_violins.subplots(1, len(cfg.metrics))

        gmap, row = _group_agreement(voxels, cfg, dataset, model)
        group_rows.append(row)
        maps[f"analytical_{dataset}"] = gmap["CMRO2"]
        maps[f"synthetic_{dataset}_{model}"] = gmap["synthCMRO2"]
        P.agreement_scatter(
            ax_scatter,
            gmap["CMRO2"],
            gmap["synthCMRO2"],
            _to_agreement(row),
            cfg.plotting["axis_limits"][lim_key],
            title="" if ctx else f"{cfg.model_label(model)} model, rest",
            stats_in_title=ctx is not None,
            **_scatter_kwargs(cfg),
        )
        if ctx:
            norm = _scatter_with_maps(ax_scatter, ctx, gmap)
            ctx.vcolorbar(sub_scatter, ax_scatter, norm)
        letter(ax_scatter)
        col.suptitle(
            f"{cfg.dataset_labels[dataset]}\n{cfg.model_label(model)} model, rest",
            fontweight="bold",
            fontsize=plt.rcParams["font.size"] + 1,
        )

        data = select(metrics, dataset=[test, dataset], model=model)
        gh = pd.concat(
            [games_howell(data[m], data["dataset"]).assign(metric=m) for m in cfg.metrics], ignore_index=True
        )
        gh_tables.append(gh)
        for ax, (metric, label) in zip(violin_axes, cfg.metrics.items(), strict=True):
            p = gh.query("metric == @metric")["p"].to_numpy()
            P.split_violins(
                ax,
                data,
                metric,
                "model",
                [model],
                "dataset",
                [test, dataset],
                [colors["light"], colors["dark"]],
                label,
                tick_labels=[""],
                p_values=p,
                star_color=cfg.plotting["significance_color"],
                width=0.55,
            )
        for ax in violin_axes:  # one letter per individual-level metric
            letter(ax)
    P.condition_legend(
        top,
        ["test set", "new dataset"],
        [colors["light"], colors["dark"]],
        loc="outside lower center",
        ncol=2,
    )
    P.panel_label(top, "", "New participants and another vendor, without retraining")

    # (b) age
    ax_ana, ax_syn = bottom.subplots(1, 2)
    aging = select(metrics, dataset=test, model=model).copy()
    aging["MAE_relative"] = aging["MAE"] / aging["CMRO2"]
    aging["median_bias"] = aging["synthCMRO2"] - aging["CMRO2"]
    # Age calibration: the model error (synthetic - analytical median) is fitted against age in out-of-fold
    # predictions of the training participants and subtracted from the test set, so the correction needs
    # only age, never the test participants' analytical maps.
    units = cfg.plotting["units"]
    calibration, correction = None, pd.DataFrame()
    train_key = cfg.datasets.get("calibration")
    if cfg.analysis.get("age_calibration", True) and cfg.age_calibration is not None:
        # public data: the training participants' ages are not shared, so the fitted calibration is read
        stored = pd.read_csv(cfg.age_calibration)
        stored = stored[stored["model"].astype(str) == model]
        if stored.empty:
            raise ValueError(f"{cfg.age_calibration}: no age calibration for model {model!r}")
        calibration = {k: (int(v) if k == "n_train" else float(v)) for k, v in stored.iloc[0].items()
                       if k in ("n_train", "a", "b", "b_se", "b_p", "r", "mean_bias", "age_min", "age_max")}  # fmt: skip
        aging["synthCMRO2_calibrated"] = apply_age_calibration(aging["synthCMRO2"], aging["age"], calibration)
        correction = pd.DataFrame([{"dataset": str(stored.iloc[0]["dataset"]), "model": model, **calibration}])
        logger.info("Figure 4b: age calibration read from %s: error = %.1f %+.2f x age (P = %.3g; %d training "
                    "participants)", cfg.age_calibration.name, calibration["a"], calibration["b"],
                    calibration["b_p"], calibration["n_train"])  # fmt: skip
    elif cfg.analysis.get("age_calibration", True) and train_key:
        train = metrics[(metrics["dataset"].astype(str) == train_key) & (metrics["model"].astype(str) == model)]
        if train.empty:
            logger.warning("Figure 4b: no %s rows for %s in the participant metrics; plotting uncorrected "
                           "synthetic CMRO2 (run `synthcmro2 model predict/export` for that dataset)",
                           train_key, model)  # fmt: skip
        else:
            calibration = age_bias_calibration(train)
            aging["synthCMRO2_calibrated"] = apply_age_calibration(aging["synthCMRO2"], aging["age"], calibration)
            correction = pd.DataFrame([{"dataset": train_key, "model": model, **calibration}])
            logger.info("Figure 4b: age calibration from %d training participants (out of fold): error = %.1f "
                        "%+.2f x age (P = %.3g)", calibration["n_train"], calibration["a"], calibration["b"],
                        calibration["b_p"])  # fmt: skip
    # the figure shows the uncorrected synthetic decline (analysis.age_plot: synthetic) unless the calibrated
    # one is requested (analysis.age_plot: calibrated); both are reported in fig4b_age_slopes
    if calibration is not None and cfg.analysis.get("age_plot", "synthetic") == "calibrated":
        syn_column, syn_ylabel = "synthCMRO2_calibrated", f"Age-calibrated synthetic\nCMRO₂ ({units})"
    else:
        syn_column, syn_ylabel = "synthCMRO2", f"Gray matter synthetic\nCMRO₂ ({units})"
    # decline per decade: analytical, synthetic and calibrated, and the excess of each over analytical
    slope_columns = {"analytical": "CMRO2", "synthetic": "synthCMRO2"}
    if calibration is not None:
        slope_columns["calibrated"] = "synthCMRO2_calibrated"
    decline = age_slopes(aging, slope_columns, "CMRO2", n_bootstrap=int(cfg.analysis["n_bootstrap"]),
                         seed=int(cfg.analysis["random_seed"]), ci_level=float(cfg.analysis["ci_level"]))  # fmt: skip
    # with analytical CMRO2 available (evaluation): the age effects adjusted for the participant-wise agreement
    # slope, which itself changes with age
    adjusted = age_slopes_adjusted(aging, {"analytical": "CMRO2", "synthetic": "synthCMRO2"}, "CMRO2", "slope",
                                   n_bootstrap=int(cfg.analysis["n_bootstrap"]), seed=int(cfg.analysis["random_seed"]),
                                   ci_level=float(cfg.analysis["ci_level"]))  # fmt: skip
    age_rows = []
    for ax, column, title, ylabel in [
        (ax_ana, "CMRO2", "Analytical CMRO₂", f"Gray matter\nCMRO₂ ({cfg.plotting['units']})"),
        (ax_syn, syn_column, f"Synthetic CMRO₂ ({cfg.model_label(model)})", syn_ylabel),
    ]:
        res = correlation(aging["age"], aging[column])
        age_rows.append({"variable": column, **res})
        P.covariate_scatter(
            ax,
            aging,
            "age",
            column,
            res["r"],
            res["p"],
            "Age (years)",
            ylabel,
            title=title,
            seed=cfg.analysis["random_seed"],
        )
    letter(ax_ana)
    letter(ax_syn)
    P.panel_label(bottom, "", f"Age-related CMRO₂ decline (test set, n = {age_rows[0]['n']})")
    # reported, not plotted: the uncorrected synthetic decline (if the plot shows the corrected one) and
    # whether model agreement itself changes with age: slope, r, MAE, MAE relative to the participant's
    # median analytical CMRO2, and the bias of the median (synthetic - analytical)
    extra = [c for c in ("synthCMRO2", "synthCMRO2_calibrated") if c in aging and c != syn_column]
    for column in [*extra, "slope", "r", "MAE", "MAE_relative", "median_bias"]:
        age_rows.append({"variable": column, **correlation(aging["age"], aging[column])})

    # Table S4: the same age associations of participant-wise agreement for every test-set model
    s4_rows = []
    for m in metrics.loc[metrics["dataset"].astype(str) == test, "model"].astype(str).unique():
        mm = select(metrics, dataset=test, model=m).copy()
        mm["MAE_relative"] = mm["MAE"] / mm["CMRO2"]
        mm["median_bias"] = mm["synthCMRO2"] - mm["CMRO2"]
        for column in ["r", "slope", "MAE", "MAE_relative", "median_bias", "synthCMRO2", "CMRO2"]:
            s4_rows.append({"model": m, "variable": column, **correlation(mm["age"], mm[column])})

    tables = {
        "tableS4_age_associations": pd.DataFrame(s4_rows),
        "fig4a_group_agreement": pd.DataFrame(group_rows),
        "fig4a_games_howell": pd.concat(gh_tables, ignore_index=True),
        "fig4b_age_correlations": pd.DataFrame(age_rows),
        **({"surface_parcel_coverage": ctx.coverage} if ctx else {}),
        "fig4b_age_calibration": correction.assign(test_dataset=test, n_test=len(aging)),
        "fig4b_age_slopes": decline,
        "fig4b_age_slopes_slope_adjusted": adjusted,
    }
    return FigureOutput(fig, tables, maps)


# --------------------------------------------------------------------------
# Figure S2: test-retest reliability across Glasser parcels
# --------------------------------------------------------------------------


def make_figure_s2(
    metrics: pd.DataFrame,
    voxels: pd.DataFrame,
    cfg: Config,
    ctx: SurfaceContext | None = None,
    retest_voxels: pd.DataFrame | None = None,
    parcel_ids: np.ndarray | None = None,
) -> FigureOutput:
    """Parcel-wise ICC(2,1) of analytical and synthetic CMRO2 between the two within-session scans.

    Scan 1 is the participants' rest maps in the test set (``voxels``), scan 2 the ``test_retest`` dataset
    (``paths.retest_voxel_values``). Parcel labels come from the surface context, or from the gray matter mask
    and Glasser atlas in the config. Without the individual voxel tables (public data), the stored parcel ICCs
    (``paths.parcel_icc``) are summarized, tested and plotted.
    """
    if retest_voxels is None and (is_group_table(voxels) or not cfg.has_retest) and cfg.has_parcel_icc:
        icc = pd.read_csv(cfg.parcel_icc)
        order = [R.ANALYTICAL, *[m for m in cfg.model_order if m in set(icc["model"].astype(str))]]
        summary = R.summarize_icc(icc)
        tests = R.compare_with_analytical(icc)
        n_subjects = int(icc["n_targets"].max())
        logger.info("Figure S2: parcel ICCs read from %s (%d parcels, maps %s)", cfg.parcel_icc.name,
                    int(summary["n_parcels"].max()), order)  # fmt: skip
        fig = plot_parcel_icc(icc, tests, order, cfg, n_subjects)
        return FigureOutput(fig, {"figS2_parcel_icc": icc, "figS2_icc_summary": summary, "figS2_vs_analytical": tests})
    if retest_voxels is None:
        if not cfg.has_retest:
            raise FileNotFoundError(f"test-retest voxel table not found: {cfg.retest_voxel_values}")
        retest_voxels = load_voxel_values(cfg.retest_voxel_values)
    if parcel_ids is None:
        parcel_ids = ctx.parcel_ids if ctx is not None else voxel_parcel_ids(cfg.surface)
    if cfg.exclude_parcels:  # excluded parcels get no ICC (label 0 = outside any parcel)
        parcel_ids = np.where(np.isin(parcel_ids, labels_for(cfg)), 0, parcel_ids)
    scan2 = select(retest_voxels, dataset=cfg.datasets["retest"])
    scan2 = scan2[scan2["condition"].astype(str) == "rest"]
    scan1 = select(voxels, dataset=cfg.datasets["test"], subj_id=sorted(scan2["subj_id"].astype(str).unique()))
    scan1 = scan1[scan1["condition"].astype(str) == "rest"]
    maps, subjects, models = R.session_maps(scan1, scan2, cfg.analysis.get("positive_values_only", True))
    order = [R.ANALYTICAL, *[m for m in cfg.model_order if m in models]]
    maps = {k: maps[k] for k in order}
    icc = R.parcel_icc(maps, parcel_ids, int(cfg.reliability.get("min_parcel_voxels", 5)))
    summary = R.summarize_icc(icc)
    tests = R.compare_with_analytical(icc)
    n_parcels = int(summary["n_parcels"].max())
    logger.info("Figure S2: %d participants, %d parcels, maps %s", len(subjects), n_parcels, order)

    fig = plot_parcel_icc(icc, tests, order, cfg, len(subjects))
    tables = {
        "figS2_parcel_icc": icc,
        "figS2_icc_summary": summary,
        "figS2_vs_analytical": tests,
    }
    return FigureOutput(fig, tables)


def plot_parcel_icc(icc: pd.DataFrame, tests: pd.DataFrame, order: list[str], cfg: Config, n_subjects: int) -> Figure:
    """Figure S2: violins of the parcel ICCs per map, reliability bands, and stars versus analytical CMRO2."""
    rel = cfg.reliability
    n_parcels = int(icc.dropna(subset=["ICC"]).groupby("model")["ICC"].size().max())
    labels = ["analytical", *[cfg.model_label(m) for m in order[1:]]]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    P.metric_violins(ax, icc.dropna(subset=["ICC"]), "ICC", order, labels, "ICC(2,1)", cut=0, show_points=False)
    ax.set_xticks(range(len(order)), [lab[0].upper() + lab[1:] for lab in labels], rotation=60, ha="right")
    lo, hi = rel.get("ylim", (0.1, 1.1))
    ax.set_ylim(lo, hi)
    for y, name, ls in zip(rel.get("bands", (0.5, 0.75, 0.9)), ("moderate", "good", "excellent"), (":", "--", "-.")):
        ax.axhline(y, color="#9e9e9e", lw=0.6, ls=ls, zorder=1)
        ax.text(1.01, y, name, fontsize=6, color="#757575", va="center", ha="left",
                transform=ax.get_yaxis_transform())  # fmt: skip
    p_fdr = tests.set_index("model")["p_fdr"]
    for i, m in enumerate(order):
        stars = significance_stars(p_fdr[m]) if m in p_fdr.index else ""
        if stars:
            ax.text(i, 1.01, stars, ha="center", va="bottom", color=cfg.plotting.get("significance_color", "#d9534f"),
                    fontsize=9, fontweight="bold")  # fmt: skip
    ax.set_title(f"Test-retest reliability ({n_parcels} Glasser parcels, n = {n_subjects})", loc="left")
    return fig
