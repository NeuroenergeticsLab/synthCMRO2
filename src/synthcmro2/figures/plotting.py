"""Reusable plotting primitives.

Each function draws onto an existing Axes and returns it. Statistics are passed
in, never computed here, so the numbers on a figure always come from
``stats.py`` and match the tables written next to the figure.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patheffects
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.ticker import MaxNLocator

from .stats import Agreement, format_num, format_p, significance_stars

# thin white outline keeps text readable over data without hiding the points behind it
_HALO = [patheffects.withStroke(linewidth=2.5, foreground="white")]


def apply_style(plotting: dict[str, Any]) -> None:
    """Set a uniform style for every figure (call once, before plotting)."""
    sns.set_style("whitegrid")
    mpl.rcParams.update(
        {
            "font.family": plotting.get("font_family", "DejaVu Sans"),
            "font.size": plotting.get("font_size", 8),
            "axes.titlesize": plotting.get("font_size", 8),
            "axes.titleweight": "bold",
            "axes.labelsize": plotting.get("font_size", 8),
            "xtick.labelsize": plotting.get("font_size", 8) - 1,
            "ytick.labelsize": plotting.get("font_size", 8) - 1,
            "legend.fontsize": plotting.get("font_size", 8) - 1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "savefig.transparent": False,
            "svg.fonttype": "none",  # keep text editable in vector editors
            "pdf.fonttype": 42,
        }
    )


def panel_label(target: Axes | SubFigure, letter: str, title: str = "") -> None:
    """Bold lowercase panel letter plus heading.

    For a subfigure, the heading is placed as its left-aligned suptitle, so
    constrained layout reserves space for it. For an Axes, it is placed just
    above the top-left corner.
    """
    size = mpl.rcParams["font.size"] + 2
    text = rf"$\bf{{{letter}}}$  {title}" if letter else title
    if isinstance(target, SubFigure):
        target.suptitle(text, x=0.0, ha="left", fontsize=size)
    else:
        target.text(0.0, 1.02, text, transform=target.transAxes, ha="left", va="bottom", fontsize=size)


def subplot_letter(ax: Axes, letter: str, offset: tuple[float, float] = (-22, 6)) -> None:
    """Bold lowercase letter at the top-left corner of one subplot, outside the axes (offset in points)."""
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=offset, textcoords="offset points",
                ha="right", va="bottom", fontweight="bold", fontsize=mpl.rcParams["font.size"] + 2,
                annotation_clip=False)  # fmt: skip


class Letters:
    """Sequential subplot letters within one figure (a, b, c, ...)."""

    def __init__(self) -> None:
        self.i = 0

    def __call__(self, ax: Axes, offset: tuple[float, float] = (-22, 6)) -> None:
        subplot_letter(ax, "abcdefghijklmnopqrstuvwxyz"[self.i], offset)
        self.i += 1


def agreement_scatter(
    ax: Axes,
    analytical: np.ndarray,
    synthetic: np.ndarray,
    result: Agreement,
    limits: Sequence[float],
    title: str = "",
    units: str = "µmol/100 g/min",
    point_size: float = 1.0,
    point_color: str = "#666666",
    line_color: str = "black",
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    stats_in_title: bool = False,
) -> Axes:
    """Voxel-wise analytical versus synthetic CMRO2 with OLS fit and statistics.

    Statistics are printed in the upper left, or on one line as the axes title
    (``stats_in_title``) when the plot area is needed for map insets.
    """
    ax.scatter(analytical, synthetic, s=point_size, color=point_color, linewidths=0)
    x_fit = np.linspace(np.min(analytical), np.max(analytical), 100)
    ax.plot(x_fit, result.intercept + result.slope * x_fit, color=line_color, lw=1.2)
    ax.set(xlim=limits, ylim=limits, title=title)
    ax.set_xlabel(f"Analytical CMRO₂\n({units})" if show_xlabel else "")
    ax.set_ylabel(f"Synthetic CMRO₂\n({units})" if show_ylabel else "")
    ax.set_aspect("equal", adjustable="box")
    ticks = np.linspace(limits[0], limits[1], 3)  # three ticks keep small panels legible
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    text = (f"r = {format_num(result.r)}, {format_p(result.p)}\nMAE = {format_num(result.mae, 1)}"
            f"\nslope = {format_num(result.slope)}")
    if stats_in_title:
        ax.set_title(
            text.replace("\nMAE", "\nMAE").replace("\nslope", "; slope"),
            fontsize=mpl.rcParams["font.size"] - 1,
            fontweight="normal",
        )
        return ax
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=mpl.rcParams["font.size"] - 1,
        linespacing=1.3,
        zorder=5,
        path_effects=_HALO,
    )
    return ax


def metric_violins(
    ax: Axes,
    data: pd.DataFrame,
    metric: str,
    order: Sequence[str],
    tick_labels: Sequence[str],
    ylabel: str,
    show_xticklabels: bool = True,
    cut: float = 1,
    show_points: bool = True,
) -> Axes:
    """Metric per model (one point per participant, or per parcel): violin, box, and (optionally) the points.

    cut=1 lets the density taper one bandwidth beyond the data instead of ending in a flat edge; use cut=0 for
    bounded metrics such as the ICC (<= 1).
    """
    common = {"data": data, "x": "model", "y": metric, "order": list(order), "ax": ax}
    sns.violinplot(**common, inner=None, color="white", linewidth=0.8, cut=cut)
    sns.boxplot(
        **common,
        width=0.15,
        showfliers=False,
        boxprops={"facecolor": "black", "zorder": 5},
        whiskerprops={"zorder": 5},
        medianprops={"color": "white", "zorder": 6},
    )
    if show_points:
        sns.stripplot(
            **common, color="gray", alpha=0.8, size=2, jitter=True, zorder=4, edgecolor="white", linewidth=0.2
        )
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set(xlabel="", ylabel=ylabel.replace(" (", "\n("))  # units on a second line
    if show_xticklabels:
        ax.set_xticks(range(len(order)), tick_labels, rotation=60, ha="right")
    else:
        ax.set_xticks(range(len(order)), [])
    return ax


def split_violins(
    ax: Axes,
    data: pd.DataFrame,
    metric: str,
    x: str,
    order: Sequence[str],
    hue: str,
    hue_order: Sequence[str],
    colors: Sequence[str],
    ylabel: str,
    tick_labels: Sequence[str] | None = None,
    p_values: Sequence[float] | None = None,
    star_color: str = "#d9534f",
    width: float = 0.8,
) -> Axes:
    """Two-condition split violins with optional significance stars per x level."""
    sns.violinplot(
        data=data,
        x=x,
        y=metric,
        order=list(order),
        hue=hue,
        hue_order=list(hue_order),
        split=True,
        fill=False,
        inner="box",
        palette=list(colors),
        linewidth=0.9,
        cut=1,  # taper beyond the data instead of a flat edge at the minimum and maximum
        width=width,
        legend=False,
        ax=ax,
    )
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set(xlabel="", ylabel=ylabel.replace(" (", "\n("))  # units on a second line
    labels = list(tick_labels) if tick_labels is not None else list(order)
    ax.set_xticks(
        range(len(order)),
        labels,
        rotation=60 if len(order) > 1 else 0,
        ha="right" if len(order) > 1 else "center",
    )
    if p_values is not None:
        add_stars(
            ax,
            range(len(order)),
            p_values,
            data.groupby(x, observed=True)[metric].max().reindex(order).to_numpy(),
            color=star_color,
        )
    return ax


def add_stars(
    ax: Axes,
    positions: Sequence[float],
    p_values: Sequence[float],
    tops: Sequence[float],
    color: str = "#d9534f",
) -> None:
    """Significance stars just above the highest value of each group."""
    lo, hi = ax.get_ylim()
    pad = 0.03 * (hi - lo)
    for xpos, p, top in zip(positions, p_values, tops, strict=True):
        stars = significance_stars(p)
        if stars:
            ax.text(
                xpos,
                top + pad,
                stars,
                ha="center",
                va="bottom",
                color=color,
                fontsize=mpl.rcParams["font.size"] + 3,
                fontweight="bold",
            )
    ax.set_ylim(lo, max(hi, max(tops) + 6 * pad))


def condition_legend(
    target: Axes | SubFigure, labels: Sequence[str], colors: Sequence[str], **kwargs: Any
) -> None:
    """Legend with one patch per condition (used for split violins), on an Axes or subfigure."""
    handles = [
        mpl.patches.Patch(facecolor="white", edgecolor=c, label=lab)
        for lab, c in zip(labels, colors, strict=True)
    ]
    target.legend(handles=handles, frameon=False, **kwargs)


def covariate_scatter(
    ax: Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    r: float,
    p: float,
    xlabel: str,
    ylabel: str,
    title: str = "",
    color: str = "#444444",
    seed: int | None = None,
) -> Axes:
    """Scatter with linear fit and 95% CI band (for example, CMRO2 versus age)."""
    sns.regplot(
        data=data,
        x=x,
        y=y,
        ax=ax,
        color=color,
        ci=95,
        seed=seed,
        scatter_kws={"s": 10},
        line_kws={"color": "black", "lw": 1.2},
    )
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    # Headroom above the highest point so the statistics label never hides data.
    lo, hi = ax.get_ylim()
    y_max = float(np.nanmax(data[y]))
    ax.set_ylim(lo, max(hi, y_max + 0.15 * (y_max - lo)))
    ax.text(
        0.03,
        0.97,
        f"r = {format_num(r)}, {format_p(p)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        zorder=5,
        path_effects=_HALO,
    )
    return ax


def save_figure(
    fig: Figure, out_dir: Path, name: str, formats: Sequence[str], dpi: float | None = None
) -> list[Path]:
    """Save a figure in every requested format and close it.

    ``dpi`` sets the resolution of raster formats (PNG); None keeps Matplotlib's default (rcParams
    ``savefig.dpi``, which is ``"figure"``: the figure's own dpi, 100 unless changed).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    kwargs = {} if dpi is None else {"dpi": dpi}
    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        fig.savefig(path, **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths
