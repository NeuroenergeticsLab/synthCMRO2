"""Statistics reported with the figures.

Every function is pure: it takes arrays or data frames and returns numbers or
data frames, without plotting, printing, or touching files. This keeps the
numbers testable independently of the figures.

The tests follow those named in the manuscript:
- agreement between analytical and synthetic CMRO2: Pearson r with a BCa
  bootstrap CI, ordinary least squares slope, and mean absolute error;
- paired comparisons (models, rest versus task): Wilcoxon signed-rank tests with
  Benjamini-Hochberg false discovery rate correction;
- independent datasets versus test set: Games-Howell test;
- age effects: Pearson correlation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------
# Agreement between analytical and synthetic maps
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Agreement:
    """Agreement between analytical (x) and synthetic (y) CMRO2 values."""

    n: int
    r: float
    p: float
    r_ci_low: float
    r_ci_high: float
    slope: float
    intercept: float
    mae: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _pearson_r_vectorized(x: np.ndarray, y: np.ndarray, axis: int = -1) -> np.ndarray:
    """Pearson r along ``axis``; works on bootstrap resample batches."""
    xm = x - x.mean(axis=axis, keepdims=True)
    ym = y - y.mean(axis=axis, keepdims=True)
    return (xm * ym).sum(axis=axis) / np.sqrt((xm**2).sum(axis=axis) * (ym**2).sum(axis=axis))


def _bootstrap_rng_kwarg():
    """SciPy >= 1.15 names the random generator ``rng``; older versions ``random_state`` (same behavior)."""
    import inspect

    name = "rng" if "rng" in inspect.signature(stats.bootstrap).parameters else "random_state"
    return lambda rng: {name: rng}


_BOOTSTRAP_RNG = _bootstrap_rng_kwarg()


def agreement(
    analytical: np.ndarray,
    synthetic: np.ndarray,
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
    rng: np.random.Generator | int | None = None,
) -> Agreement:
    """Pearson r (BCa bootstrap CI), OLS slope of synthetic on analytical, and MAE."""
    x = np.asarray(analytical, dtype=float)
    y = np.asarray(synthetic, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("analytical and synthetic must be 1-D arrays of equal length")
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]

    r, p = stats.pearsonr(x, y)
    fit = stats.linregress(x, y)
    boot = stats.bootstrap(
        (x, y),
        _pearson_r_vectorized,
        paired=True,
        vectorized=True,
        n_resamples=n_bootstrap,
        confidence_level=ci_level,
        method="BCa",
        batch=100,  # bounds memory for maps with thousands of voxels
        **_BOOTSTRAP_RNG(rng),
    )
    return Agreement(
        n=int(x.size),
        r=float(r),
        p=float(p),
        r_ci_low=float(boot.confidence_interval.low),
        r_ci_high=float(boot.confidence_interval.high),
        slope=float(fit.slope),
        intercept=float(fit.intercept),
        mae=float(np.mean(np.abs(x - y))),
    )


# --------------------------------------------------------------------------
# Paired comparisons
# --------------------------------------------------------------------------


def fdr_bh(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaN-aware)."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        adjusted[ok] = stats.false_discovery_control(p[ok], method="bh")
    return adjusted


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Two-sided Wilcoxon signed-rank test on paired samples (a - b).

    W is the smaller of the positive and negative rank sums (as in SciPy and
    Pingouin). The exact null distribution is used for small samples without
    ties, which reproduces Pingouin's ``wilcoxon``/``pairwise_tests`` output.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    res = stats.wilcoxon(a, b, alternative="two-sided")
    return {"n": int(a.size), "W": float(res.statistic), "p": float(res.pvalue)}


def paired_difference_summary(a: np.ndarray, b: np.ndarray, ddof: int = 1) -> dict[str, float]:
    """Mean and SD of the paired difference a - b, plus the Wilcoxon test."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return {
        "mean_diff": float(diff.mean()),
        "sd_diff": float(diff.std(ddof=ddof)),
        **wilcoxon_paired(a, b),
    }


def pairwise_wilcoxon(wide: pd.DataFrame) -> pd.DataFrame:
    """All pairwise Wilcoxon tests between the columns of a subject x level table.

    Differences are column A minus column B. P-values are FDR-corrected across
    all pairs.
    """
    rows = []
    for a, b in combinations(wide.columns, 2):
        rows.append({"A": a, "B": b, **paired_difference_summary(wide[a], wide[b])})
    out = pd.DataFrame(rows)
    out["p_fdr"] = fdr_bh(out["p"].to_numpy())
    return out


# --------------------------------------------------------------------------
# Independent groups
# --------------------------------------------------------------------------


def games_howell(values: pd.Series, groups: pd.Series) -> pd.DataFrame:
    """Games-Howell post hoc test for all pairs of groups (unequal variances).

    Implements the same formulas as ``pingouin.pairwise_gameshowell``:
    Welch standard error and degrees of freedom, and p-values from the
    studentized range distribution. Differences are group A minus group B,
    with groups in sorted order.
    """
    df = pd.DataFrame({"value": np.asarray(values, dtype=float), "group": np.asarray(groups)})
    summary = df.groupby("group")["value"].agg(["mean", "var", "count"]).sort_index()
    k = len(summary)
    if k < 2:
        raise ValueError("Games-Howell needs at least two groups")

    rows = []
    for a, b in combinations(summary.index, 2):
        ma, va, na = summary.loc[a]
        mb, vb, nb = summary.loc[b]
        se = np.sqrt(va / na + vb / nb)
        t = (ma - mb) / se
        dof = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
        p = stats.studentized_range.sf(np.sqrt(2) * abs(t), k, dof)
        rows.append(
            {
                "A": a,
                "B": b,
                "mean_A": ma,
                "mean_B": mb,
                "diff": ma - mb,
                "se": se,
                "T": t,
                "df": dof,
                "p": min(float(p), 1.0),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Correlation with age and the age calibration of synthetic CMRO2 (Fig. 4b)
# --------------------------------------------------------------------------


def correlation(x: pd.Series, y: pd.Series) -> dict[str, float]:
    """Pearson correlation after listwise removal of missing values."""
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    r, p = stats.pearsonr(pair["x"], pair["y"])
    return {"n": len(pair), "r": float(r), "p": float(p)}


def age_bias_calibration(train: pd.DataFrame) -> dict[str, float]:
    """Fit the model error (synthetic - analytical median CMRO2) as a linear function of age, ``a + b * age``,
    in out-of-fold predictions of the training participants (columns age, synthCMRO2, CMRO2)."""
    t = train[["age", "synthCMRO2", "CMRO2"]].dropna()
    if len(t) < 3:
        raise ValueError(f"age calibration needs >= 3 training participants with age (got {len(t)})")
    fit = stats.linregress(t["age"], t["synthCMRO2"] - t["CMRO2"])
    return {"n_train": len(t), "a": float(fit.intercept), "b": float(fit.slope), "b_se": float(fit.stderr),
            "b_p": float(fit.pvalue), "r": float(fit.rvalue), "mean_bias": float((t["synthCMRO2"] - t["CMRO2"]).mean()),
            "age_min": float(t["age"].min()), "age_max": float(t["age"].max())}  # fmt: skip


def apply_age_calibration(synthetic: pd.Series, age: pd.Series, fit: dict[str, float]) -> pd.Series:
    """Synthetic CMRO2 minus the error expected at the participant's age (needs no analytical CMRO2)."""
    return synthetic - (fit["a"] + fit["b"] * age)


def age_slopes(
    data: pd.DataFrame, columns: dict[str, str], reference: str, per: float = 10.0, n_bootstrap: int = 2000,
    seed: int = 1234, ci_level: float = 0.95,
) -> pd.DataFrame:  # fmt: skip
    """Linear change with age (per ``per`` years) of each column, and its difference from ``reference``.

    One row per column (label -> column): slope with a participant bootstrap percentile CI, the same in % of the
    mean of ``reference`` (common denominator, so the percentages are comparable), and Pearson r/P with age. One
    row per non-reference column for the excess slope (column - reference), with the bootstrap CI and the P of
    the regression of the paired difference on age (identical to the difference of the two slopes).
    """
    cols = list(dict.fromkeys(columns.values()))
    d = data[["age", *cols]].dropna()
    x = d["age"].to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_bootstrap, len(d)))

    def slope(y: np.ndarray) -> np.ndarray:  # OLS slope for every bootstrap sample (rows of idx) at once
        xs, ys = x[idx], y[idx]
        xc = xs - xs.mean(axis=1, keepdims=True)
        return per * (xc * (ys - ys.mean(axis=1, keepdims=True))).sum(axis=1) / (xc**2).sum(axis=1)

    q = [(1 - ci_level) / 2 * 100, (1 + ci_level) / 2 * 100]
    ref = d[reference].to_numpy(float)
    denom = ref.mean() / 100
    rows = []
    for label, c in columns.items():
        y = d[c].to_numpy(float)
        fit = stats.linregress(x, y)
        lo, hi = np.percentile(slope(y), q)
        rows.append({"variable": label, "column": c, "kind": "slope", "n": len(d), "per_years": per,
                     "slope": per * fit.slope, "ci_low": lo, "ci_high": hi, "pct_of_reference_mean": per * fit.slope / denom,
                     "r": fit.rvalue, "p": fit.pvalue})  # fmt: skip
    for label, c in columns.items():
        if c == reference:
            continue
        diff = d[c].to_numpy(float) - ref
        fit = stats.linregress(x, diff)
        lo, hi = np.percentile(slope(diff), q)
        rows.append({"variable": f"{label} - reference", "column": f"{c} - {reference}", "kind": "excess",
                     "n": len(d), "per_years": per, "slope": per * fit.slope, "ci_low": lo, "ci_high": hi,
                     "pct_of_reference_mean": per * fit.slope / denom, "r": fit.rvalue, "p": fit.pvalue})  # fmt: skip
    return pd.DataFrame(rows)


def age_slopes_adjusted(
    data: pd.DataFrame, columns: dict[str, str], reference: str, covariate: str, per: float = 10.0,
    n_bootstrap: int = 2000, seed: int = 1234, ci_level: float = 0.95,
) -> pd.DataFrame:  # fmt: skip
    """Like :func:`age_slopes`, but each decline is the age coefficient of ``y ~ age + covariate``.

    Used with the participant-wise agreement slope as covariate: the slope needs the analytical map, so this is
    the correction available when analytical CMRO2 exists (evaluation), in contrast to the age calibration. P of
    each coefficient from the OLS t-test; CIs from a participant bootstrap (percentile). The excess rows regress
    the paired difference (column - reference) on age and the covariate.
    """
    cols = list(dict.fromkeys([*columns.values(), covariate]))
    d = data[["age", *cols]].dropna()
    x = d["age"].to_numpy(float)
    z = d[covariate].to_numpy(float)
    n = len(d)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    q = [(1 - ci_level) / 2 * 100, (1 + ci_level) / 2 * 100]

    def fit(y: np.ndarray, rows: np.ndarray | None = None) -> tuple[float, float]:
        xs, zs, ys = (x, z, y) if rows is None else (x[rows], z[rows], y[rows])
        design = np.column_stack([np.ones(len(xs)), xs, zs])
        beta, *_ = np.linalg.lstsq(design, ys, rcond=None)
        resid = ys - design @ beta
        dof = len(ys) - 3
        se = np.sqrt(np.linalg.inv(design.T @ design)[1, 1] * (resid @ resid) / dof)
        return float(beta[1]), float(2 * stats.t.sf(abs(beta[1] / se), dof))

    ref = d[reference].to_numpy(float)
    denom = ref.mean() / 100
    rows = []

    def row(label: str, column: str, kind: str, y: np.ndarray) -> dict:
        b, p = fit(y)
        boot = np.array([fit(y, i)[0] for i in idx]) * per
        lo, hi = np.percentile(boot, q)
        return {"variable": label, "column": column, "kind": kind, "covariate": covariate, "n": n,
                "per_years": per, "slope": per * b, "ci_low": lo, "ci_high": hi,
                "pct_of_reference_mean": per * b / denom, "p": p}  # fmt: skip

    for label, c in columns.items():
        rows.append(row(label, c, "slope", d[c].to_numpy(float)))
    for label, c in columns.items():
        if c != reference:
            rows.append(row(f"{label} - reference", f"{c} - {reference}", "excess", d[c].to_numpy(float) - ref))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Formatting helpers (kept here so text and figures report identical rounding)
# --------------------------------------------------------------------------


def format_p(p: float, italic: bool = True) -> str:
    """P value in Radiology style: 'P < .001'; three decimals below .01 ('P = .004'); otherwise two ('P = .02').

    No leading zero; values that round to 1 are reported as 'P > .99'. With ``italic`` the P is set in
    matplotlib mathtext italics, as the journal requires an italic P.
    """
    sym = r"$\it{P}$" if italic else "P"
    if p < 0.001:
        return f"{sym} < .001"
    if p < 0.01:
        return f"{sym} = {p:.3f}".replace("0.", ".", 1)
    if round(p, 2) >= 1:
        return f"{sym} > .99"
    return f"{sym} = {p:.2f}".replace("0.", ".", 1)


def format_num(x: float, decimals: int = 2) -> str:
    """Number with a typographic minus sign (U+2212) instead of a hyphen."""
    return f"{x:.{decimals}f}".replace("-", "\u2212")


def significance_stars(p: float) -> str:
    """Stars following the journal convention used in the legends."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""
