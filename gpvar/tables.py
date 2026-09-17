"""
Table writers (Markdown and LaTeX/booktabs) plus a few ready-made tables.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


def to_markdown(df: pd.DataFrame, floatfmt: str = "{:.3f}", index: bool = False) -> str:
    d = df.copy()
    if index:
        d = d.reset_index()
    cols = list(d.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in d.iterrows():
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def to_latex(df: pd.DataFrame, caption: str = "", label: str = "", floatfmt: str = "{:.3f}",
             index: bool = False, notes: Optional[str] = None) -> str:
    d = df.copy()
    if index:
        d = d.reset_index()
    cols = list(d.columns)
    align = "l" + "r" * (len(cols) - 1)
    out = ["\\begin{table}[htbp]", "\\centering"]
    if caption:
        out.append(f"\\caption{{{caption}}}")
    if label:
        out.append(f"\\label{{{label}}}")
    out += ["\\begin{tabular}{" + align + "}", "\\toprule",
            " & ".join(str(c).replace("_", "\\_").replace("%", "\\%") for c in cols) + " \\\\", "\\midrule"]
    for _, r in d.iterrows():
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&"))
        out.append(" & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    if notes:
        out.append("\\begin{minipage}{0.95\\linewidth}\\footnotesize\\emph{Notes:} " + notes + "\\end{minipage}")
    out.append("\\end{table}")
    return "\n".join(out)


def save_table(df: pd.DataFrame, path_no_ext: str, caption: str = "", label: str = "", floatfmt: str = "{:.3f}",
               index: bool = False, notes: Optional[str] = None) -> None:
    """Write ``<path>.md``, ``<path>.tex`` and ``<path>.csv``."""
    os.makedirs(os.path.dirname(path_no_ext) or ".", exist_ok=True)
    with open(path_no_ext + ".md", "w", encoding="utf-8") as fh:
        fh.write(to_markdown(df, floatfmt, index))
        if notes:
            fh.write(f"\n\n*Notes:* {notes}\n")
    with open(path_no_ext + ".tex", "w", encoding="utf-8") as fh:
        fh.write(to_latex(df, caption, label, floatfmt, index, notes))
    (df.reset_index() if index else df).to_csv(path_no_ext + ".csv", index=False)


# --------------------------------------------------------------------------- #
# Ready-made tables
# --------------------------------------------------------------------------- #
def hyper_table(results) -> pd.DataFrame:
    """Wide table: per variable the posterior median [IQR] of kappa and xi for own/other lags."""
    h = results.hyper
    rows = []
    for j, nm in enumerate(results.var_names):
        r = [nm]
        for k in (1, 0, 3, 2):
            q25, q50, q75 = np.quantile(h[:, j, k], [0.25, 0.5, 0.75])
            r.append(f"{q50:.3f} [{q25:.3f}, {q75:.3f}]")
        rows.append(r)
    return pd.DataFrame(rows, columns=["Variable", "kappa own", "xi own", "kappa other", "xi other"])


def sv_table(results) -> pd.DataFrame:
    s = results.sv_params
    rows = []
    for j, nm in enumerate(results.var_names):
        r = [nm]
        for k in range(4):
            q05, q50, q95 = np.quantile(s[:, j, k], [0.05, 0.5, 0.95])
            r.append(f"{q50:.3f} [{q05:.3f}, {q95:.3f}]")
        rows.append(r)
    return pd.DataFrame(rows, columns=["Variable", "mu_h", "rho_h", "sigma^2_h", "h_0"])


def girf_peak_table(girfs: Dict[str, "GIRFResult"], mask=None, original_units: bool = False) -> pd.DataFrame:
    """Peak responses (posterior median with 68% band) for several GIRF sets."""
    parts = []
    for lab, g in girfs.items():
        t = g.peak_table(mask, original_units)
        t.insert(0, "shock", lab)
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def girf_horizon_table(g: "GIRFResult", horizons: Sequence[int] = (0, 1, 4, 8, 12, 16), mask=None,
                       original_units: bool = False) -> pd.DataFrame:
    """Median [16%, 84%] responses at selected horizons (wide, variables in rows)."""
    avg = g.average(mask, original_units)
    q16, q50, q84 = np.quantile(avg, [0.16, 0.5, 0.84], axis=0)
    rows = []
    for j, nm in enumerate(g.var_names):
        r = [nm]
        for k in horizons:
            r.append(f"{q50[k, j]:.3f} [{q16[k, j]:.3f}, {q84[k, j]:.3f}]")
        rows.append(r)
    return pd.DataFrame(rows, columns=["Variable"] + [f"h={k}" for k in horizons])


def asymmetry_table(pos: "GIRFResult", neg: "GIRFResult", horizons: Sequence[int] = (1, 4, 8), mask=None) -> pd.DataFrame:
    """
    Sign-asymmetry statistics: posterior median of delta_pos(h) + delta_neg(h)
    (zero under symmetry) with the posterior probability that it is positive.
    """
    ap, an = pos.average(mask), neg.average(mask)
    s = ap + an
    rows = []
    for j, nm in enumerate(pos.var_names):
        r = [nm]
        for k in horizons:
            r.append(f"{np.median(s[:, k, j]):.3f} ({np.mean(s[:, k, j] > 0):.2f})")
        rows.append(r)
    return pd.DataFrame(rows, columns=["Variable"] + [f"h={k}" for k in horizons])


def size_asymmetry_table(g1: "GIRFResult", g2: "GIRFResult", horizons: Sequence[int] = (1, 4, 8), mask=None) -> pd.DataFrame:
    """Ratio of the 2-sd to the 1-sd response (2 under proportionality), posterior median."""
    a1, a2 = g1.average(mask), g2.average(mask)
    rows = []
    for j, nm in enumerate(g1.var_names):
        r = [nm]
        for k in horizons:
            num, den = np.median(a2[:, k, j]), np.median(a1[:, k, j])
            r.append(f"{num / den:.2f}" if abs(den) > 1e-8 else "n/a")
        rows.append(r)
    return pd.DataFrame(rows, columns=["Variable"] + [f"h={k}" for k in horizons])
