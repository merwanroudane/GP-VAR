"""
Data utilities for the US macroeconomic application (Section 5.1 / Table B.1).

Two data sets ship with the package (``gpvar/datasets``):

* ``fredqd_subset.csv`` - 26 series of the quarterly FRED-QD database of
  McCracken and Ng (2020), vintage 2026-08 (1959Q1-2026Q2), including the row
  of FRED-QD transformation codes.  Source: Federal Reserve Bank of St. Louis,
  https://www.stlouisfed.org/research/economists/mccracken/fred-databases
* ``jln_macro_uncertainty.csv`` - the monthly macroeconomic uncertainty index
  of Jurado, Ludvigson and Ng (2015) for h = 1, 3, 12 (1960:07-2026:06), from
  Sydney C. Ludvigson's web page (August 2026 update).

The paper's transformations (Table B.1) are: 1 = none, 2 = year-on-year growth
rate, 3 = quarter-on-quarter growth rate, 4 = quarter-on-quarter percentage
change.  Growth rates are computed as 100 x log differences.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "datasets")

FRED_QD_URL = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/quarterly/current.csv"
JLN_URL = "https://www.sydneyludvigson.com/macro-and-financial-uncertainty-indexes"

# --------------------------------------------------------------------------- #
# Variable definitions (Table B.1 of the Online Appendix)
# --------------------------------------------------------------------------- #
# name -> (FRED-QD mnemonic, paper transformation code, description)
GPVAR8: Dict[str, Tuple[str, int, str]] = {
    "UNC":   ("JLN",             1, "Macroeconomic uncertainty index (Jurado, Ludvigson and Ng, 2015), h = 1"),
    "RGDP":  ("GDPC1",           2, "Real gross domestic product (yoy growth, %)"),
    "EMP":   ("CE16OV",          2, "Civilian employment (yoy growth, %)"),
    "AWH":   ("AWHMAN",          1, "Average weekly hours, manufacturing"),
    "CPI":   ("CPIAUCSL",        2, "Consumer price index, all items (yoy inflation, %)"),
    "AHE":   ("CES3000000008x",  2, "Real average hourly earnings, manufacturing (yoy growth, %)"),
    "FFR":   ("FEDFUNDS",        1, "Effective federal funds rate (%)"),
    "SP500": ("S&P 500",         3, "S&P 500 composite index (qoq return, %)"),
}

GPVAR16_EXTRA: Dict[str, Tuple[str, int, str]] = {
    "PCE":    ("PCECC96",        2, "Real personal consumption expenditures (yoy growth, %)"),
    "FPI":    ("FPIx",           2, "Real private fixed investment (yoy growth, %)"),
    "UNRATE": ("UNRATE",         1, "Civilian unemployment rate (%)"),
    "AWHG":   ("CES0600000007",  1, "Average weekly hours, goods-producing"),
    "CLAIMS": ("CLAIMSx",        2, "Initial claims (yoy growth, %)"),
    "HOUST":  ("HOUST",          2, "Housing starts (yoy growth, %)"),
    "AHEALL": ("CES0600000008",  2, "Average hourly earnings, goods-producing (yoy growth, %)"),
    "M2":     ("M2REAL",         2, "Real M2 money stock (yoy growth, %)"),
}

NBER_RECESSIONS: List[Tuple[str, str]] = [
    ("1960-04-01", "1961-02-28"), ("1969-12-01", "1970-11-30"), ("1973-11-01", "1975-03-31"),
    ("1980-01-01", "1980-07-31"), ("1981-07-01", "1982-11-30"), ("1990-07-01", "1991-03-31"),
    ("2001-03-01", "2001-11-30"), ("2007-12-01", "2009-06-30"), ("2020-02-01", "2020-04-30"),
]

PAPER_SUBSAMPLES = {
    "1970Q1-1984Q4 (great inflation)":       ("1970-01-01", "1984-12-31"),
    "1985Q1-2006Q4 (great moderation)":      ("1985-01-01", "2006-12-31"),
    "2007Q1-2019Q4 (post great moderation)": ("2007-01-01", "2019-12-31"),
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_fred_qd_subset(path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Return the bundled FRED-QD subset (quarterly PeriodIndex) and its transformation codes."""
    path = path or os.path.join(DATA_DIR, "fredqd_subset.csv")
    raw = pd.read_csv(path)
    tcodes = raw.iloc[0, 1:].astype(int)
    df = raw.iloc[1:].copy()
    df["sasdate"] = pd.PeriodIndex(pd.to_datetime(df["sasdate"], format="%m/%d/%Y"), freq="Q")
    df = df.set_index("sasdate").astype(float)
    df.index.name = "quarter"
    tcodes.index = df.columns
    return df, tcodes


def load_jln_uncertainty(h: int = 1, freq: str = "Q", agg: str = "mean", path: Optional[str] = None) -> pd.Series:
    """JLN macro uncertainty for horizon ``h`` in {1, 3, 12}; aggregated to quarterly if ``freq="Q"``."""
    path = path or os.path.join(DATA_DIR, "jln_macro_uncertainty.csv")
    raw = pd.read_csv(path, parse_dates=["date"])
    s = raw.set_index("date")[f"h{h}"].rename("UNC")
    s.index = pd.PeriodIndex(s.index, freq="M")
    if freq.upper().startswith("Q"):
        s = s.groupby(s.index.asfreq("Q")).agg(agg)
        s.index.name = "quarter"
    return s


def fetch_fred_qd(url: str = FRED_QD_URL) -> Tuple[pd.DataFrame, pd.Series]:
    """Download the full current FRED-QD vintage (requires internet access)."""
    raw = pd.read_csv(url)
    raw = raw[~raw["sasdate"].astype(str).str.lower().eq("factors")]
    tcodes = raw.iloc[0, 1:].astype(int)
    df = raw.iloc[1:].copy()
    df["sasdate"] = pd.PeriodIndex(pd.to_datetime(df["sasdate"]), freq="Q")
    df = df.set_index("sasdate").astype(float)
    tcodes.index = df.columns
    return df, tcodes


# --------------------------------------------------------------------------- #
# Transformations
# --------------------------------------------------------------------------- #
def transform_series(x: pd.Series, code: int, scheme: str = "paper") -> pd.Series:
    """
    Apply a transformation code.

    scheme="paper" (Table B.1): 1 none | 2 yoy growth (100*dlog4) | 3 qoq growth (100*dlog) | 4 qoq % change.
    scheme="fred"  (FRED-QD):   1 none | 2 diff | 3 diff^2 | 4 log | 5 dlog | 6 d^2 log | 7 pct change diff.
    """
    if scheme == "paper":
        if code == 1:
            return x
        if code == 2:
            return 100.0 * (np.log(x) - np.log(x.shift(4)))
        if code == 3:
            return 100.0 * np.log(x).diff()
        if code == 4:
            return 100.0 * x.pct_change()
    elif scheme == "fred":
        if code == 1:
            return x
        if code == 2:
            return x.diff()
        if code == 3:
            return x.diff().diff()
        if code == 4:
            return np.log(x)
        if code == 5:
            return 100.0 * np.log(x).diff()
        if code == 6:
            return 100.0 * np.log(x).diff().diff()
        if code == 7:
            return 100.0 * x.pct_change().diff()
    raise ValueError(f"unknown code {code} for scheme {scheme}")


def build_dataset(size: int = 8, start: str = "1970Q1", end: str = "2019Q4", uncertainty_h: int = 1,
                  order: Optional[Sequence[str]] = None, scheme: str = "paper",
                  fred: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assemble the GP-VAR-8 (or 16) data set of the paper.

    Returns ``(Y, info)`` where ``Y`` has a quarterly ``DatetimeIndex`` (quarter
    end) and ``info`` documents mnemonic, transformation and description.
    """
    fred = load_fred_qd_subset()[0] if fred is None else fred
    spec = dict(GPVAR8)
    if size >= 16:
        spec.update(GPVAR16_EXTRA)
    unc = load_jln_uncertainty(h=uncertainty_h)
    cols, rows = {}, []
    for name, (mn, code, desc) in spec.items():
        src = unc if mn == "JLN" else fred[mn]
        cols[name] = transform_series(src, code, scheme)
        rows.append([name, mn, code, desc])
    Y = pd.DataFrame(cols)
    Y = Y.loc[pd.Period(start, "Q"):pd.Period(end, "Q")]
    if order is not None:
        Y = Y[list(order)]
    Y.index = Y.index.to_timestamp(how="end").normalize()
    Y.index.name = "date"
    if Y.isna().any().any():
        missing = Y.columns[Y.isna().any()].tolist()
        raise ValueError(f"missing values in {missing} for the requested sample")
    info = pd.DataFrame(rows, columns=["variable", "mnemonic", "transformation", "description"]).set_index("variable")
    return Y, info


def standardize(Y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    mu, sd = Y.mean(), Y.std(ddof=1)
    return (Y - mu) / sd, mu, sd


def recession_mask(dates: pd.Index) -> np.ndarray:
    """Boolean array marking quarters that overlap an NBER recession."""
    d = pd.DatetimeIndex(dates)
    m = np.zeros(len(d), dtype=bool)
    for s, e in NBER_RECESSIONS:
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        # quarter containing date d overlaps [s, e] if quarter start <= e and quarter end >= s
        qs = d.to_period("Q").start_time
        qe = d.to_period("Q").end_time
        m |= (qs <= e) & (qe >= s)
    return m


def describe(Y: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics table."""
    d = Y.describe().T[["mean", "std", "min", "max"]]
    d["skew"] = Y.skew()
    d["kurt"] = Y.kurt()
    d["AR(1)"] = [Y[c].autocorr(1) for c in Y.columns]
    return d
