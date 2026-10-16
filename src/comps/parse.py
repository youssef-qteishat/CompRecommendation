"""Generic parsers used by the normalize step.

Changes from the first version, based on the raw profile:
  - more null tokens ('unknown', 'otherinput')
  - parse_area_sqft: unit-aware (sq ft, sq m, acres, hectares) instead of
    grabbing the first number
  - parse_baths: handles "2:1", "2F", "2F 1H", "2F1H", "2F1P", "2 Full/1Half", "2"
  - parse_beds: handles "3+1", "2 + 1"
  - age_from_year_or_age: handles years ("2016"), ages ("21", "43+/-yrs",
    "100+") and "New"
"""
import re

import numpy as np
import pandas as pd

# 'none' is deliberately absent: basement/cooling "None" is a real value.
NULL_STRINGS = {"", "n/a", "na", "n.a.", "null", "nan", "unknown", "otherinput", "-", "--"}


def clean_str(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if x is pd.NA:
        return np.nan
    s = re.sub(r"\s+", " ", str(x)).strip().strip('"').strip()
    return np.nan if s.lower() in NULL_STRINGS else s


def to_num(x):
    """'378,900' -> 378900.0; first number in the string; NaN if none."""
    s = clean_str(x)
    if pd.isna(s):
        return np.nan
    m = re.search(r"-?\d[\d,]*\.?\d*", s)
    return float(m.group().replace(",", "")) if m else np.nan


def parse_area_sqft(x, unit_rules):
    """Area in square feet.

    unit_rules: list of (regex, factor) from mappings.yaml `area_units`.
    Uses the number directly before the first matching unit, so
    "60' x 110' / 6,600 sf" -> 6600, "94 SqM" -> 1011.8, "1.25 Acres" -> 54450.
    No unit -> assume square feet.
    """
    s = clean_str(x)
    if pd.isna(s):
        return np.nan
    t = s.lower()
    for pattern, factor in unit_rules:
        m = re.search(rf"(\d[\d,]*\.?\d*)\s*(?:\+/-|\+/\-)?\s*(?:{pattern})", t)
        if m:
            return float(m.group(1).replace(",", "")) * float(factor)
    return to_num(s)


def parse_baths(x):
    """Return (full, half).

    "2:1" -> (2, 1)    "2F 1H" / "2F1H" / "2F1P" / "2 Full/1Half" -> (2, 1)
    "3F" -> (3, 0)     "2" -> (2, nan)
    """
    s = clean_str(x)
    if pd.isna(s):
        return (np.nan, np.nan)
    t = s.lower()
    m = re.fullmatch(r"(\d+)\s*:\s*(\d+)", t)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    full = re.findall(r"(\d+)\s*f", t)          # "2F", "2 Full"; "2F 1F" -> 3
    half = re.findall(r"(\d+)\s*[hp]", t)       # "1H", "1Half", "1P" (partial bath)
    if full or half:
        return (float(sum(map(int, full))) if full else np.nan,
                float(sum(map(int, half))) if half else 0.0)
    if re.fullmatch(r"\d+(\.\d+)?", t):
        return (float(t), np.nan)
    return (np.nan, np.nan)


def parse_beds(x):
    """Return (above_grade, below_grade). "3+1" -> (3, 1); "3" -> (3, 0)."""
    s = clean_str(x)
    if pd.isna(s):
        return (np.nan, np.nan)
    parts = [to_num(p) for p in str(s).split("+")]
    above = parts[0]
    below = float(np.nansum(parts[1:])) if len(parts) > 1 else 0.0
    return (above, below)


def age_from_year_or_age(x, as_of_year, min_year=1800, new_tokens=("new",)):
    """Age in years at `as_of_year`.

    "2016" -> as_of_year - 2016;  "21" / "43+/-yrs" / "100+" -> that number;
    "New" -> 0.
    """
    s = clean_str(x)
    if pd.isna(s):
        return np.nan
    t = str(s).lower()
    if any(t.startswith(tok) for tok in new_tokens):
        return 0.0
    v = to_num(s)
    if np.isnan(v):
        return np.nan
    if v >= min_year:
        return float(max(as_of_year - v, 0))
    return v


def parse_date(series: pd.Series, fmt: str) -> pd.Series:
    return pd.to_datetime(series.map(clean_str), format=fmt, errors="coerce")