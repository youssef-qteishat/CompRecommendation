"""Stages 3-4: map each source to a canonical schema, then canonicalize categories.

Reads  data/interim/raw_{subjects,comps,properties}.parquet
Writes data/interim/clean_{subjects,comps,properties}.parquet
       data/interim/comp_audit.parquet   comp-only fields, for explanations, never features
"""
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from comps.parse import (age_from_year_or_age, clean_str, parse_area_sqft, parse_baths,
                         parse_beds, parse_date, to_num)

log = logging.getLogger(__name__)
CFG = yaml.safe_load(open("config/mappings.yaml"))
INTERIM = Path("data/interim")
UNITS = CFG["area_units"]
YA = CFG["year_age"]
COMP_AUDIT_COLS = ["location_similarity", "condition", "dom", "parking", "distance_to_subject"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def txt(s: pd.Series) -> pd.Series:
    return s.map(clean_str)


def join_text(*cols: pd.Series) -> pd.Series:
    """Row-wise ' | ' join of the distinct non-missing cleaned values; NaN if none."""
    parts = pd.concat([txt(c) for c in cols], axis=1)
    out = parts.apply(lambda r: " | ".join(dict.fromkeys(r.dropna())), axis=1)
    return out.where(out.ne(""))


def split2(s: pd.Series, fn) -> tuple[pd.Series, pd.Series]:
    pairs = s.map(fn)
    return pairs.str[0].astype(float), pairs.str[1].astype(float)


def area(s: pd.Series) -> pd.Series:
    return s.map(lambda x: parse_area_sqft(x, UNITS))


def age_at(s: pd.Series, years: pd.Series) -> pd.Series:
    return pd.Series([age_from_year_or_age(x, y, YA["min_year"], tuple(YA["new_tokens"]))
                      for x, y in zip(s, years)], index=s.index, dtype=float)


def apply_rules(text: pd.Series, rules: list, field: str) -> pd.Series:
    compiled = [(re.compile(p, re.I), label) for p, label in rules]

    def match(t):
        if not isinstance(t, str) or not t.strip(" |"):
            return np.nan
        t = t.strip().lower()
        for pattern, label in compiled:
            if pattern.search(t):
                return np.nan if label is None else label     # YAML null -> missing
        return "other"

    out = text.map(match)
    unmapped = text[out.eq("other")].value_counts()
    if len(unmapped):
        log.warning("%s: %d unmapped rows, e.g. %s", field, unmapped.sum(), unmapped.head(5).to_dict())
    return out


def flag(text: pd.Series, pattern: str) -> pd.Series:
    """1/0 where text exists, NaN where it doesn't."""
    rx = re.compile(pattern, re.I)
    return text.map(lambda t: float(bool(rx.search(t))) if isinstance(t, str) else np.nan)


# ---------------------------------------------------------------------------
# stage 3: canonical schema
# ---------------------------------------------------------------------------
def map_subjects(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({"orderID": raw["orderID"]})
    df["address_raw"] = txt(raw["address"])
    df["city_raw"] = txt(raw["subject_city_province_zip"])
    df["postal_raw"] = df["address_raw"]                       # postal sits inside the address
    df["effective_date"] = parse_date(raw["effective_date"], "%b/%d/%Y")
    df["gla_sqft"] = area(raw["gla"])
    df["lot_size_sf"] = area(raw["lot_size_sf"])
    df["bedrooms_above"], df["bedrooms_below"] = split2(raw["num_beds"], parse_beds)
    df["full_baths"], df["half_baths"] = split2(raw["num_baths"], parse_baths)
    df["room_count"] = raw["room_count"].map(to_num)
    year = df["effective_date"].dt.year
    df["age_years"] = age_at(raw["subject_age"], year).fillna(age_at(raw["year_built"], year))
    df["age_is_approx"] = False
    df["structure_text"] = txt(raw["structure_type"])
    df["condo_text"] = join_text(raw["structure_type"], raw["site_dimensions"], raw["lot_size_sf"])
    df["style_text"] = txt(raw["style"])
    df["basement_raw"] = txt(raw["basement"])
    df["cooling_raw"] = txt(raw["cooling"])
    df["heating_raw"] = txt(raw["heating"])
    df["fuel_raw"] = txt(raw["fuel_type"])
    df["condition"] = txt(raw["condition"]).str.lower().map(CFG["condition_scale"])
    return df


def map_comps(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = raw[["orderID", "comp_pos"]].copy()
    df["address_raw"] = txt(raw["address"])
    df["city_raw"] = txt(raw["city_province"])
    df["postal_raw"] = df["city_raw"]                          # postal sits inside city_province
    df["close_date"] = parse_date(raw["sale_date"], "%b/%d/%Y")
    df["close_price"] = raw["sale_price"].map(to_num)
    df["gla_sqft"] = area(raw["gla"])
    df["lot_size_sf"] = area(raw["lot_size"])
    df["bedrooms_above"], df["bedrooms_below"] = split2(raw["bed_count"], parse_beds)
    df["full_baths"], df["half_baths"] = split2(raw["bath_count"], parse_baths)
    df["room_count"] = split2(raw["room_count"], parse_beds)[0]  # "6+3" -> 6 above grade
    df["age_years"] = age_at(raw["age"], df["close_date"].dt.year)
    df["age_is_approx"] = False
    df["structure_text"] = txt(raw["prop_type"])
    df["condo_text"] = join_text(raw["prop_type"], raw["lot_size"])
    df["style_text"] = txt(raw["stories"])
    df["basement_raw"] = txt(raw["basement_finish"])
    df["cooling_raw"] = df["heating_raw"] = df["fuel_raw"] = np.nan
    df["condition"] = np.nan                                   # comp condition is excluded

    audit = raw[["orderID", "comp_pos", *COMP_AUDIT_COLS]].copy()
    return df, audit


def map_properties(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[["orderID", "prop_pos", "raw_id"]].copy()
    df["address_raw"] = txt(raw["address"])
    df["city_raw"] = np.nan                                    # unreliable in this source
    df["postal_raw"] = txt(raw["postal_code"])
    df["close_date"] = parse_date(raw["close_date"], "%Y-%m-%d")
    df["close_price"] = raw["close_price"].map(to_num)
    df["gla_sqft"] = area(raw["gla"])
    df["lot_size_sf"] = area(raw["lot_size_sf"])
    df["bedrooms_above"] = raw["bedrooms"].map(to_num)
    df["bedrooms_below"] = np.nan
    df["full_baths"] = raw["full_baths"].map(to_num)
    df["half_baths"] = raw["half_baths"].map(to_num)
    df["room_count"] = raw["room_count"].map(to_num)
    yb = raw["year_built"].map(to_num)
    df["age_is_approx"] = yb.isin(YA["property_age_codes"])
    real = yb.where(yb.ge(YA["min_year"]))
    df["age_years"] = (df["close_date"].dt.year - real).clip(lower=0).where(~df["age_is_approx"], yb)
    df["structure_text"] = join_text(raw["property_sub_type"], raw["structure_type"])
    df["condo_text"] = df["structure_text"]
    df["style_text"] = join_text(raw["levels"], raw["style"], raw["structure_type"])
    df["basement_raw"] = txt(raw["basement"])
    df["cooling_raw"] = txt(raw["cooling"])
    df["heating_raw"] = df["fuel_raw"] = txt(raw["heating"])
    df["condition"] = np.nan
    df["lat"], df["lon"] = raw["latitude"], raw["longitude"]
    return df


# ---------------------------------------------------------------------------
# stage 4: canonical categories
# ---------------------------------------------------------------------------
def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["structure"] = apply_rules(df["structure_text"], CFG["structure_rules"], "structure")
    df["is_condo"] = df["condo_text"].str.contains(CFG["condo_pattern"], case=False, regex=True, na=False)
    df["style"] = apply_rules(df["style_text"], CFG["style_rules"], "style")

    fill = CFG["condo_structure_from_style"]
    need = df["structure"].isna() & df["is_condo"]
    df.loc[need, "structure"] = np.where(df.loc[need, "style"].eq("apartment"),
                                         fill["apartment"], fill["default"])

    B = CFG["basement"]
    bsmt = df["basement_raw"].where(~df["basement_raw"].str.strip().str.lower().isin(B["null_values"]))
    for col, rules in [("has_basement", B["presence_rules"]), ("basement_finish", B["finish_rules"])]:
        df[col] = apply_rules(bsmt, rules, col).replace("other", np.nan).astype(float)
    for name in ["walkout", "walkup", "separate_entrance", "suite"]:
        df[f"basement_{name}"] = flag(bsmt, B[name])
    df.loc[df["structure"].eq("apartment"), ["has_basement", "basement_finish"]] = np.nan  # not applicable

    df["cooling"] = apply_rules(df["cooling_raw"], CFG["cooling_rules"], "cooling")
    df["has_ac"] = df["cooling"].map({"central_air": 1, "heat_pump": 1, "room_unit": 1,
                                      "ac_unspecified": 1, "none": 0})
    df["heating"] = apply_rules(df["heating_raw"], CFG["heating_rules"], "heating")
    df["fuel"] = apply_rules(df["fuel_raw"], CFG["fuel_rules"], "fuel")

    df["total_baths"] = df["full_baths"] + 0.5 * df["half_baths"].fillna(0)
    df["gla_is_bucketed"] = df["gla_sqft"].isin(CFG["gla_bucket_values"])
    return df


def run():
    subjects = pd.read_parquet(INTERIM / "raw_subjects.parquet")
    comps = pd.read_parquet(INTERIM / "raw_comps.parquet")
    props = pd.read_parquet(INTERIM / "raw_properties.parquet")

    comps, audit = map_comps(comps)
    out = {
        "clean_subjects": canonicalize(map_subjects(subjects)),
        "clean_comps": canonicalize(comps),
        "clean_properties": canonicalize(map_properties(props)),
        "comp_audit": audit,
    }
    for name, df in out.items():
        df.to_parquet(INTERIM / f"{name}.parquet", index=False)
        log.info("%s: %d rows", name, len(df))
