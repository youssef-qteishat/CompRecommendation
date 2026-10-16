"""Stage 0: profile the raw flattened tables.

Writes:
  reports/profile_raw.csv       one row per column (dtype, missing %, top values)
  reports/value_counts.csv      FULL value counts for categorical columns
  reports/profile_answers.md    answers to the stage-0 questions
"""
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

INTERIM = Path("data/interim")
REPORTS = Path("reports")

# Strings that mean "missing" even though they are not null.
BLANK_TOKENS = {"", "n/a", "na", "null", "nan", "unknown", "otherinput"}

CATEGORICAL = {
    "subjects": ["structure_type", "style", "basement", "cooling", "heating", "fuel_type",
                 "condition", "num_baths", "num_beds", "units_sq_ft", "site_dimensions"],
    "comps": ["prop_type", "stories", "basement_finish", "condition", "bath_count",
              "bed_count", "location_similarity", "lot_size"],
    "properties": ["property_sub_type", "structure_type", "style", "levels", "basement",
                   "cooling", "heating", "city", "province"],
}


def profile(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({
            "table": table_name,
            "column": col,
            "dtype": str(s.dtype),
            "missing_pct": round(s.isna().mean() * 100, 1),
            "n_unique": s.nunique(dropna=True),
            "top_values": s.astype(str).value_counts().head(15).to_dict(),
        })
    return pd.DataFrame(rows)


def _blank_mask(s: pd.Series) -> pd.Series:
    """True where a value is null OR a blank-like string."""
    if pd.api.types.is_numeric_dtype(s):
        return s.isna()
    t = s.astype("string").str.strip().str.lower()
    return s.isna() | t.isin(BLANK_TOKENS).fillna(False)


def write_value_counts(tables: dict[str, pd.DataFrame]) -> None:
    rows = []
    for tname, cols in CATEGORICAL.items():
        df = tables[tname]
        for col in cols:
            if col not in df.columns:
                continue
            vc = df[col].astype("string").fillna("<NA>").value_counts()
            for val, cnt in vc.items():
                rows.append({
                    "table": tname,
                    "column": col,
                    "value": val,
                    "value_repr": repr(val),          # shows trailing spaces
                    "normalized": str(val).strip().lower(),
                    "count": int(cnt),
                })
    pd.DataFrame(rows).to_csv(REPORTS / "value_counts.csv", index=False)


def _block(title: str, body: str) -> list[str]:
    return [f"## {title}", "", "```", body, "```", ""]


def write_answers(tables: dict[str, pd.DataFrame]) -> None:
    s, c, p = tables["subjects"], tables["comps"], tables["properties"]
    lines = ["# Raw data profile: answers", ""]

    # 1. Missingness: true nulls vs nulls + blank strings
    rows = []
    for tname, df in tables.items():
        for col in df.columns:
            rows.append({
                "table": tname,
                "column": col,
                "null_pct": round(df[col].isna().mean() * 100, 1),
                "null_or_blank_pct": round(_blank_mask(df[col]).mean() * 100, 1),
            })
    miss = pd.DataFrame(rows)
    miss = miss[miss["null_or_blank_pct"] > 0].sort_values(
        ["table", "null_or_blank_pct"], ascending=[True, False])
    lines += _block("Missingness (null vs null-or-blank)", miss.to_string(index=False))

    # 2. Records per appraisal, and sharing across appraisals
    n_props = p.groupby("orderID").size()
    n_comps = c.groupby("orderID").size()
    shared = p.groupby("raw_id")["orderID"].nunique()
    body = "\n".join([
        f"appraisals: {s['orderID'].nunique()}",
        f"property records: {len(p)}   comps: {len(c)}",
        "",
        "candidates per appraisal:",
        n_props.describe().round(1).to_string(),
        "",
        "comps per appraisal:",
        n_comps.describe().round(1).to_string(),
        "",
        f"appraisals with < 10 candidates: {(n_props < 10).sum()}",
        f"unique raw_id: {p['raw_id'].nunique()}",
        f"raw_ids appearing in more than one appraisal: {(shared > 1).sum()}",
    ])
    lines += _block("Records per appraisal", body)

    # 3. Dates: candidates after the effective date, comp ages
    eff = pd.to_datetime(s["effective_date"].astype("string").str.strip(),
                         format="%b/%d/%Y", errors="coerce")
    subj_dates = pd.DataFrame({"orderID": s["orderID"], "eff": eff})

    close = pd.to_datetime(p["close_date"].astype("string").str.strip(),
                           format="%Y-%m-%d", errors="coerce")
    m = pd.DataFrame({"orderID": p["orderID"], "close": close}).merge(
        subj_dates, on="orderID", how="left")
    after = m["close"] > m["eff"]
    per_appraisal = after.groupby(m["orderID"]).mean()

    sale = pd.to_datetime(c["sale_date"].astype("string").str.strip(),
                          format="%b/%d/%Y", errors="coerce")
    mc = pd.DataFrame({"orderID": c["orderID"], "sale": sale}).merge(
        subj_dates, on="orderID", how="left")
    comp_age_days = (mc["eff"] - mc["sale"]).dt.days

    body = "\n".join([
        f"unparsed dates: effective={eff.isna().sum()}, candidate close={close.isna().sum()}, "
        f"comp sale={sale.isna().sum()}",
        f"effective dates: {eff.min().date()} to {eff.max().date()}",
        f"candidate close dates: {close.min().date()} to {close.max().date()}",
        f"comp sale dates: {sale.min().date()} to {sale.max().date()}",
        "",
        f"candidates closing AFTER the effective date: {after.sum()} of {len(m)} "
        f"({after.mean():.1%})",
        "share per appraisal:",
        per_appraisal.describe().round(3).to_string(),
        "",
        f"comps sold AFTER the effective date: {(comp_age_days < 0).sum()}",
        f"comps sold BEFORE the earliest candidate close date: {(mc['sale'] < close.min()).sum()}",
        "comp age at effective date (days):",
        comp_age_days.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).round(0).to_string(),
    ])
    lines += _block("Dates", body)

    # 4. GLA: frequent values and which property types they come from
    gla = p["gla"]
    vc = gla.value_counts().head(30)
    top_vals = vc.head(15).index
    by_type = (p[p["gla"].isin(top_vals)]
               .groupby(["gla", "property_sub_type"]).size()
               .sort_values(ascending=False).head(40))
    comp_gla = c["gla"].astype("string")
    body = "\n".join([
        f"non-null GLA values: {gla.notna().sum()}",
        "most frequent values:",
        vc.to_string(),
        "",
        "frequent values by property_sub_type:",
        by_type.to_string(),
        "",
        "comp GLA unit counts:",
        f"  sq m: {comp_gla.str.contains(r'sq\s*m|m2', case=False, regex=True, na=False).sum()}",
        f"  sf/sqft: {comp_gla.str.contains(r'sq\s*f|sqft|\bsf\b', case=False, regex=True, na=False).sum()}",
    ])
    lines += _block("GLA", body)

    # 5. year_built: real years vs small codes
    yb = p["year_built"]
    body = "\n".join([
        f"null: {yb.isna().sum()}",
        f">= 1800 (real years): {(yb >= 1800).sum()}",
        f"< 1800 (age codes?): {(yb < 1800).sum()}",
        "values < 1800:",
        yb[yb < 1800].value_counts().to_string(),
    ])
    lines += _block("properties.year_built", body)

    (REPORTS / "profile_answers.md").write_text("\n".join(lines))


def run():
    REPORTS.mkdir(parents=True, exist_ok=True)
    tables = {name: pd.read_parquet(INTERIM / f"raw_{name}.parquet")
              for name in ["subjects", "comps", "properties"]}

    out = pd.concat([profile(df, name) for name, df in tables.items()], ignore_index=True)
    out.to_csv(REPORTS / "profile_raw.csv", index=False)
    write_value_counts(tables)
    write_answers(tables)
    log.info("wrote reports/profile_raw.csv, value_counts.csv, profile_answers.md")