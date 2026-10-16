import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

INTERIM = Path("data/interim")
REPORTS = Path("reports")


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


def run():
    REPORTS.mkdir(parents=True, exist_ok=True)
    frames = []
    for name in ["subjects", "comps", "properties"]:
        df = pd.read_parquet(INTERIM / f"raw_{name}.parquet")
        frames.append(profile(df, name))
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(REPORTS / "profile_raw.csv", index=False)
    log.info("wrote %d column profiles to reports/profile_raw.csv", len(out))