import pandas as pd

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