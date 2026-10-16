import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

RAW_PATH = Path("data/raw/appraisals_dataset.json")
INTERIM = Path("data/interim")


def flatten(appraisals: list[dict]):
    subjects, comps, props = [], [], []
    for a in appraisals:
        oid = str(a["orderID"])
        subjects.append({"orderID": oid, **a["subject"]})
        for i, c in enumerate(a.get("comps", [])):
            comps.append({"orderID": oid, "comp_pos": i, **c})
        for i, p in enumerate(a.get("properties", [])):
            rec = {"orderID": oid, "prop_pos": i, **p}
            rec["raw_id"] = rec.pop("id", None)
            props.append(rec)
    return pd.DataFrame(subjects), pd.DataFrame(comps), pd.DataFrame(props)


def run():
    with open(RAW_PATH) as f:
        appraisals = json.load(f)["appraisals"]  # adjust to the real top-level key

    n_all = len(appraisals)
    appraisals = [a for a in appraisals if a.get("subject")]
    log.info("appraisals: %d total, %d dropped (no subject)", n_all, n_all - len(appraisals))

    raw_subjects, raw_comps, raw_props = flatten(appraisals)

    INTERIM.mkdir(parents=True, exist_ok=True)
    raw_subjects.to_parquet(INTERIM / "raw_subjects.parquet", index=False)
    raw_comps.to_parquet(INTERIM / "raw_comps.parquet", index=False)
    raw_props.to_parquet(INTERIM / "raw_properties.parquet", index=False)
    log.info("rows: subjects=%d comps=%d properties=%d",
             len(raw_subjects), len(raw_comps), len(raw_props))