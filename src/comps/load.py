import json
import pandas as pd

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

with open("data/raw/appraisals.json") as f:
    appraisals = json.load(f)["appraisals"]   # adjust to the real top-level key

raw_subjects, raw_comps, raw_props = flatten(appraisals)
raw_subjects.to_parquet("data/interim/raw_subjects.parquet")
raw_comps.to_parquet("data/interim/raw_comps.parquet")
raw_props.to_parquet("data/interim/raw_properties.parquet")