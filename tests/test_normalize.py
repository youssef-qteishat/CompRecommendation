import math

import pandas as pd

from comps.normalize import CFG, apply_rules, join_text


def rule(field, value):
    return apply_rules(pd.Series([value]), CFG[field], field).iloc[0]


def test_ordering_traps():
    assert rule("style_rules", "1 1/2 Storey") == "storey_1_5"
    assert rule("style_rules", "Bungalow Raised") == "bi_level"
    assert rule("style_rules", "3 Level Split") == "split_level"
    assert rule("structure_rules", "Semi-Detached") == "semi_detached"
    assert math.isnan(rule("cooling_rules", "Baseboard"))          # heating value in cooling
    assert rule("cooling_rules", "Baseboard, Heat Pump") == "heat_pump"
    assert rule("heating_rules", "HOT WATER BASEBOARD") == "hot_water"


def test_basement_first_rule_wins():
    finish = CFG["basement"]["finish_rules"]
    assert apply_rules(pd.Series(["Finished, None"]), finish, "b").iloc[0] == 0


def test_unmapped_is_other():
    assert rule("style_rules", "zzz") == "other"


def test_join_text():
    out = join_text(pd.Series(["Residential", None]), pd.Series(["Residential", "n/a"]))
    assert out.iloc[0] == "Residential"
    assert pd.isna(out.iloc[1])
