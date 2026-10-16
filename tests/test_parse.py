import math
import yaml
from comps.parse import (to_num, clean_str, parse_area_sqft, parse_baths,
                         parse_beds, age_from_year_or_age)

UNITS = yaml.safe_load(open("config/mappings.yaml"))["area_units"]

def test_to_num():
    assert to_num("378,900") == 378900.0
    assert math.isnan(to_num("n/a"))

def test_clean_str():
    assert clean_str('"Twin Oak Meadows"') == "Twin Oak Meadows"
    assert clean_str("Calgary ") == "Calgary"
    assert clean_str("None") == "None"          # meaningful for basement/cooling

def test_area():
    assert parse_area_sqft("1044 SqFt", UNITS) == 1044.0
    assert round(parse_area_sqft("94 SqM", UNITS)) == 1012
    assert parse_area_sqft("1.25 Acres", UNITS) == 54450.0
    assert parse_area_sqft("60' x 110' / 6,600 sf", UNITS) == 6600.0
    assert parse_area_sqft("3332+/-SqFt", UNITS) == 3332.0

def test_baths():
    assert parse_baths("2:1") == (2.0, 1.0)
    assert parse_baths("2F1P") == (2.0, 1.0)
    assert parse_baths("2 Full/1Half") == (2.0, 1.0)
    assert parse_baths("3F") == (3.0, 0.0)

def test_beds():
    assert parse_beds("3+1") == (3.0, 1.0)
    assert parse_beds("2 + 1") == (2.0, 1.0)

def test_age():
    assert age_from_year_or_age("2016", 2025) == 9.0
    assert age_from_year_or_age("43+/-yrs", 2025) == 43.0
    assert age_from_year_or_age("New", 2025) == 0.0
