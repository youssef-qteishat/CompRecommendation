import argparse
import logging
import time

from src.comps import load, profile, normalize, dedupe, link, tables, validate

STEPS = [
    ("load", load.run),
    ("profile", profile.run),
    ("normalize", normalize.run),
    ("dedupe", dedupe.run),
    ("link", link.run),
    ("tables", tables.run),
    ("validate", validate.run),
]
NAMES = [name for name, _ in STEPS]

def main():
    ap = argparse.ArgumentParser(description="Comp recommender data pipeline")
    ap.add_argument("--only", choices=NAMES, help="run a single step")
    ap.add_argument("--start", choices=NAMES, help="run from this step onward")
    ap.add_argument("--stop", choices=NAMES, help="stop after this step")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.only:
        selected = [args.only]
    else:
        lo = NAMES.index(args.start) if args.start else 0
        hi = NAMES.index(args.stop) + 1 if args.stop else len(NAMES)
        selected = NAMES[lo:hi]

    for name, fn in STEPS:
        if name in selected:
            t0 = time.time()
            logging.info("▶ %s", name)
            fn()
            logging.info("✔ %s (%.1fs)", name, time.time() - t0)

if __name__ == "__main__":
    main()