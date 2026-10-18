#!/usr/bin/env python3
import argparse
import csv
import glob
import os
from collections import defaultdict


def at_recall(rows, target):
    rows = sorted(rows, key=lambda r: int(r["L"]))
    prev = None
    for r in rows:
        rec = float(r["Recall"])
        if rec >= target:
            if prev is None or float(prev["Recall"]) >= rec:
                return float(r["QPS"]), float(r["IOs"])
            r0, r1 = float(prev["Recall"]), rec
            f = (target - r0) / (r1 - r0)
            q = float(prev["QPS"]) + f * (float(r["QPS"]) - float(prev["QPS"]))
            io = float(prev["IOs"]) + f * (float(r["IOs"]) - float(prev["IOs"]))
            return q, io
        prev = r
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tables", nargs="*")
    p.add_argument("--recall", type=float, default=0.9)
    args = p.parse_args()

    tables = args.tables
    if not tables:
        root = os.environ.get("NEMO_RESULTS", os.path.join(os.path.dirname(__file__), "..", "results"))
        tables = sorted(glob.glob(os.path.join(root, "*.tsv")))

    series = defaultdict(list)
    for path in tables:
        with open(path) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                key = (r["dataset"], r["sel"], r["threads"], r["system"], r["R_max"])
                series[key].append(r)

    groups = defaultdict(dict)
    for (ds, sel, t, system, r_max), rows in series.items():
        groups[(ds, sel, t)][(system, r_max)] = at_recall(rows, args.recall)

    print(f"QPS and mean I/Os per query at recall {args.recall}")
    print(f"{'dataset':<12}{'sel%':>5}{'T':>4}  {'system':<16}{'R_max':>6}{'QPS':>10}{'IOs':>9}{'NemoANN/x':>11}")
    for (ds, sel, t) in sorted(groups, key=lambda k: (k[0], int(k[1]), int(k[2]))):
        g = groups[(ds, sel, t)]
        ref = g.get(("nemoann", "32"))
        for (system, r_max) in sorted(g):
            v = g[(system, r_max)]
            if v is None:
                print(f"{ds:<12}{sel:>5}{t:>4}  {system:<16}{r_max:>6}{'n/a':>10}{'n/a':>9}")
                continue
            ratio = f"{ref[0] / v[0]:.1f}x" if ref and v[0] > 0 else "-"
            print(f"{ds:<12}{sel:>5}{t:>4}  {system:<16}{r_max:>6}{v[0]:>10.0f}{v[1]:>9.1f}{ratio:>11}")


if __name__ == "__main__":
    main()
