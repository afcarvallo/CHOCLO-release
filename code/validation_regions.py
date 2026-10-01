"""Per-region Q/A validation (reviewer NCxe W4) from the task app (src/annotation_app, --task qa).

Outputs: src/results/E18_validation_regions_*.csv
Usage:  python src/validation_regions.py
"""
import itertools
import json

import numpy as np
import pandas as pd
from scipy import stats

from common import RESULTS, ROOT
from human_validation import gwet_ac1, krippendorff_nominal

DIR = ROOT / "annotation_app" / "annotations_qa"
VALID = {"valid", "valid_grammar"}


def load():
    items = pd.DataFrame(json.loads((DIR / "items.json").read_text(encoding="utf-8")))
    recs = []
    for f in sorted(DIR.glob("*.jsonl")):
        latest = {}
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                latest[r["item_id"]] = r
        recs += latest.values()
    a = pd.DataFrame(recs).merge(items[["item_id", "region", "category", "difficulty"]], on="item_id")
    a = a[a.label != "na"].copy()
    a["valid"] = a.label.isin(VALID)
    return a


def summarize(a, tag):
    units = [g.valid.tolist() for _, g in a.groupby("item_id") if len(g) > 1]
    pairs = [(x, y) for u in units for x, y in itertools.combinations(u, 2)]
    item = a.groupby(["item_id", "region"]).valid.mean().reset_index()
    item["majority_valid"] = item.valid >= 0.5
    chi_p = stats.chi2_contingency(pd.crosstab(item.region, item.majority_valid))[1]
    by_region = a.groupby("region").agg(judgments=("valid", "size"), validation_rate=("valid", "mean"))
    by_region["majority_valid"] = item.groupby("region").majority_valid.mean()
    for lab in ["ambiguity", "incorrect", "nonsense", "valid_grammar"]:
        by_region[lab] = a.assign(x=a.label == lab).groupby("region").x.mean()
    by_region = by_region.reset_index()
    by_region.insert(0, "annotators", tag)
    overall = dict(annotators=tag, items=a.item_id.nunique(), judgments=len(a), validation_rate=a.valid.mean(),
                   percent_agreement=np.mean([x == y for x, y in pairs]), gwet_ac1=gwet_ac1(units),
                   krippendorff_alpha=krippendorff_nominal(units), region_chi2_p=chi_p)
    return by_region, overall


def main():
    a = load()
    reg, ov = summarize(a, "all")
    reg.drop(columns="annotators").to_csv(RESULTS / "E18_validation_regions_by_region.csv", index=False)
    pd.DataFrame([ov]).drop(columns="annotators").to_csv(RESULTS / "E18_validation_regions_overall.csv", index=False)
    print(reg.round(3).to_string(index=False))
    print(pd.DataFrame([ov]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
