"""Audit of the entity-relevance filter (reviewer NCxe) from the task app (--task relevance).

Outputs: src/results/E19_relevance_*.csv
Usage:  python src/relevance_audit.py
"""
import itertools
import json

import numpy as np
import pandas as pd
from scipy import stats

from common import RESULTS, ROOT
from human_validation import gwet_ac1, krippendorff_nominal

DIR = ROOT / "annotation_app" / "annotations_relevance"
LABELS = ["representative", "representative_wrong_category", "associated_only", "wrong_country", "unclear"]


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
    a = pd.DataFrame(recs).merge(items[["item_id", "region", "category", "entity", "country"]], on="item_id")
    a = a[a.label != "na"].copy()
    a["representative"] = a.label.isin(["representative", "representative_wrong_category"])
    a["category_ok"] = a.label != "representative_wrong_category"
    return a


def shares(g):
    out = {lab: (g.label == lab).mean() for lab in LABELS}
    out["representative_any"] = g.representative.mean()
    out["judgments"] = len(g)
    return pd.Series(out)


def main():
    a = load()
    by_region = a.groupby("region").apply(shares).reset_index()
    by_cat = a.groupby(["region", "category"]).apply(shares).reset_index()
    item = a.groupby(["item_id", "region", "category"]).representative.mean().reset_index()
    item["majority_representative"] = item.representative >= 0.5
    maj = item.groupby("region").majority_representative.mean()
    by_region["majority_representative"] = by_region.region.map(maj)
    chi_p = stats.chi2_contingency(pd.crosstab(item.region, item.majority_representative))[1]
    units5 = [g.label.tolist() for _, g in a.groupby("item_id") if len(g) > 1]
    units2 = [g.representative.tolist() for _, g in a.groupby("item_id") if len(g) > 1]
    pairs = [(x, y) for u in units2 for x, y in itertools.combinations(u, 2)]
    overall = dict(items=a.item_id.nunique(), judgments=len(a), representative=a.representative.mean(),
                   associated_only=(a.label == "associated_only").mean(), unclear=(a.label == "unclear").mean(),
                   wrong_category=(a.label == "representative_wrong_category").mean(),
                   wrong_country=(a.label == "wrong_country").mean(),
                   agreement_binary=np.mean([x == y for x, y in pairs]), ac1_binary=gwet_ac1(units2),
                   alpha_binary=krippendorff_nominal(units2), alpha_labels=krippendorff_nominal(units5),
                   region_chi2_p=chi_p)
    by_region.to_csv(RESULTS / "E19_relevance_by_region.csv", index=False)
    by_cat.to_csv(RESULTS / "E19_relevance_by_region_category.csv", index=False)
    pd.Series(overall).to_csv(RESULTS / "E19_relevance_overall.csv")
    item.to_csv(RESULTS / "E19_relevance_items.csv", index=False)
    print(pd.Series(overall).round(3).to_string())
    print(by_region.round(3).to_string(index=False))
    print(by_cat.pivot(index="category", columns="region", values="representative_any").round(2))


if __name__ == "__main__":
    main()
