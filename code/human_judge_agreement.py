"""Human–judge agreement from the annotation app (reviewer NCxe W6, QYTU).

Reads src/annotation_app/annotations/*.jsonl, joins them with the blinded key (model,
judge score) and reports: human–human agreement, human–judge agreement overall and by
region (to test for region-dependent judge bias), and by model (self-preference).

Usage:  python src/human_judge_agreement.py
Outputs: src/results/E16_human_judge_*.csv
"""
import itertools
import json

import numpy as np
import pandas as pd
from scipy import stats

from common import RESULTS, ROOT

ANN = ROOT / "annotation_app" / "annotations"
KEY = RESULTS / "annotation" / "judge_human_agreement_key.csv"
SAMPLE = RESULTS / "annotation" / "judge_human_agreement_sample.csv"


def krippendorff_interval(units):
    """Krippendorff's alpha, interval metric. units: list of lists of numeric ratings."""
    units = [u for u in units if len(u) >= 2]
    values = np.concatenate([np.asarray(u, float) for u in units])
    n = len(values)
    d_o = sum(sum((a - b) ** 2 for a, b in itertools.permutations(u, 2)) / (len(u) - 1) for u in units) / n
    d_e = sum((a - b) ** 2 for a, b in itertools.permutations(values, 2)) / (n * (n - 1))
    return 1 - d_o / d_e


def load():
    recs = []
    for p in sorted(ANN.glob("*.jsonl")):
        latest = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                latest[r["item_id"]] = r
        recs += latest.values()
    a = pd.DataFrame(recs)
    a = a[a.score != "na"].copy()
    a["score"] = a.score.astype(float)
    key = pd.read_csv(KEY)
    meta = pd.read_csv(SAMPLE)[["item_id", "region", "categoria"]]
    return a.merge(key, on="item_id").merge(meta, on="item_id"), (pd.DataFrame(recs).score == "na").sum()


def main():
    a, n_na = load()
    annotators = sorted(a.annotator.unique())
    units = [g.score.tolist() for _, g in a.groupby("item_id") if len(g) >= 2]
    pairs = [(x, y) for u in units for x, y in itertools.combinations(u, 2)]
    hh = dict(annotators=len(annotators), items_multi=len(units), cannot_judge=int(n_na),
              krippendorff_alpha_interval=krippendorff_interval(units) if units else np.nan,
              exact_agreement=np.mean([x == y for x, y in pairs]) if pairs else np.nan,
              within_025=np.mean([abs(x - y) <= 0.25 for x, y in pairs]) if pairs else np.nan)
    item = a.groupby(["item_id", "model", "region", "score_gpt"], as_index=False).score.mean()
    rows = [dict(group="all", n=len(item), spearman=stats.spearmanr(item.score, item.score_gpt)[0],
                 pearson=stats.pearsonr(item.score, item.score_gpt)[0], mae=(item.score - item.score_gpt).abs().mean(),
                 mean_human=item.score.mean(), mean_judge=item.score_gpt.mean())]
    for col in ["region", "model"]:
        for k, g in item.groupby(col):
            rows.append(dict(group=f"{col}={k}", n=len(g), spearman=stats.spearmanr(g.score, g.score_gpt)[0],
                             pearson=stats.pearsonr(g.score, g.score_gpt)[0], mae=(g.score - g.score_gpt).abs().mean(),
                             mean_human=g.score.mean(), mean_judge=g.score_gpt.mean()))
    hj = pd.DataFrame(rows)
    hj["judge_minus_human"] = hj.mean_judge - hj.mean_human
    pd.Series(hh).to_csv(RESULTS / "E16_human_human_agreement.csv")
    hj.to_csv(RESULTS / "E16_human_judge_agreement.csv", index=False)

    # Pairwise statistics averaged over all annotator pairs (works for 2 or more annotators).
    from sklearn.metrics import cohen_kappa_score
    w = a.pivot_table(index="item_id", columns="annotator", values="score")
    pair_stats = []
    for x, y in itertools.combinations(w.columns, 2):
        both = w[[x, y]].dropna()
        pair_stats.append(dict(spearman=stats.spearmanr(both[x], both[y])[0],
                               kappa=cohen_kappa_score((both[x] * 4).round().astype(int), (both[y] * 4).round().astype(int),
                                                       weights="quadratic"),
                               binary=((both[x] >= 0.5) == (both[y] >= 0.5)).mean()))
    ps = pd.DataFrame(pair_stats)
    per_ann = [stats.spearmanr(g.score, g.score_gpt)[0] for _, g in a.groupby("annotator")]
    lat, oth = item[item.region == "LATAM"], item[item.region != "LATAM"]
    extra = dict(annotators=len(w.columns), human_human_spearman=ps.spearman.mean(), weighted_kappa=ps.kappa.mean(),
                 binary_agreement_human_human=ps.binary.mean(),
                 binary_agreement_judge_human=((item.score >= 0.5) == (item.score_gpt >= 0.5)).mean(),
                 latam_minus_others_bias=(lat.score_gpt - lat.score).mean() - (oth.score_gpt - oth.score).mean(),
                 latam_vs_others_mwu_p=stats.mannwhitneyu(lat.score_gpt - lat.score, oth.score_gpt - oth.score).pvalue,
                 annotator_judge_spearman_min=min(per_ann), annotator_judge_spearman_max=max(per_ann))
    pd.Series(extra).to_csv(RESULTS / "E16_human_judge_extra.csv")
    rng = np.random.default_rng(0)
    bias = []
    for r, g in item.groupby("region"):
        d = (g.score_gpt - g.score).to_numpy()
        bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
        bias.append(dict(region=r, n=len(g), judge_minus_human=d.mean(), ci_lo=np.percentile(bs, 2.5),
                         ci_hi=np.percentile(bs, 97.5)))
    pd.DataFrame(bias).to_csv(RESULTS / "E16_judge_bias_by_region.csv", index=False)
    print(pd.Series(hh).round(3).to_string())
    print(pd.Series(extra).round(3).to_string())
    print(hj.round(3).to_string(index=False))
    print(pd.DataFrame(bias).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
