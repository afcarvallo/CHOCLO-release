"""Second LLM judge (reviewer: reliance on a single judge, regional leniency).

Re-scores, with gpt-4o-2024-08-06 and the same judge prompt, (a) the four models' answers on a
stratified subsample (per region x category x difficulty cell) and (b) the 120 answers rated by
humans. Compares the second judge with the GPT-5-mini judge used in the paper and with humans.

Outputs: src/results/E22_second_judge_*.csv (gpt-4o) or E23_open_judge_*.csv (other judges)
Usage:   python src/second_judge.py [--per-cell 50]
         python src/second_judge.py --judge meta-llama/llama-3.3-70b-instruct --provider openrouter --tag E23_open_judge
"""
import argparse
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from common import CACHE, MODEL_NAMES, RESULTS, complete_models, load
from human_judge_agreement import load as load_human
from openai_eval import JUDGE_PROMPT, parse_score, read, run

SEED = 13
PROVIDERS = {"openai": ("https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY"),
             "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY")}
REGIONS = ["LATAM", "Europe", "USA"]


def gaps(x, col):
    m = x.groupby(["model", "region"])[col].mean().unstack().mul(100)
    return pd.DataFrame({"LATAM-Europe": m.LATAM - m.Europe, "LATAM-USA": m.LATAM - m.USA})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=50)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--judge", default="gpt-4o-2024-08-06")
    ap.add_argument("--provider", default="openai", choices=list(PROVIDERS))
    ap.add_argument("--tag", default="E22_second_judge")
    args = ap.parse_args()
    JUDGE, tag = args.judge, args.tag
    url, key_env = PROVIDERS[args.provider]
    key = os.environ[key_env]
    models, _ = complete_models(verbose=False)
    d = load(models)

    ref = d[d.model == d.model.iloc[0]].drop_duplicates(["region", "entity_id", "pregunta"])
    qs = (ref.groupby(["region", "categoria", "dificultad"], group_keys=False)
          .apply(lambda g: g.sample(min(len(g), args.per_cell), random_state=SEED)))[["region", "entity_id", "pregunta"]]
    x = d.merge(qs, on=["region", "entity_id", "pregunta"]).drop_duplicates(["model", "region", "entity_id", "pregunta"])
    x = x[["model", "region", "categoria", "dificultad", "entity_id", "pregunta", "respuesta", "respuesta_gpt5",
           "score_gpt", "score_f1"]].reset_index(drop=True)
    x["jid"] = "s|" + x.model + "|" + x.index.astype(str)

    h, _ = load_human()
    hi = h.groupby(["item_id", "model", "region", "score_gpt"], as_index=False).score.mean()
    items = pd.read_csv(RESULTS / "annotation" / "judge_human_agreement_sample.csv")[["item_id", "pregunta", "respuesta", "model_answer"]]
    hi = hi.merge(items, on="item_id")
    hi["jid"] = "h|" + hi.item_id.astype(str)

    jobs = [{"id": j, "model": JUDGE, "params": {"temperature": 0},
             "prompt": JUDGE_PROMPT.format(question=q, reference=r, answer=a)}
            for j, q, r, a in zip(pd.concat([x.jid, hi.jid]), pd.concat([x.pregunta, hi.pregunta]),
                                  pd.concat([x.respuesta, hi.respuesta]), pd.concat([x.respuesta_gpt5, hi.model_answer]))]
    path = CACHE / "openai" / f"judge_{JUDGE.replace('/', '_')}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    print("answers:", len(x), "| human items:", len(hi), flush=True)
    run(jobs, path, key, args.workers, url)
    js = read(path).rename(columns={"id": "jid"})
    js["judge2"] = js.text.map(parse_score)
    x = x.merge(js[["jid", "judge2"]], on="jid", how="left").dropna(subset=["judge2", "score_gpt"])
    hi = hi.merge(js[["jid", "judge2"]], on="jid", how="left")
    x["model"] = x.model.map(lambda m: MODEL_NAMES.get(m, m))

    # (a) Agreement between judges, overall and by region.
    agree = [dict(group="all", n=len(x), spearman=stats.spearmanr(x.score_gpt, x.judge2)[0],
                  mean_judge1=x.score_gpt.mean(), mean_judge2=x.judge2.mean())]
    for r in REGIONS:
        g = x[x.region == r]
        agree.append(dict(group=r, n=len(g), spearman=stats.spearmanr(g.score_gpt, g.judge2)[0],
                          mean_judge1=g.score_gpt.mean(), mean_judge2=g.judge2.mean()))
    agree = pd.DataFrame(agree)

    # (b) Regional gaps under each judge (and lexical F1), by model and by category.
    g1, g2, g3 = gaps(x, "score_gpt"), gaps(x, "judge2"), gaps(x, "score_f1")
    by_model = pd.concat({"judge1": g1, "judge2": g2, "f1": g3}, axis=1)
    cat = x.groupby(["categoria", "region"])[["score_gpt", "judge2"]].mean().mul(100).unstack()
    by_cat = pd.DataFrame({f"{j}_{c}": cat[(j, "LATAM")] - cat[(j, r)]
                           for j in ["score_gpt", "judge2"] for c, r in [("eur", "Europe"), ("usa", "USA")]})
    # Regional leniency of the paper's judge relative to the second judge: at equal second-judge score.
    x["j2bin"] = pd.cut(x.judge2, [-0.01, 0, 0.25, 0.5, 0.75, 0.99, 1.0])
    f = smf.ols("I(100*score_gpt) ~ C(region, Treatment('LATAM')) + C(j2bin) + C(model) + C(categoria) + C(dificultad)",
                x).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.entity_id)[0]})
    cond = pd.DataFrame({"coef": f.params, "lo": f.conf_int()[0], "hi": f.conf_int()[1], "p": f.pvalues}).filter(like="region", axis=0)

    # (c) Human items: both judges vs. the mean human score, deviation by region.
    hum = []
    for grp, g in [("all", hi)] + [(r, hi[hi.region == r]) for r in REGIONS]:
        hum.append(dict(group=grp, n=len(g), rho_judge1=stats.spearmanr(g.score, g.score_gpt)[0],
                        rho_judge2=stats.spearmanr(g.score, g.judge2)[0],
                        dev_judge1=(g.score_gpt - g.score).mean(), dev_judge2=(g.judge2 - g.score).mean()))
    hum = pd.DataFrame(hum)
    kw = stats.kruskal(*[(g.judge2 - g.score).values for _, g in hi.groupby("region")]).pvalue

    summary = dict(answers=len(x), questions=x.pregunta.nunique(), human_items=len(hi),
                   spearman_judges=agree.spearman.iloc[0],
                   gap_corr_model_region=np.corrcoef(g1.values.ravel(), g2.values.ravel())[0, 1],
                   gap_corr_category=np.corrcoef(by_cat[["score_gpt_eur", "score_gpt_usa"]].values.ravel(),
                                                 by_cat[["judge2_eur", "judge2_usa"]].values.ravel())[0, 1],
                   human_region_dev_kruskal_p_judge2=kw)
    agree.to_csv(RESULTS / f"{tag}_agreement.csv", index=False)
    by_model.to_csv(RESULTS / f"{tag}_gaps_model.csv")
    by_cat.to_csv(RESULTS / f"{tag}_gaps_category.csv")
    cond.to_csv(RESULTS / f"{tag}_conditional.csv")
    hum.to_csv(RESULTS / f"{tag}_human.csv", index=False)
    pd.Series(summary).to_csv(RESULTS / f"{tag}_summary.csv")
    for t in [pd.Series(summary), agree, by_model, by_cat, cond, hum]:
        print(t.round(3).to_string(), "\n")


if __name__ == "__main__":
    main()
