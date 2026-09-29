"""Analyses of the recent-model subsample (E12) and Wikipedia page views (E13).

Run after openai_eval.py and pageviews.py:  python src/analyze_extra.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from common import DATA, FIGURES, METRICS, RESULTS, complete_models, entity_scores, holm, load, token_f1
from experiments import CATEGORY_ORDER, REGION_COLOR, REGION_MARKER, REGION_ORDER, boot_means, save

NEW_MODEL = "gpt-5.5"


# ---------------------------------------------------------------------------
# E12. Recent model on a stratified subsample, all models re-judged identically.
# ---------------------------------------------------------------------------

def _gap(q, metric, a="LATAM", b="USA"):
    out = {}
    for r in (a, b):
        g = q[q.region == r].groupby("entity_id")[metric].agg(["sum", "count"])
        out[r] = boot_means(g["sum"].to_numpy(), g["count"].to_numpy(), b=2000)
    diff = out[a] - out[b]
    point = q[q.region == a][metric].mean() - q[q.region == b][metric].mean()
    p = max(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean())), 1 / 2000)
    return 100 * point, 100 * np.percentile(diff, 2.5), 100 * np.percentile(diff, 97.5), p


def recent_model():
    s = pd.read_csv(RESULTS / "E12_openai_subsample_scores.csv")
    s["f1"] = [token_f1(a, b) for a, b in zip(s.respuesta, s.answer.fillna(""))]
    rows = []
    for model, g in s.groupby("model"):
        row = {"model": model, "n": g.qid.nunique()}
        for r in REGION_ORDER:
            row[f"judge {r}"] = 100 * g[g.region == r].judge.mean()
            row[f"f1 {r}"] = 100 * g[g.region == r].f1.mean()
        for ref in ["Europe", "USA"]:
            pt, lo, hi, p = _gap(g, "judge", "LATAM", ref)
            row.update({f"gap {ref}": pt, f"gap {ref} lo": lo, f"gap {ref} hi": hi, f"p {ref}": p})
        rows.append(row)
    t = pd.DataFrame(rows)
    pcols = [c for c in t.columns if c.startswith("p ")]
    adj = holm(t[pcols].to_numpy().ravel()).reshape(len(t), len(pcols))
    for k, c in enumerate(pcols):
        t[c.replace("p ", "p_holm ")] = adj[:, k]
    save(t, "E12_recent_model_regional")

    cat = []
    for (model, c), g in s.groupby(["model", "categoria"]):
        for ref in ["Europe", "USA"]:
            pt, lo, hi, p = _gap(g, "judge", "LATAM", ref)
            cat.append(dict(model=model, category=c, comparison=f"LATAM-{ref}", gap=pt, ci_lo=lo, ci_hi=hi, p=p))
    cat = pd.DataFrame(cat)
    cat["p_holm"] = cat.groupby("model").p.transform(holm)
    save(cat, "E12_recent_model_category")

    # Agreement between the re-judged scores and the scores shipped with the data.
    old = s[s.model != NEW_MODEL].dropna(subset=["judge", "original_judge"])
    agree = []
    for model, g in old.groupby("model"):
        agree.append(dict(model=model, n=len(g), spearman=stats.spearmanr(g.judge, g.original_judge)[0],
                          pearson=stats.pearsonr(g.judge, g.original_judge)[0],
                          mae=(g.judge - g.original_judge).abs().mean(),
                          mean_rejudged=100 * g.judge.mean(), mean_original=100 * g.original_judge.mean()))
    agree = pd.DataFrame(agree)
    save(agree, "E12_rejudge_agreement")
    plot_recent_category(cat)
    return t, cat, agree


def plot_recent_category(cat):
    t = cat[cat.model == NEW_MODEL]
    fig, ax = plt.subplots(figsize=(3.1, 2.4))
    ypos = {c: i for i, c in enumerate(CATEGORY_ORDER[::-1])}
    ax.axvline(0, color="#5c5c58", lw=1)
    for k, ref in enumerate(["Europe", "USA"]):
        tr = t[t.comparison == f"LATAM-{ref}"]
        y = tr.category.map(ypos) + (0.17 if k == 0 else -0.17)
        ax.errorbar(tr.gap, y, xerr=[tr.gap - tr.ci_lo, tr.ci_hi - tr.gap], fmt=REGION_MARKER[ref], ms=4.5,
                    color=REGION_COLOR[ref], mec="white", mew=0.6, elinewidth=1.2, capsize=0, label=f"LATAM $-$ {ref}")
    ax.set_yticks(range(len(CATEGORY_ORDER)))
    ax.set_yticklabels([c.replace("_", " ") for c in CATEGORY_ORDER[::-1]])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Gap in LLM judge (points)")
    ax.legend(frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.4, 1.0), handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES / "recent_model_category_gaps.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# E13. Wikipedia page views: correlation with scores, and the regional gap
# controlling for popularity.
# ---------------------------------------------------------------------------

def pageviews():
    pv = pd.read_csv(RESULTS / "E13_pageviews.csv")
    models, _ = complete_models(verbose=False)
    d = load(models)
    e = entity_scores(d).merge(pv[["entity_id", "match", "pageviews_60d"]], on="entity_id", how="left")
    e["log_pv"] = np.log10(e.pageviews_60d + 1)
    cov = (pv.assign(has=pv.pageviews_60d.notna()).groupby("region")
           .agg(entities=("entity_id", "size"), exact=("match", lambda m: (m == "exact").mean()),
                search=("match", lambda m: (m == "search").mean()), with_views=("has", "mean"),
                median_views=("pageviews_60d", "median")))
    save(cov.reset_index(), "E13_pageview_coverage")

    ok = e[(e.match == "exact") & e.pageviews_60d.notna()]
    rows = []
    for (model, region), g in list(ok.groupby(["model", "region"])) + [((m, "All"), g) for m, g in ok.groupby("model")]:
        for metric, mname in METRICS.items():
            rho, p = stats.spearmanr(g.log_pv, g[metric], nan_policy="omit")
            within = [stats.spearmanr(h.log_pv, h[metric], nan_policy="omit")[0] for _, h in g.groupby("categoria") if len(h) > 30]
            rows.append(dict(model=model, region=region, metric=mname, n=len(g), spearman=rho, p=p,
                             within_category=np.nanmean(within)))
    corr = pd.DataFrame(rows)
    save(corr, "E13_pageview_spearman")

    # Web hits vs page views (LATAM) as a sanity check of the two popularity measures.
    wh = pd.read_csv(DATA / "web_hits_latam.csv").rename(columns={"entity": "entidad"})
    lat = pv[(pv.region == "LATAM") & (pv.match == "exact")].merge(wh, on="entidad")
    hits_vs_pv = stats.spearmanr(np.log10(lat.web_hits + 1), np.log10(lat.pageviews_60d + 1), nan_policy="omit")[0]

    # Regional gap with and without controlling for popularity (entity-level OLS).
    reg = []
    for model, g in ok.groupby("model"):
        g = g.copy()
        g["pv_decile"] = pd.qcut(g.log_pv.rank(method="first"), 10, labels=False)
        for label, f in [("category", "score_gpt ~ C(region, Treatment('LATAM')) + C(categoria)"),
                         ("category + popularity", "score_gpt ~ C(region, Treatment('LATAM')) + C(categoria) + C(pv_decile)")]:
            m = smf.ols(f, data=g).fit(cov_type="HC1")
            for ref in ["Europe", "USA"]:
                k = f"C(region, Treatment('LATAM'))[T.{ref}]"
                lo, hi = m.conf_int().loc[k]
                reg.append(dict(model=model, controls=label, comparison=f"LATAM-{ref}", gap=-100 * m.params[k],
                                ci_lo=-100 * hi, ci_hi=-100 * lo, p=m.pvalues[k]))
    reg = pd.DataFrame(reg)
    save(reg, "E13_gap_controlling_popularity")

    q = ok.assign(q=pd.qcut(ok.log_pv.rank(method="first"), 5, labels=False))
    qt = q.groupby(["model", "region", "q"]).score_gpt.agg(["mean", "sem"]).reset_index()
    qt[["mean", "sem"]] *= 100
    save(qt, "E13_pageview_quintiles")
    plot_pv(qt)
    return cov, corr, reg, hits_vs_pv


def plot_pv(qt):
    models = sorted(qt.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6.3, 2.2), sharey=True)
    for ax, model in zip(np.atleast_1d(axes), models):
        for r in REGION_ORDER:
            g = qt[(qt.model == model) & (qt.region == r)]
            ax.errorbar(g.q, g["mean"], yerr=1.96 * g["sem"], color=REGION_COLOR[r], marker=REGION_MARKER[r], ms=4,
                        mec="white", mew=0.6, lw=1.5, capsize=0, label=r)
        ax.set_title(model)
        ax.set_xticks(range(5))
        ax.set_xticklabels(["Q1\nleast", "Q2", "Q3", "Q4", "Q5\nmost"])
    np.atleast_1d(axes)[0].set_ylabel("LLM-judge score")
    fig.supxlabel("Wikipedia page-view quintile (last 60 days)", fontsize=9, y=0.02)
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(w_pad=0.6)
    fig.savefig(FIGURES / "pageview_quintiles.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    if "e12" in sys.argv or len(sys.argv) == 1:
        t, cat, agree = recent_model()
        print(t.round(2).T.to_string())
        print(cat[cat.model == NEW_MODEL].round(2).to_string())
        print(agree.round(3).to_string())
    if "e13" in sys.argv or len(sys.argv) == 1:
        cov, corr, reg, hv = pageviews()
        print(cov.to_string())
        print(corr[corr.metric == "LLM judge"].round(3).to_string())
        print(reg.round(2).to_string())
        print("spearman(web hits, page views) LATAM:", round(hv, 3))
