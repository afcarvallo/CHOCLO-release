"""Reviewer-requested analyses for CHOCLO.

Run from the repo root:  python src/experiments.py
Writes CSV/LaTeX tables to src/results/ and figures to paper/figures/.
Only models with complete data for all three regions are analysed
(see common.complete_models).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from common import (DATA, FIGURES, LEGACY_LEXICAL, METRICS, RESULTS, complete_models,
                    entity_scores, holm, load)

B = 2000
SEED = 13
RNG = np.random.default_rng(SEED)
REGION_ORDER = ["LATAM", "Europe", "USA"]
# Validated categorical slots 1-3 (dataviz reference palette).
REGION_COLOR = {"LATAM": "#2a78d6", "Europe": "#eb6834", "USA": "#1baf7a"}
REGION_MARKER = {"LATAM": "o", "Europe": "s", "USA": "D"}
CATEGORY_ORDER = ["tradition", "public_figure", "object", "dish", "flora", "geography", "fauna"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8a85",
    "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "savefig.dpi": 300,
})


def save(df, name, **kw):
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / f"{name}.csv", index=kw.pop("index", False))


# ---------------------------------------------------------------------------
# Entity-cluster bootstrap helpers. A region/category mean is the question-level
# mean, i.e. sum of entity sums over sum of entity question counts; resampling
# entities keeps each entity's questions together.
# ---------------------------------------------------------------------------

def _entity_arrays(q, metric):
    g = q.groupby("entity_id")[metric].agg(["sum", "count"])
    return g["sum"].to_numpy(), g["count"].to_numpy()


def boot_means(sums, cnts, b=B, chunk=200):
    n = len(sums)
    out = np.empty(b)
    for s in range(0, b, chunk):
        k = min(chunk, b - s)
        idx = RNG.integers(0, n, (k, n))
        out[s:s + k] = sums[idx].sum(1) / cnts[idx].sum(1)
    return out


def gap_test(q_a, q_b, metric):
    """Mean(a) - mean(b), bootstrap 95% CI and two-sided bootstrap p-value."""
    sa, ca = _entity_arrays(q_a, metric)
    sb, cb = _entity_arrays(q_b, metric)
    ba, bb = boot_means(sa, ca), boot_means(sb, cb)
    diff = ba - bb
    point = sa.sum() / ca.sum() - sb.sum() / cb.sum()
    p = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))
    p = max(p, 1 / B)
    return point, np.percentile(diff, 2.5), np.percentile(diff, 97.5), p, ba, bb


# ---------------------------------------------------------------------------
# E0. Dataset statistics recomputed from the score files (Tables 2 and 4).
# ---------------------------------------------------------------------------

def dataset_stats(d):
    u = d.drop_duplicates(["region", "entity_id", "pregunta"])
    ents = u.drop_duplicates("entity_id").groupby(["categoria", "region"]).size().unstack()[REGION_ORDER]
    qs = u.groupby(["categoria", "region"]).size().unstack()[REGION_ORDER]
    tab = pd.concat({"entities": ents, "questions": qs}, axis=1)
    tab.loc["total"] = tab.sum()
    save(tab.reset_index(), "E0_dataset_stats")
    share = (ents / ents.sum()).round(3)
    save(share.reset_index(), "E0_category_share")
    c = u.drop_duplicates("entity_id").groupby(["region", "pais"]).size().rename("entities")
    cq = u.groupby(["region", "pais"]).size().rename("questions")
    save(pd.concat([c, cq], axis=1).reset_index().sort_values(["region", "entities"], ascending=[True, False]),
         "E0_country_counts")
    lang = {"note": "All questions and reference answers are in Spanish for every region."}
    (RESULTS / "E0_language.json").write_text(json.dumps(lang))
    return tab


# ---------------------------------------------------------------------------
# E1. Regional scores with CIs and LATAM-vs-reference gap tests (W1, W6).
# ---------------------------------------------------------------------------

def regional_gaps(d):
    rows, tests = [], []
    for model, dm in d.groupby("model"):
        by_r = {r: dm[dm.region == r] for r in REGION_ORDER}
        for metric, mname in METRICS.items():
            boots = {}
            for ref in ["Europe", "USA"]:
                pt, lo, hi, p, ba, bb = gap_test(by_r["LATAM"], by_r[ref], metric)
                boots["LATAM"], boots[ref] = ba, bb
                tests.append(dict(model=model, metric=mname, comparison=f"LATAM-{ref}",
                                  gap=100 * pt, ci_lo=100 * lo, ci_hi=100 * hi, p=p))
            for r in REGION_ORDER:
                s, c = _entity_arrays(by_r[r], metric)
                rows.append(dict(model=model, metric=mname, region=r, mean=100 * s.sum() / c.sum(),
                                 ci_lo=100 * np.percentile(boots[r], 2.5), ci_hi=100 * np.percentile(boots[r], 97.5)))
    means, tests = pd.DataFrame(rows), pd.DataFrame(tests)
    tests["p_holm"] = holm(tests.p)
    tests["significant"] = tests.p_holm < 0.05
    save(means, "E1_regional_means")
    save(tests, "E1_regional_gap_tests")
    return means, tests


# ---------------------------------------------------------------------------
# E2. Composition control (W2): category-standardised means and regression.
# ---------------------------------------------------------------------------

def composition_control(d):
    ent = d.drop_duplicates("entity_id")
    pooled_w = ent.groupby("categoria").size() / len(ent)
    weights = {"pooled": pooled_w, "uniform": pd.Series(1 / 7, index=pooled_w.index),
               "europe_usa": (ent[ent.region != "LATAM"].groupby("categoria").size()
                              / (ent.region != "LATAM").sum())}
    rows = []
    for model, dm in d.groupby("model"):
        for metric, mname in METRICS.items():
            cat_means = {}
            cat_boot = {}
            for (r, c), q in dm.groupby(["region", "categoria"]):
                s, n = _entity_arrays(q, metric)
                cat_means[(r, c)] = s.sum() / n.sum()
                cat_boot[(r, c)] = boot_means(s, n, b=1000)
            for wname, w in weights.items():
                std = {r: sum(w[c] * cat_means[(r, c)] for c in w.index) for r in REGION_ORDER}
                bstd = {r: sum(w[c] * cat_boot[(r, c)] for c in w.index) for r in REGION_ORDER}
                for ref in ["Europe", "USA"]:
                    diff = bstd["LATAM"] - bstd[ref]
                    p = max(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean())), 1 / 1000)
                    rows.append(dict(model=model, metric=mname, weights=wname, comparison=f"LATAM-{ref}",
                                     latam=100 * std["LATAM"], ref=100 * std[ref],
                                     gap=100 * (std["LATAM"] - std[ref]),
                                     ci_lo=100 * np.percentile(diff, 2.5), ci_hi=100 * np.percentile(diff, 97.5), p=p))
    std = pd.DataFrame(rows)
    std["p_holm"] = std.groupby("weights").p.transform(holm)
    save(std, "E2_standardised_gaps")

    # Question-level OLS with entity-clustered SEs:
    # score ~ region + category + difficulty (main effects only).
    reg = []
    for model, dm in d.groupby("model"):
        dm = dm.dropna(subset=list(METRICS)).copy()
        dm["entity_code"] = pd.factorize(dm.entity_id)[0]
        for metric, mname in METRICS.items():
            f = f"{metric} ~ C(region, Treatment('LATAM')) + C(categoria) + C(dificultad)"
            m = smf.ols(f, data=dm).fit(cov_type="cluster", cov_kwds={"groups": dm.entity_code})
            for ref in ["Europe", "USA"]:
                k = f"C(region, Treatment('LATAM'))[T.{ref}]"
                lo, hi = m.conf_int().loc[k]
                # Coefficient is ref - LATAM; report LATAM - ref for consistency.
                reg.append(dict(model=model, metric=mname, comparison=f"LATAM-{ref}",
                                gap=-100 * m.params[k], ci_lo=-100 * hi, ci_hi=-100 * lo, p=m.pvalues[k]))
    reg = pd.DataFrame(reg)
    reg["p_holm"] = holm(reg.p)
    save(reg, "E2_regression_gaps")
    return std, reg


# ---------------------------------------------------------------------------
# E3. Category-level gaps with CIs (the paper's main finding, Fig. 2).
# ---------------------------------------------------------------------------

def category_gaps(d):
    rows = []
    for (model, cat), dm in d.groupby(["model", "categoria"]):
        for metric, mname in METRICS.items():
            for ref in ["Europe", "USA"]:
                pt, lo, hi, p, _, _ = gap_test(dm[dm.region == "LATAM"], dm[dm.region == ref], metric)
                rows.append(dict(model=model, category=cat, metric=mname, comparison=f"LATAM-{ref}",
                                 gap=100 * pt, ci_lo=100 * lo, ci_hi=100 * hi, p=p))
    t = pd.DataFrame(rows)
    t["p_holm"] = t.groupby("metric").p.transform(holm)
    save(t, "E3_category_gaps")
    return t


def plot_category_gaps(t, metric="LLM judge", fname="category_gaps_judge.pdf"):
    t = t[t.metric == metric]
    models = sorted(t.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6.3, 2.5), sharey=True, sharex=True)
    ypos = {c: i for i, c in enumerate(CATEGORY_ORDER[::-1])}
    for ax, model in zip(np.atleast_1d(axes), models):
        tm = t[t.model == model]
        ax.axvline(0, color="#5c5c58", lw=1)
        for k, ref in enumerate(["Europe", "USA"]):
            tr = tm[tm.comparison == f"LATAM-{ref}"]
            y = tr.category.map(ypos) + (0.17 if k == 0 else -0.17)
            ax.errorbar(tr.gap, y, xerr=[tr.gap - tr.ci_lo, tr.ci_hi - tr.gap], fmt=REGION_MARKER[ref],
                        ms=4.5, color=REGION_COLOR[ref], mec="white", mew=0.6, elinewidth=1.2, capsize=0,
                        label=f"LATAM $-$ {ref}")
        ax.set_title(model)
        ax.set_yticks(range(len(CATEGORY_ORDER)))
        ax.set_yticklabels([c.replace("_", " ") for c in CATEGORY_ORDER[::-1]])
        ax.grid(axis="y", visible=False)
    fig.supxlabel(f"Gap in {metric} (points, LATAM minus reference region)", fontsize=9, y=0.02)
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.07), handletextpad=0.3)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(FIGURES / fname, bbox_inches="tight")
    plt.close(fig)


def plot_parity(d, fname="category_parity.pdf"):
    """Absolute score (mean of the three metrics) per category and region, one panel per model.

    Replaces the submission's LATAM-vs-Europe/USA scatter: the same values, drawn as a
    dot plot so that no category labels overlap.
    """
    x = d.assign(avg=d[list(METRICS)].mean(1))
    t = (x.groupby(["model", "categoria", "region"]).avg.mean() * 100).unstack()
    save(t.round(2).reset_index(), "E3_category_parity")
    models = sorted(t.index.get_level_values(0).unique())
    ypos = {c: i for i, c in enumerate(CATEGORY_ORDER[::-1])}
    fig, axes = plt.subplots(1, len(models), figsize=(6.3, 2.5), sharey=True)
    for ax, model in zip(np.atleast_1d(axes), models):
        tm = t.loc[model]
        y = tm.index.map(ypos)
        ax.hlines(y, tm.min(1), tm.max(1), color="#c9c9c4", lw=1.2, zorder=1)
        for r in REGION_ORDER:
            ax.scatter(tm[r], y, marker=REGION_MARKER[r], s=24, color=REGION_COLOR[r], edgecolor="white",
                       linewidth=0.6, zorder=3 if r == "LATAM" else 2, label=r)
        ax.set_title(model)
        ax.set_yticks(range(len(CATEGORY_ORDER)))
        ax.set_yticklabels([c.replace("_", " ") for c in CATEGORY_ORDER[::-1]])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Mean score (3 metrics)")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.07), handletextpad=0.2)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(FIGURES / fname, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# E4. Difficulty levels by region.
# ---------------------------------------------------------------------------

def difficulty(d):
    t = (d.groupby(["model", "region", "dificultad"])[list(METRICS)].mean() * 100).round(2)
    t["average"] = t.mean(1).round(2)
    save(t.reset_index(), "E4_difficulty_by_region")
    t2 = (d.groupby(["model", "dificultad"])[list(METRICS)].mean() * 100)
    t2["average"] = t2.mean(1)
    save(t2.round(2).reset_index(), "E4_difficulty")
    return t, t2


# ---------------------------------------------------------------------------
# E5. Country-level breakdown (reviewer QYTU).
# ---------------------------------------------------------------------------

def countries(d):
    e = entity_scores(d)
    t = e.groupby(["region", "pais", "model"]).agg(entities=("entity_id", "size"), judge=("score_gpt", "mean")).reset_index()
    wide = t.pivot_table(index=["region", "pais"], columns="model", values="judge").mul(100).round(1)
    wide["entities"] = t.groupby(["region", "pais"]).entities.first()
    wide = wide.reset_index().sort_values(["region", "entities"], ascending=[True, False])
    save(wide, "E5_country_scores")
    return wide


# ---------------------------------------------------------------------------
# E6. Popularity (Sec. 6.3): Spearman on log web hits, LATAM only.
# ---------------------------------------------------------------------------

def popularity(d):
    w = pd.read_csv(DATA / "web_hits_latam.csv").rename(columns={"entity": "entidad"})
    w["log_hits"] = np.log10(w.web_hits.astype(float) + 1)
    e = entity_scores(d[d.region == "LATAM"])
    e["entidad"] = e.entity_id.str.split("|", n=2).str[2]
    m = e.merge(w[["entidad", "web_hits", "log_hits"]], on="entidad", how="inner")
    rows = []
    for model, mm in m.groupby("model"):
        for metric, mname in METRICS.items():
            rho, p = stats.spearmanr(mm.log_hits, mm[metric], nan_policy="omit")
            r_log, _ = stats.pearsonr(*mm[["log_hits", metric]].dropna().to_numpy().T)
            # Within-category Spearman, then averaged (controls for category composition).
            within = [stats.spearmanr(g.log_hits, g[metric], nan_policy="omit")[0] for _, g in mm.groupby("categoria")]
            rows.append(dict(model=model, metric=mname, n=len(mm), spearman=rho, p=p, pearson_log=r_log,
                             within_category_spearman=np.nanmean(within)))
    corr = pd.DataFrame(rows)
    save(corr, "E6_popularity_spearman")
    bycat = []
    for (model, cat), g in m.groupby(["model", "categoria"]):
        rho, p = stats.spearmanr(g.log_hits, g.score_gpt, nan_policy="omit")
        bycat.append(dict(model=model, category=cat, n=len(g), spearman_judge=rho, p=p))
    save(pd.DataFrame(bycat), "E6_popularity_by_category")
    m["pop_quintile"] = pd.qcut(m.log_hits.rank(method="first"), 5, labels=["Q1 (least)", "Q2", "Q3", "Q4", "Q5 (most)"])
    q = (m.groupby(["model", "pop_quintile"], observed=True).score_gpt.agg(["mean", "sem", "size"]).reset_index())
    q["mean"] *= 100
    q["sem"] *= 100
    save(q, "E6_popularity_quintiles")
    plot_popularity(q)
    return corr, m


def plot_popularity(q):
    fig, ax = plt.subplots(figsize=(3.1, 2.2))
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
    markers = ["o", "s", "^", "D"]
    for (model, g), c, mk in zip(q.groupby("model"), colors, markers):
        x = np.arange(len(g))
        ax.errorbar(x, g["mean"], yerr=1.96 * g["sem"], color=c, marker=mk, ms=4.5, mec="white", mew=0.6,
                    lw=1.5, capsize=0, label=model)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["Q1\nleast", "Q2", "Q3", "Q4", "Q5\nmost"])
    ax.set_xlabel("Web-popularity quintile (LATAM entities)")
    ax.set_ylabel("LLM-judge score")
    ax.legend(frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0), handlelength=1.2,
              columnspacing=0.8, handletextpad=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "popularity_quintiles.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# E8. Lexical-metric sensitivity: LATAM gaps under the un-normalised token F1 and
# under the lexical score shipped with the original score files.
# ---------------------------------------------------------------------------

def lexical_sensitivity(d):
    rows = []
    for variant in ["score_f1", "score_f1_raw", LEGACY_LEXICAL]:
        for model, dm in d.groupby("model"):
            means = {r: 100 * dm[dm.region == r][variant].mean() for r in REGION_ORDER}
            for ref in ["Europe", "USA"]:
                pt, lo, hi, p, _, _ = gap_test(dm[dm.region == "LATAM"], dm[dm.region == ref], variant)
                rows.append(dict(variant=variant, model=model, comparison=f"LATAM-{ref}", gap=100 * pt,
                                 ci_lo=100 * lo, ci_hi=100 * hi, p=p, **{f"mean {r}": v for r, v in means.items()}))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p)
    save(t, "E8_lexical_variants")
    return t


# ---------------------------------------------------------------------------
# E9. Judge behaviour: score distribution and agreement with other metrics by region.
# ---------------------------------------------------------------------------

def judge_profile(d):
    rows = []
    for (model, r), g in d.groupby(["model", "region"]):
        s = g.score_gpt.dropna()
        rows.append(dict(model=model, region=r, share_zero=(s == 0).mean(), share_one=(s == 1).mean(),
                         share_graded=((s > 0) & (s < 1)).mean(), n_unique=s.nunique(),
                         spearman_judge_embedding=stats.spearmanr(g.score_gpt, g.score_embedding, nan_policy="omit")[0],
                         spearman_judge_lexical=stats.spearmanr(g.score_gpt, g.score_f1, nan_policy="omit")[0]))
    t = pd.DataFrame(rows)
    save(t, "E9_judge_profile")
    return t


# ---------------------------------------------------------------------------
# E11. Judge consistency: does the GPT-5-mini judge score an answer differently by
# region or by evaluated model once lexical and embedding similarity to the
# reference (plus category and difficulty) are held fixed?
# ---------------------------------------------------------------------------

def judge_conditional(d):
    x = d.dropna(subset=list(METRICS)).copy()
    x["f1_bin"] = pd.qcut(x.score_f1.rank(method="first"), 10, labels=False)
    x["emb_bin"] = pd.qcut(x.score_embedding.rank(method="first"), 10, labels=False)
    x["entity_code"] = pd.factorize(x.model + "|" + x.entity_id)[0]
    f = ("score_gpt ~ C(region, Treatment('LATAM')) + C(model, Treatment('DeepSeek-V3.1')) + C(f1_bin) + C(emb_bin)"
         " + C(categoria) + C(dificultad)")
    m = smf.ols(f, data=x).fit(cov_type="cluster", cov_kwds={"groups": x.entity_code})
    rows = []
    for k in m.params.index:
        if k.startswith("C(region") or k.startswith("C(model"):
            lo, hi = m.conf_int().loc[k]
            rows.append(dict(term=k.split("[T.")[1].rstrip("]"), kind="region" if "region" in k else "model",
                             coef=100 * m.params[k], ci_lo=100 * lo, ci_hi=100 * hi, p=m.pvalues[k]))
    t = pd.DataFrame(rows)
    t["r2"] = m.rsquared
    save(t, "E11_judge_conditional")
    return t


# ---------------------------------------------------------------------------
# E10. Stratified samples for the human studies that cannot be run from here.
# ---------------------------------------------------------------------------

def annotation_samples(d):
    out = RESULTS / "annotation"
    out.mkdir(parents=True, exist_ok=True)
    u = d.drop_duplicates(["region", "entity_id", "pregunta"])
    qa = (u.groupby(["region", "categoria", "dificultad"], group_keys=False)
          .apply(lambda g: g.sample(min(len(g), 5), random_state=SEED)))
    qa = qa[["region", "pais", "categoria", "dificultad", "entidad", "pregunta", "respuesta"]].sample(frac=1, random_state=SEED)
    qa.insert(0, "item_id", range(len(qa)))
    for c in ["annotator_id", "label (correct/incorrect/ambiguous)", "error_type (none/ambiguity/grammar/factual)"]:
        qa[c] = ""
    qa.to_csv(out / "qa_validation_iaa_sample.csv", index=False)

    ent = u.drop_duplicates("entity_id")
    rel = (ent.groupby(["region", "categoria"], group_keys=False)
           .apply(lambda g: g.sample(min(len(g), 15), random_state=SEED)))
    rel = rel[["region", "pais", "categoria", "entidad"]].sample(frac=1, random_state=SEED)
    rel.insert(0, "item_id", range(len(rel)))
    for c in ["annotator_id", "culturally_representative_of_country (yes/no/unsure)", "category_correct (yes/no)"]:
        rel[c] = ""
    rel.to_csv(out / "relevance_filter_sample.csv", index=False)

    j = d.dropna(subset=["score_gpt"]).copy()
    j["judge_bin"] = pd.cut(j.score_gpt, [-0.01, 0.2, 0.5, 0.8, 1.0], labels=["0-0.2", "0.2-0.5", "0.5-0.8", "0.8-1"])
    js = (j.groupby(["model", "region", "judge_bin"], observed=True, group_keys=False)
          .apply(lambda g: g.sample(min(len(g), 9), random_state=SEED)))
    js = js[["model", "region", "categoria", "entidad", "pregunta", "respuesta", "respuesta_gpt5", "score_gpt"]]
    js = js.sample(frac=1, random_state=SEED)
    js.insert(0, "item_id", range(len(js)))
    blind = js.drop(columns=["model", "score_gpt"]).rename(columns={"respuesta_gpt5": "model_answer"})
    for c in ["annotator_id", "human_score (0-1)", "equivalent (yes/no)"]:
        blind[c] = ""
    blind.to_csv(out / "judge_human_agreement_sample.csv", index=False)
    js[["item_id", "model", "score_gpt"]].to_csv(out / "judge_human_agreement_key.csv", index=False)
    return len(qa), len(rel), len(js)


def main():
    models, report = complete_models()
    save(report, "E_models_included")
    d = load(models)
    print("models:", models, "rows:", len(d))
    summary = {}
    summary["stats"] = dataset_stats(d)
    print("E0 done")
    means, tests = regional_gaps(d)
    print(tests.round(3).to_string())
    std, reg = composition_control(d)
    print(std[std.weights == "pooled"].round(3).to_string())
    print(reg.round(3).to_string())
    cat = category_gaps(d)
    for m in METRICS.values():
        plot_category_gaps(cat, m, f"category_gaps_{m.split()[0].lower()}.pdf")
    plot_parity(d)
    print("E3 done")
    difficulty(d)
    countries(d)
    corr, pop = popularity(d)
    print(corr.round(3).to_string())
    print(lexical_sensitivity(d).round(2).to_string())
    print(judge_conditional(d).round(3).to_string())
    print(judge_profile(d).round(3).to_string())
    print("annotation samples:", annotation_samples(d))


if __name__ == "__main__":
    main()
