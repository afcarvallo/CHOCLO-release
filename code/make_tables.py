"""Render the CSVs in src/results/ as LaTeX tables in paper/tables/.

Run after experiments.py:  python src/make_tables.py
"""
import pandas as pd

from common import RESULTS, ROOT

OUT = ROOT.parent / "tables"
MODELS = ["DeepSeek-V3.1", "GPT-4o-mini", "Gemma-3-4B", "Qwen2.5-7B"]
METRICS = ["Lexical", "Embedding", "LLM judge"]
REGIONS = ["LATAM", "Europe", "USA"]
CATS = ["dish", "fauna", "flora", "geography", "object", "public_figure", "tradition"]
COUNTRY_EN = {
    "México": "Mexico", "Perú": "Peru", "Panamá": "Panama", "República Dominicana": "Dominican Republic",
    "Alemania": "Germany", "Bélgica": "Belgium", "Croacia": "Croatia", "Dinamarca": "Denmark",
    "Eslovaquia": "Slovakia", "España": "Spain", "Finlandia": "Finland", "Francia": "France", "Grecia": "Greece",
    "Hungría": "Hungary", "Irlanda": "Ireland", "Islandia": "Iceland", "Italia": "Italy", "Noruega": "Norway",
    "Polonia": "Poland", "Reino Unido": "United Kingdom", "República Checa": "Czech Republic", "Rumania": "Romania",
    "Rusia": "Russia", "Suecia": "Sweden", "Suiza": "Switzerland", "Ucrania": "Ukraine",
}


def tex(s):
    return s.replace("_", r"\_")


def write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text(body)


def fmt_gap(r):
    star = "$^{*}$" if r.p_holm < 0.05 else ""
    return f"${r.gap:+.1f}$" + star


def dataset_stats():
    t = pd.read_csv(RESULTS / "E0_dataset_stats.csv", header=[0, 1], index_col=0)
    lines = []
    for cat in CATS + ["total"]:
        row = t.loc[cat]
        cells = []
        for r in REGIONS:
            e, q = int(row[("entities", r)]), int(row[("questions", r)])
            share = "" if cat == "total" else f" ({100 * e / t.loc['total'][('entities', r)]:.1f}\\%)"
            cells += [f"{e:,}{share}", f"{q:,}"]
        tot_e = sum(int(row[("entities", r)]) for r in REGIONS)
        tot_q = sum(int(row[("questions", r)]) for r in REGIONS)
        name = r"\textbf{Total}" if cat == "total" else tex(cat.replace("_", " "))
        if cat == "total":
            lines.append(r"\midrule")
        lines.append(" & ".join([name] + cells + [f"{tot_e:,}", f"{tot_q:,}"]) + r" \\")
    write("dataset_stats", "\n".join(lines) + "\n")


def regional_means():
    m = pd.read_csv(RESULTS / "E1_regional_means.csv")
    lines = []
    for model in MODELS:
        for i, r in enumerate(REGIONS):
            cells = [model if i == 0 else "", r]
            for metric in METRICS:
                x = m[(m.model == model) & (m.region == r) & (m.metric == metric)].iloc[0]
                sub = m[(m.model == model) & (m.metric == metric)]
                v = f"{x['mean']:.2f}"
                cells.append(rf"\textbf{{{v}}}" if x["mean"] == sub["mean"].min() else v)
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule" if model != MODELS[-1] else "")
    write("regional_means", "\n".join(lines) + "\n")


def gaps():
    raw = pd.read_csv(RESULTS / "E1_regional_gap_tests.csv")
    std = pd.read_csv(RESULTS / "E2_standardised_gaps.csv")
    std = std[std.weights == "pooled"]
    reg = pd.read_csv(RESULTS / "E2_regression_gaps.csv")
    lines = []
    for model in MODELS:
        for i, comp in enumerate(["LATAM-Europe", "LATAM-USA"]):
            cells = [model if i == 0 else "", comp.replace("LATAM-", r"vs.\ ")]
            for src in [raw, std, reg]:
                for metric in METRICS:
                    x = src[(src.model == model) & (src.comparison == comp) & (src.metric == metric)].iloc[0]
                    cells.append(fmt_gap(x))
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule" if model != MODELS[-1] else "")
    write("gaps", "\n".join(lines) + "\n")


def category_gaps():
    t = pd.read_csv(RESULTS / "E3_category_gaps.csv")
    t = t[t.metric == "LLM judge"]
    lines = []
    for cat in ["tradition", "public_figure", "object", "dish", "flora", "geography", "fauna"]:
        cells = [tex(cat.replace("_", " "))]
        for comp in ["LATAM-Europe", "LATAM-USA"]:
            for model in MODELS:
                x = t[(t.model == model) & (t.category == cat) & (t.comparison == comp)].iloc[0]
                cells.append(fmt_gap(x))
        lines.append(" & ".join(cells) + r" \\")
    write("category_gaps", "\n".join(lines) + "\n")


def difficulty():
    t = pd.read_csv(RESULTS / "E4_difficulty.csv")
    lines = []
    for model in MODELS:
        cells = [model]
        for col in ["score_gpt", "average"]:
            sub = t[t.model == model].set_index("dificultad")[col]
            for lvl in ["easy", "medium", "hard"]:
                v = f"{sub[lvl]:.2f}"
                cells.append(rf"\textbf{{{v}}}" if sub[lvl] == sub.min() else v)
        lines.append(" & ".join(cells) + r" \\")
    write("difficulty", "\n".join(lines) + "\n")


def popularity():
    t = pd.read_csv(RESULTS / "E6_popularity_spearman.csv")
    lines = []
    for model in MODELS:
        cells = [model]
        for metric in METRICS:
            x = t[(t.model == model) & (t.metric == metric)].iloc[0]
            cells += [f"${x.spearman:+.2f}$", f"${x.within_category_spearman:+.2f}$"]
        lines.append(" & ".join(cells) + r" \\")
    write("popularity", "\n".join(lines) + "\n")


def probe():
    t = pd.read_csv(RESULTS / "E7_probe_controls.csv").set_index("region")
    lines = []
    for r in REGIONS:
        x = t.loc[r]
        pop = "--" if pd.isna(x["popularity only (GBM)"]) else f"{x['popularity only (GBM)']:.3f}"
        popc = "--" if pd.isna(x["popularity+category (GBM)"]) else f"{x['popularity+category (GBM)']:.3f}"
        lines.append(" & ".join([r, f"{x.target_sd:.3f}", f"{x['global mean']:.3f}", f"{x['category+region mean']:.3f}",
                                 pop, popc, f"{x.mlp_paper:.3f}"]) + r" \\")
    write("probe_controls", "\n".join(lines) + "\n")


def countries():
    t = pd.read_csv(RESULTS / "E5_country_scores.csv")
    lines = []
    for region in ["LATAM", "Europe"]:
        for _, x in t[t.region == region].iterrows():
            name = COUNTRY_EN.get(x.pais, x.pais)
            lines.append(" & ".join([region, name, f"{int(x.entities):,}"] + [f"{x[m]:.1f}" for m in MODELS]) + r" \\")
        lines.append(r"\midrule")
    means = pd.read_csv(RESULTS / "E1_regional_means.csv")
    means = means[(means.region == "USA") & (means.metric == "LLM judge")].set_index("model")["mean"]
    n_usa = int(t[t.region == "USA"].entities.sum())
    lines.append(" & ".join(["USA", "all (22 labels)", f"{n_usa:,}"] + [f"{means[m]:.1f}" for m in MODELS]) + r" \\")
    write("countries", "\n".join(lines) + "\n")


def difficulty_by_region():
    t = pd.read_csv(RESULTS / "E4_difficulty_by_region.csv")
    lines = []
    for model in MODELS:
        for r in REGIONS:
            sub = t[(t.model == model) & (t.region == r)].set_index("dificultad").score_gpt
            cells = [model if r == "LATAM" else "", r]
            for lvl in ["easy", "medium", "hard"]:
                v = f"{sub[lvl]:.2f}"
                cells.append(rf"\textbf{{{v}}}" if sub[lvl] == sub.min() else v)
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule" if model != MODELS[-1] else "")
    write("difficulty_by_region", "\n".join(lines) + "\n")


def judge_profile():
    t = pd.read_csv(RESULTS / "E9_judge_profile.csv")
    lines = []
    for model in MODELS:
        for r in REGIONS:
            x = t[(t.model == model) & (t.region == r)].iloc[0]
            lines.append(" & ".join([model if r == "LATAM" else "", r, f"{100 * x.share_zero:.1f}", f"{100 * x.share_graded:.1f}",
                                     f"{100 * x.share_one:.1f}", f"{x.spearman_judge_embedding:.2f}",
                                     f"{x.spearman_judge_lexical:.2f}"]) + r" \\")
        lines.append(r"\midrule" if model != MODELS[-1] else "")
    write("judge_profile", "\n".join(lines) + "\n")


def lexical_variants():
    t = pd.read_csv(RESULTS / "E8_lexical_variants.csv")
    names = {"score_f1": "Normalized F1 (main)", "score_f1_raw": "Raw token F1", "score_lexico": "Provided overlap score"}
    lines = []
    for v, vname in names.items():
        for i, model in enumerate(MODELS):
            s = t[(t.variant == v) & (t.model == model)].set_index("comparison")
            cells = [vname if i == 0 else "", model] + [f"{s.iloc[0][f'mean {r}']:.2f}" for r in REGIONS]
            for comp in ["LATAM-Europe", "LATAM-USA"]:
                x = s.loc[comp]
                cells.append(f"${x.gap:+.2f}$" + ("$^{*}$" if x.p_holm < 0.05 else ""))
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule" if v != "score_lexico" else "")
    write("lexical_variants", "\n".join(lines) + "\n")


def judge_conditional():
    t = pd.read_csv(RESULTS / "E11_judge_conditional.csv")
    lines = []
    for _, x in t.iterrows():
        label = f"{x.term} vs.\\ " + ("LATAM" if x.kind == "region" else "DeepSeek-V3.1")
        lines.append(f"{label} & ${x.coef:+.2f}$ & [${x.ci_lo:+.2f}$, ${x.ci_hi:+.2f}$] \\\\")
    write("judge_conditional", "\n".join(lines) + "\n")


def difficulty_by_region():
    t = pd.read_csv(RESULTS / "E4_difficulty_by_region.csv")
    lines = []
    for model in MODELS:
        for r in REGIONS:
            sub = t[(t.model == model) & (t.region == r)].set_index("dificultad").score_gpt
            cells = [model if r == "LATAM" else "", r]
            for lvl in ["easy", "medium", "hard"]:
                v = f"{sub[lvl]:.2f}"
                cells.append(rf"\textbf{{{v}}}" if sub[lvl] == sub.min() else v)
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule" if model != MODELS[-1] else "")
    write("difficulty_by_region", "\n".join(lines) + "\n")


def judge_profile():
    t = pd.read_csv(RESULTS / "E9_judge_profile.csv")
    lines = []
    for model in MODELS:
        for r in REGIONS:
            x = t[(t.model == model) & (t.region == r)].iloc[0]
            lines.append(" & ".join([model if r == "LATAM" else "", r, f"{100 * x.share_zero:.1f}", f"{100 * x.share_graded:.1f}",
                                     f"{100 * x.share_one:.1f}", f"{x.spearman_judge_embedding:.2f}",
                                     f"{x.spearman_judge_lexical:.2f}"]) + r" \\")
        lines.append(r"\midrule" if model != MODELS[-1] else "")
    write("judge_profile", "\n".join(lines) + "\n")


def normalised_f1():
    t = pd.read_csv(RESULTS / "E8_normalised_f1.csv")
    lines = []
    for model in MODELS:
        s = t[t.model == model].set_index("comparison")
        cells = [model] + [f"{s.loc[f'mean {r}', 'gap']:.2f}" for r in REGIONS]
        for comp in ["LATAM-Europe", "LATAM-USA"]:
            x = s.loc[comp]
            star = "$^{*}$" if x.p < 0.05 / 6 else ""
            cells.append(f"${x.gap:+.2f}$" + star + f" [${x.ci_lo:+.2f}$, ${x.ci_hi:+.2f}$]")
        lines.append(" & ".join(cells) + r" \\")
    write("normalised_f1", "\n".join(lines) + "\n")


def recent_model():
    p = RESULTS / "E12_recent_model_regional.csv"
    if not p.exists():
        return
    t = pd.read_csv(p).set_index("model")
    order = ["gpt-5.5"] + MODELS
    names = {"gpt-5.5": "GPT-5.5"}
    lines = []
    for m in order:
        x = t.loc[m]
        cells = [names.get(m, m)] + [f"{x[f'judge {r}']:.1f}" for r in REGIONS]
        for ref in ["Europe", "USA"]:
            cells.append(f"${x[f'gap {ref}']:+.1f}$" + ("$^{*}$" if x[f"p_holm {ref}"] < 0.05 else ""))
        lines.append(" & ".join(cells) + r" \\")
        if m == "gpt-5.5":
            lines.append(r"\midrule")
    write("recent_model", "\n".join(lines) + "\n")


def probe_table():
    p = RESULTS / "E14_probe.csv"
    if not p.exists():
        return
    t = pd.read_csv(p)
    t = t[t.region == "All"].set_index(["predictor", "model"])
    rows = [("category x region mean", "Category$\\times$region mean"),
            ("page views + category x region (GBM)", "Page views + cat.$\\times$reg."),
            ("ridge (contextual)", "Ridge (contextual)"),
            ("mlp (contextual)", "MLP (contextual)"),
            ("mlp (name)", "MLP (name only)"),
            ("mlp (contextual, permuted within category x region)", "MLP (permuted)")]
    lines = []
    for key, label in rows:
        if key not in t.index.get_level_values(0):
            continue
        cells = [label]
        for m in MODELS:
            x = t.loc[(key, m)]
            fmt = lambda v: f"${0.0 if abs(v) < 0.005 else v:.2f}$"
            wg = "--" if pd.isna(x.within_group_spearman) or "mean" in key else fmt(x.within_group_spearman)
            cells += [fmt(x.r2), wg]
        lines.append(" & ".join(cells) + r" \\")
    write("probe", "\n".join(lines) + "\n")


def popularity_pageviews():
    c = RESULTS / "E13_pageview_spearman.csv"
    if not c.exists():
        return
    corr = pd.read_csv(c)
    corr = corr[corr.metric == "LLM judge"].set_index(["model", "region"])
    hits = pd.read_csv(RESULTS / "E6_popularity_spearman.csv")
    hits = hits[hits.metric == "LLM judge"].set_index("model")
    reg = pd.read_csv(RESULTS / "E13_gap_controlling_popularity.csv").set_index(["model", "controls", "comparison"])
    lines = []
    for m in MODELS:
        short = {"DeepSeek-V3.1": "DeepSeek", "GPT-4o-mini": "GPT-4o-m.", "Gemma-3-4B": "Gemma-3", "Qwen2.5-7B": "Qwen2.5"}
        cells = [short[m], f"${hits.loc[m, 'within_category_spearman']:+.2f}$"]
        cells += [f"${corr.loc[(m, r), 'within_category']:+.2f}$" for r in REGIONS]
        for comp in ["LATAM-Europe", "LATAM-USA"]:
            a = reg.loc[(m, "category", comp)]
            b = reg.loc[(m, "category + popularity", comp)]
            cells.append(f"${b.gap:+.1f}$")
        lines.append(" & ".join(cells) + r" \\")
    write("popularity_pageviews", "\n".join(lines) + "\n")


def composition_weights():
    t = pd.read_csv(RESULTS / "E2_standardised_gaps.csv")
    t = t[t.metric == "LLM judge"]
    names = {"pooled": "Pooled", "uniform": "Uniform", "europe_usa": "Europe+USA"}
    lines = []
    for m in MODELS:
        cells = [m]
        for w in ["pooled", "uniform", "europe_usa"]:
            for comp in ["LATAM-Europe", "LATAM-USA"]:
                x = t[(t.model == m) & (t.weights == w) & (t.comparison == comp)].iloc[0]
                cells.append(fmt_gap(x))
        lines.append(" & ".join(cells) + r" \\")
    write("composition_weights", "\n".join(lines) + "\n")


def recent_model_category():
    t = pd.read_csv(RESULTS / "E12_recent_model_category.csv")
    order = ["gpt-5.5"] + MODELS
    short = {"gpt-5.5": "GPT-5.5", "DeepSeek-V3.1": "DS", "GPT-4o-mini": "4o-m", "Gemma-3-4B": "Gm", "Qwen2.5-7B": "Qw"}
    lines = []
    for cat in ["tradition", "public_figure", "object", "dish", "flora", "geography", "fauna"]:
        cells = [tex(cat.replace("_", " "))]
        for comp in ["LATAM-Europe", "LATAM-USA"]:
            for m in order:
                x = t[(t.model == m) & (t.category == cat) & (t.comparison == comp)].iloc[0]
                cells.append(fmt_gap(x))
        lines.append(" & ".join(cells) + r" \\")
    write("recent_model_category", "\n".join(lines) + "\n")


def rejudge_agreement():
    t = pd.read_csv(RESULTS / "E12_rejudge_agreement.csv")
    lines = [" & ".join([x.model, f"{x.n:,}", f"{x.mean_original:.1f}", f"{x.mean_rejudged:.1f}",
                         f"{x.spearman:.2f}", f"{x.pearson:.2f}"]) + r" \\" for x in t.itertuples()]
    write("rejudge_agreement", "\n".join(lines) + "\n")


def pageview_correlations():
    t = pd.read_csv(RESULTS / "E13_pageview_spearman.csv")
    lines = []
    for m in MODELS:
        for i, r in enumerate(REGIONS + ["All"]):
            cells = [m if i == 0 else "", r]
            for metric in METRICS:
                x = t[(t.model == m) & (t.region == r) & (t.metric == metric)].iloc[0]
                cells += [f"${x.spearman:+.2f}$", f"${x.within_category:+.2f}$"]
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule" if m != MODELS[-1] else "")
    write("pageview_correlations", "\n".join(lines) + "\n")


def probe_by_region():
    t = pd.read_csv(RESULTS / "E14_probe.csv")
    keys = [("category x region mean", "Cat.$\\times$reg. mean"), ("mlp (contextual)", "MLP (contextual)"),
            ("mlp (contextual, permuted within category x region)", "MLP (permuted)")]
    lines = []
    for m in MODELS:
        for i, (k, label) in enumerate(keys):
            cells = [m if i == 0 else "", label]
            for r in REGIONS:
                x = t[(t.model == m) & (t.predictor == k) & (t.region == r)].iloc[0]
                wg = "--" if "mean" in k else f"${0.0 if abs(x.within_group_spearman) < 0.005 else x.within_group_spearman:.2f}$"
                cells += [f"{x.mae:.3f}", f"${x.r2:.2f}$", wg]
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule" if m != MODELS[-1] else "")
    write("probe_by_region", "\n".join(lines) + "\n")


def probe_transfer():
    t = pd.read_csv(RESULTS / "E14_probe_transfer.csv")
    lines = []
    for src in MODELS:
        cells = [src] + [f"{t[(t.trained_on == src) & (t.evaluated_on == tgt)].within_group_spearman.iloc[0]:.2f}"
                         for tgt in MODELS]
        lines.append(" & ".join(cells) + r" \\")
    write("probe_transfer", "\n".join(lines) + "\n")


def validation_tables():
    if not (RESULTS / "E15_validation_summary.csv").exists():
        return
    d = pd.read_csv(RESULTS / "E15_validation_by_difficulty.csv").set_index("dificultad").loc[["easy", "medium", "hard"]]
    write("validation_difficulty", "\n".join(
        f"{lvl.capitalize()} & {int(x['items']):,} & {int(x.judgments):,} & {100 * x.validation_rate:.1f} & {100 * x.error_free:.1f} \\\\"
        for lvl, x in d.iterrows()) + "\n")
    c = pd.read_csv(RESULTS / "E15_validation_by_category.csv").sort_values("validation_rate", ascending=False)
    write("validation_category", "\n".join(
        f"{tex(x.categoria.replace('_', ' '))} & {int(x['items']):,} & {int(x.judgments):,} & {100 * x.validation_rate:.1f} & {100 * x.error_free:.1f} \\\\"
        for _, x in c.iterrows()) + "\n")
    e = pd.read_csv(RESULTS / "E15_validation_errors.csv")
    order = ["none", "ambiguity", "hallucination", "grammar", "unspecified"]
    names = {"none": "No error", "ambiguity": "Ambiguity", "hallucination": "Hallucination", "grammar": "Grammatical",
             "unspecified": "Not specified"}
    e = e.set_index("error").reindex(order).dropna()
    lines = [f"{names[k]} & {int(x.judgments):,} & {100 * x.share:.2f} \\\\" for k, x in e.iterrows()]
    lines += [r"\midrule", f"Total & {int(e.judgments.sum()):,} & 100.00 \\\\"]
    write("validation_errors", "\n".join(lines) + "\n")
    i = pd.read_csv(RESULTS / "E15_validation_iaa.csv").set_index("label")
    names = {"valid": "Valid / not valid", "error": "Error type (5 classes)"}
    write("validation_iaa", "\n".join(
        f"{names[k]} & {int(x['items'])} & {100 * x.percent_agreement:.1f} & {x.gwet_ac1:.2f} & {x.krippendorff_alpha:.2f} \\\\"
        for k, x in i.iterrows()) + "\n")
    m = pd.read_csv(RESULTS / "E15_validation_model_scores.csv").set_index("model").reindex(MODELS)
    write("validation_model_scores", "\n".join(
        f"{k} & {x.judge_valid:.1f} & {x.judge_flagged:.1f} & ${x.latam_effect_if_flagged_removed:+.2f}$ \\\\"
        for k, x in m.iterrows()) + "\n")
    t = pd.read_csv(RESULTS / "E15_validation_by_country.csv").sort_values("items", ascending=False)
    write("validation_country", "\n".join(
        f"{COUNTRY_EN.get(x.pais, x.pais)} & {int(x['items'])} & {100 * x.validation_rate:.1f} \\\\" for _, x in t.iterrows()) + "\n")


def human_judge():
    p = RESULTS / "E16_human_judge_agreement.csv"
    if not p.exists():
        return
    t = pd.read_csv(p)
    bias = pd.read_csv(RESULTS / "E16_judge_bias_by_region.csv").set_index("region")
    names = {"all": "All"}
    lines = []
    for _, x in t.iterrows():
        g = x.group
        label = "All" if g == "all" else g.split("=")[1]
        extra = ""
        if g.startswith("region="):
            b = bias.loc[label]
            z = lambda v: 0.0 if abs(v) < 0.005 else v
            extra = f" [${z(b.ci_lo):+.2f}$, ${z(b.ci_hi):+.2f}$]"
        lines.append(f"{label} & {int(x.n)} & {x.spearman:.2f} & {x.mae:.2f} & {x.mean_human:.2f} & {x.mean_judge:.2f} & ${x.judge_minus_human:+.2f}${extra} \\\\")
        if g == "all" or g == "region=USA":
            lines.append(r"\midrule")
    write("human_judge", "\n".join(lines) + "\n")


if __name__ == "__main__":
    for f in [dataset_stats, regional_means, gaps, category_gaps, difficulty, popularity, countries,
              judge_profile, lexical_variants, judge_conditional, difficulty_by_region, recent_model, probe_table, popularity_pageviews, composition_weights, recent_model_category,
              rejudge_agreement, pageview_correlations, probe_by_region, probe_transfer, validation_tables, human_judge]:
        f()
    print("wrote", sorted(p.name for p in OUT.glob("*.tex")))
