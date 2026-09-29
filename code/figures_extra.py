"""Additional paper figures built from src/results/ (no recomputation).

Run after experiments.py:  python src/figures_extra.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURES, RESULTS
from experiments import CATEGORY_ORDER, REGION_COLOR, REGION_MARKER, REGION_ORDER

MODELS = ["DeepSeek-V3.1", "GPT-4o-mini", "Gemma-3-4B", "Qwen2.5-7B"]
MODEL_COLOR = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]))
MODEL_MARKER = dict(zip(MODELS, ["o", "s", "^", "D"]))
COUNTRY_EN = {"México": "Mexico", "Perú": "Peru", "Panamá": "Panama", "República Dominicana": "Dominican Rep."}


def composition():
    """(a) category share per region; (b) raw vs composition-standardised LLM-judge gap."""
    share = pd.read_csv(RESULTS / "E0_category_share.csv").set_index("categoria")
    raw = pd.read_csv(RESULTS / "E1_regional_gap_tests.csv")
    std = pd.read_csv(RESULTS / "E2_standardised_gaps.csv")
    raw = raw[raw.metric == "LLM judge"].set_index(["model", "comparison"]).gap
    std = std[(std.metric == "LLM judge") & (std.weights == "pooled")].set_index(["model", "comparison"])

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.35), gridspec_kw={"width_ratios": [1, 1.15]})
    ypos = {c: i for i, c in enumerate(CATEGORY_ORDER[::-1])}
    for k, r in enumerate(REGION_ORDER):
        y = share.index.map(ypos) + (k - 1) * 0.22
        a.barh(y, 100 * share[r], height=0.2, color=REGION_COLOR[r], label=r)
    a.set_yticks(range(len(CATEGORY_ORDER)))
    a.set_yticklabels([c.replace("_", " ") for c in CATEGORY_ORDER[::-1]])
    a.grid(axis="y", visible=False)
    a.set_xlabel("Share of region's entities (%)")
    a.set_title("(a) Category composition", loc="left")
    a.legend(frameon=False, loc="upper right", handlelength=1, handletextpad=0.3, fontsize=7)

    rows = [(m, c) for m in MODELS for c in ["LATAM-Europe", "LATAM-USA"]]
    b.axvline(0, color="#5c5c58", lw=1)
    for i, (m, c) in enumerate(rows):
        y = len(rows) - 1 - i
        x0, x1 = raw[(m, c)], std.loc[(m, c), "gap"]
        b.annotate("", xy=(x1, y), xytext=(x0, y),
                   arrowprops=dict(arrowstyle="-|>", color=MODEL_COLOR[m], lw=1.4, shrinkA=0, shrinkB=0, mutation_scale=8))
        b.scatter([x0], [y], s=22, facecolor="white", edgecolor=MODEL_COLOR[m], linewidth=1.2, zorder=3)
        b.scatter([x1], [y], s=26, color=MODEL_COLOR[m], marker=MODEL_MARKER[m], edgecolor="white", linewidth=0.6, zorder=3)
    b.set_yticks(range(len(rows)))
    b.set_yticklabels([f"{m.split('-')[0] if m != 'GPT-4o-mini' else 'GPT-4o-mini'} vs. {c.split('-')[1]}"
                       for m, c in rows][::-1])
    b.grid(axis="y", visible=False)
    b.set_xlabel("LATAM gap in LLM judge (points)")
    b.set_title("(b) Raw (○) → composition-standardized", loc="left")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(FIGURES / "composition_effect.pdf", bbox_inches="tight")
    plt.close(fig)


def countries():
    t = pd.read_csv(RESULTS / "E5_country_scores.csv")
    t = t[t.region == "LATAM"].copy()
    t["label"] = t.pais.map(lambda p: COUNTRY_EN.get(p, p)) + " (" + t.entities.map("{:,}".format) + ")"
    t = t.sort_values("DeepSeek-V3.1")
    fig, ax = plt.subplots(figsize=(3.1, 3.6))
    y = np.arange(len(t))
    ax.hlines(y, t[MODELS].min(1), t[MODELS].max(1), color="#c9c9c4", lw=1.2, zorder=1)
    for m in MODELS:
        ax.scatter(t[m], y, s=20, color=MODEL_COLOR[m], marker=MODEL_MARKER[m], edgecolor="white", linewidth=0.6,
                   zorder=3, label=m)
    ax.set_yticks(y)
    ax.set_yticklabels(t.label, fontsize=7)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean entity-level LLM-judge score")
    ax.legend(frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.35, 1.0), handletextpad=0.2,
              columnspacing=0.6, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "country_scores.pdf", bbox_inches="tight")
    plt.close(fig)


def regional_means():
    """Reviewer NCxe W1: regional means with 95% CIs for every model and metric."""
    m = pd.read_csv(RESULTS / "E1_regional_means.csv")
    metrics = ["Lexical", "Embedding", "LLM judge"]
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 1.9), sharey=True)
    for ax, metric in zip(axes, metrics):
        t = m[m.metric == metric]
        for k, r in enumerate(REGION_ORDER):
            tr = t[t.region == r].set_index("model").reindex(MODELS)
            y = np.arange(len(MODELS))[::-1] + (1 - k) * 0.2
            ax.errorbar(tr["mean"], y, xerr=[tr["mean"] - tr.ci_lo, tr.ci_hi - tr["mean"]], fmt=REGION_MARKER[r],
                        ms=4.5, color=REGION_COLOR[r], mec="white", mew=0.6, elinewidth=1.2, capsize=0, label=r)
        ax.set_title(metric)
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels(MODELS[::-1])
        ax.grid(axis="y", visible=False)
    axes[1].set_xlabel("Mean score (95% CI)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.55, 1.1))
    fig.tight_layout(w_pad=0.8)
    fig.savefig(FIGURES / "regional_means.pdf", bbox_inches="tight")
    plt.close(fig)


def judge_calibration():
    """Reviewers NCxe W6 / QYTU: judge score as a function of lexical and embedding similarity, by region."""
    from common import complete_models, load
    models, _ = complete_models(verbose=False)
    d = load(models).dropna(subset=["score_gpt", "score_f1", "score_embedding"])
    out = []
    # Lexical F1 has many exact zeros, so it gets fixed-width bins with 0 as its own bin;
    # embedding similarity is continuous and uses deciles.
    f1_edges = [-0.001, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for sim, label in [("score_f1", "Lexical F1"), ("score_embedding", "Embedding similarity")]:
        if sim == "score_f1":
            d["bin"] = pd.cut(d[sim], f1_edges, labels=False)
        else:
            d["bin"] = pd.qcut(d[sim], 10, labels=False)
        g = d.groupby(["bin", "region"]).score_gpt.agg(["mean", "sem", "size"]).reset_index()
        g["similarity"] = label
        out.append(g)
    t = pd.concat(out)
    t[["mean", "sem"]] *= 100
    t.to_csv(RESULTS / "E11_judge_calibration.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.1), sharey=True)
    for ax, label in zip(axes, ["Lexical F1", "Embedding similarity"]):
        for r in REGION_ORDER:
            g = t[(t.similarity == label) & (t.region == r)]
            ax.errorbar(g["bin"], g["mean"], yerr=1.96 * g["sem"], color=REGION_COLOR[r], marker=REGION_MARKER[r],
                        ms=4, mec="white", mew=0.6, lw=1.5, capsize=0, label=r)
        if label == "Lexical F1":
            ax.set_xticks(range(11))
            ax.set_xticklabels(["0"] + [f".{i}" if i < 10 else "1" for i in range(1, 11)], fontsize=7)
            ax.set_xlabel("Lexical F1 (bin upper edge; all models)")
        else:
            ax.set_xticks(range(10))
            ax.set_xticklabels(range(1, 11))
            ax.set_xlabel("Embedding-similarity decile (all models)")
    axes[0].set_ylabel("LLM-judge score")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(w_pad=0.8)
    fig.savefig(FIGURES / "judge_calibration.pdf", bbox_inches="tight")
    plt.close(fig)


def category_scatter():
    """LATAM vs Europe/USA score per model and category (mean of the three metrics).

    Color = category, marker = model, fill = reference region (filled: Europe, open: USA).
    Points below the diagonal: lower score on LATAM entities.
    """
    t = pd.read_csv(RESULTS / "E3_category_parity.csv")
    cats = ["dish", "fauna", "flora", "geography", "object", "public_figure", "tradition"]
    cat_color = dict(zip(cats, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]))
    markers = dict(zip(MODELS, ["o", "s", "^", "D"]))
    lo = np.floor(min(t[["LATAM", "Europe", "USA"]].min()) / 5) * 5 - 1
    hi = np.ceil(max(t[["LATAM", "Europe", "USA"]].max()) / 5) * 5 + 1
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    ax.plot([lo, hi], [lo, hi], ls="--", color="#8a8a85", lw=1, zorder=1)
    for _, r in t.iterrows():
        c, mk = cat_color[r.categoria], markers[r.model]
        ax.scatter(r.Europe, r.LATAM, marker=mk, s=38, color=c, edgecolor="white", linewidth=0.6, zorder=3)
        ax.scatter(r.USA, r.LATAM, marker=mk, s=34, facecolor="white", edgecolor=c, linewidth=1.4, zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Europe / USA score (mean of 3 metrics)")
    ax.set_ylabel("LATAM score (mean of 3 metrics)")
    ax.text(hi - 0.5, lo + 0.8, "below diagonal:\nLATAM lower", ha="right", va="bottom", fontsize=7, color="#5c5c58")
    from matplotlib.lines import Line2D
    h_cat = [Line2D([], [], ls="", marker="o", ms=6, color=cat_color[c], label=c.replace("_", " ")) for c in cats]
    h_mod = [Line2D([], [], ls="", marker=markers[m], ms=6, color="#5c5c58", label=m) for m in MODELS]
    h_reg = [Line2D([], [], ls="", marker="o", ms=6, color="#5c5c58", label="vs. Europe (filled)"),
             Line2D([], [], ls="", marker="o", ms=6, markerfacecolor="white", markeredgecolor="#5c5c58",
                    markeredgewidth=1.4, label="vs. USA (open)")]
    kw = dict(frameon=False, fontsize=7.5, handletextpad=0.2, borderaxespad=0, loc="upper left")
    l1 = ax.legend(handles=h_cat, title="Category", title_fontsize=8, bbox_to_anchor=(1.04, 1.0), **kw)
    ax.add_artist(l1)
    l2 = ax.legend(handles=h_mod, title="Model", title_fontsize=8, bbox_to_anchor=(1.04, 0.5), **kw)
    ax.add_artist(l2)
    ax.legend(handles=h_reg, title="Reference", title_fontsize=8, bbox_to_anchor=(1.04, 0.15), **kw)
    fig.savefig(FIGURES / "category_scatter.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    composition()
    countries()
    judge_calibration()
    category_scatter()
    print("wrote composition_effect, country_scores, judge_calibration")
