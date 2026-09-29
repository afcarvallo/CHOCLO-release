"""Human validation of CHOCLO Q/A pairs (reviewer NCxe W4, uCVE).

Reads the annotation files in src/data/respuestas/ (one file per annotator batch; not
tracked by git because file names contain annotator names). Annotator names are replaced
by anonymous IDs in every output.

Outputs (src/results/): E15_validation_*.csv
Usage:  python src/human_validation.py
"""
import glob
import re

import numpy as np
import pandas as pd

from common import DATA, RESULTS, complete_models, load, normalize_country

SRC = DATA / "respuestas"
DIFFICULTY = {"FÁCIL": "easy", "INTERMEDIA": "medium", "DIFÍCIL": "hard"}
ERROR = {"sin error": "none", "ambiguedad": "ambiguity", "ambigüedad": "ambiguity", "gramatical": "grammar",
         "alucinacion": "hallucination", "alucinación": "hallucination"}


def read_all():
    frames = []
    for f in sorted(glob.glob(str(SRC / "*"))):
        if f.endswith(".csv"):
            for sep in [",", ";"]:
                d = pd.read_csv(f, encoding="latin-1", sep=sep)
                if d.shape[1] >= 8:
                    break
        else:
            d = pd.read_excel(f)
        frames.append(d.loc[:, [c for c in d.columns if not str(c).startswith("Unnamed")]])
    a = pd.concat(frames, ignore_index=True)
    a["annotator_raw"] = a.revisor.astype(str).str.strip()
    # Batches of the same annotator are suffixed v2, v3, ...; they are the same person.
    person = a.annotator_raw.str.replace(r"v\d+$", "", regex=True)
    ids = {p: f"A{i + 1:02d}" for i, p in enumerate(sorted(person.unique()))}
    a["annotator"] = person.map(ids)
    a["batch"] = a.annotator_raw.map({r: i for i, r in enumerate(sorted(a.annotator_raw.unique()))})
    a["valid"] = a.validacion.astype(str).str.strip().str.lower().eq("si")
    a["error"] = a.error.astype(str).str.strip().str.lower().map(ERROR).fillna("unspecified")
    a["pais"] = a.pais.map(normalize_country)
    a["dificultad"] = a.dificultad.map(DIFFICULTY).fillna(a.dificultad)
    a["item"] = a.entidad.astype(str).str.strip().str.lower() + "||" + a.pregunta.astype(str).str.strip()
    # A few items were re-annotated by the same person in a later batch; keep the latest.
    a = a.sort_values("batch").drop_duplicates(["item", "annotator"], keep="last")
    return a.drop(columns=["revisor", "annotator_raw", "Comentarios", "validacion"], errors="ignore")


def krippendorff_nominal(units):
    """Krippendorff's alpha for nominal data. units: list of lists of labels (>=2 per unit)."""
    labels = sorted({v for u in units for v in u})
    idx = {v: i for i, v in enumerate(labels)}
    o = np.zeros((len(labels), len(labels)))
    for u in units:
        m = len(u)
        if m < 2:
            continue
        c = np.bincount([idx[v] for v in u], minlength=len(labels))
        o += (np.outer(c, c) - np.diag(c)) / (m - 1)
    n_c = o.sum(1)
    n = n_c.sum()
    d_o = n - np.trace(o)
    d_e = (n * n - (n_c ** 2).sum()) / (n - 1)
    return 1 - d_o / d_e


def gwet_ac1(units):
    """Gwet's AC1 for nominal data with a variable number of raters per unit."""
    labels = sorted({v for u in units for v in u})
    q = len(labels)
    pa, pk = [], np.zeros(q)
    for u in units:
        r = len(u)
        c = np.array([u.count(v) for v in labels])
        pa.append((c * (c - 1)).sum() / (r * (r - 1)))
        pk += c / r
    pa = np.mean(pa)
    pk /= len(units)
    pe = (pk * (1 - pk)).sum() / (q - 1)
    return (pa - pe) / (1 - pe)


def agreement(a, col):
    units = [g[col].tolist() for _, g in a.groupby("item") if g.annotator.nunique() > 1]
    pairwise = np.mean([np.mean([x == y for i, x in enumerate(u) for y in u[i + 1:]]) for u in units])
    return dict(label=col, items=len(units), judgments=sum(len(u) for u in units), percent_agreement=pairwise,
                krippendorff_alpha=krippendorff_nominal(units), gwet_ac1=gwet_ac1(units))


def model_scores_by_validity(a):
    """Do models score lower on items that annotators flagged? (item-quality effect on LATAM)."""
    models, _ = complete_models(verbose=False)
    d = load(models)
    d = d[d.region == "LATAM"].copy()
    d["item"] = d.entidad.str.strip().str.lower() + "||" + d.pregunta.str.strip()
    flag = a.groupby("item").valid.mean().rename("share_valid")
    m = d.merge(flag, on="item")
    m["flagged"] = m.share_valid < 1
    rows = []
    for model, g in m.groupby("model"):
        rows.append(dict(model=model, items=g.item.nunique(), judge_valid=100 * g[~g.flagged].score_gpt.mean(),
                         judge_flagged=100 * g[g.flagged].score_gpt.mean(), share_flagged=g.flagged.mean(),
                         latam_all=100 * d[d.model == model].score_gpt.mean()))
    t = pd.DataFrame(rows)
    # Upper bound on how much flagged items lower the LATAM mean: remove them entirely.
    t["latam_effect_if_flagged_removed"] = t.share_flagged * (t.judge_valid - t.judge_flagged)
    return t


def main():
    a = read_all()
    summary = dict(judgments=len(a), items=a.item.nunique(), annotators=a.annotator.nunique(),
                   items_multi=int((a.groupby("item").annotator.nunique() > 1).sum()),
                   validation_rate=a.valid.mean(), countries=a.pais.nunique(),
                   regions="LATAM only" if a.pais.isin(["Argentina", "Chile", "México"]).any() else "?")
    pd.Series(summary).to_csv(RESULTS / "E15_validation_summary.csv")
    for col, name in [("dificultad", "difficulty"), ("categoria", "category"), ("pais", "country")]:
        t = a.groupby(col).agg(judgments=("valid", "size"), items=("item", "nunique"), validation_rate=("valid", "mean"),
                               error_free=("error", lambda e: (e == "none").mean())).reset_index()
        t.to_csv(RESULTS / f"E15_validation_by_{name}.csv", index=False)
    err = a.error.value_counts().rename_axis("error").reset_index(name="judgments")
    err["share"] = err.judgments / len(a)
    err.to_csv(RESULTS / "E15_validation_errors.csv", index=False)
    pd.crosstab(a.valid, a.error).to_csv(RESULTS / "E15_validation_valid_by_error.csv")
    iaa = pd.DataFrame([agreement(a, "valid"), agreement(a, "error")])
    iaa.to_csv(RESULTS / "E15_validation_iaa.csv", index=False)
    per_ann = a.groupby("annotator").agg(judgments=("valid", "size"), validation_rate=("valid", "mean")).reset_index()
    per_ann.to_csv(RESULTS / "E15_validation_by_annotator.csv", index=False)
    q = model_scores_by_validity(a)
    q.to_csv(RESULTS / "E15_validation_model_scores.csv", index=False)
    print(pd.Series(summary).to_string())
    print(err.round(4).to_string(index=False))
    print(iaa.round(3).to_string(index=False))
    for n in ["difficulty", "category"]:
        print(pd.read_csv(RESULTS / f"E15_validation_by_{n}.csv").round(4).to_string(index=False))
    print(q.round(2).to_string(index=False))
    print(per_ann.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
