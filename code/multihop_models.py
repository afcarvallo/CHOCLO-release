"""Question structure by model (reviewer: multi-hop breakdown in the main text).

Uses the triplets of the stratified sample (triplets_from_qa.py, E20). Each question is
  single : one relation;
  multi  : two or more relations about the entity itself (compositional);
  bridge : a triplet whose subject is the object of another triplet (entity -> x -> y, i.e., a 2-hop chain).
Outputs: src/results/E21_multihop_models.csv, E21_question_types.csv
Usage:   python src/multihop_models.py
"""
import json

import numpy as np
import pandas as pd

from common import CACHE, MODEL_NAMES, RESULTS, complete_models, load, normalize_text

B, SEED = 2000, 13
TYPES = ["single", "multi", "bridge"]
REGIONS = ["LATAM", "Europe", "USA"]


def norm(x):
    return " ".join(normalize_text(str(x)))


def question_types(d):
    out = CACHE / "triplets_qa"
    names = pd.read_csv(out / "sample.csv").set_index("entity_id").entidad.map(norm).to_dict()
    ref = d[d.model == d.model.iloc[0]].drop_duplicates(["entity_id", "pregunta"])
    ref = ref[ref.entity_id.isin(names)]
    qs = {e: g.pregunta.tolist() for e, g in ref.groupby("entity_id")}
    rows = []
    for line in (out / "triplets_gpt-4o-2024-08-06.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        ent = names[r["entity_id"]]
        for p in r["pairs"]:
            try:
                preg = qs[r["entity_id"]][int(p.get("id"))]
            except (TypeError, ValueError, IndexError, KeyError):
                continue
            t = [x for x in p.get("triplets", []) if isinstance(x, dict) and x.get("relation") and x.get("object")]
            if not t:
                continue
            objs = {norm(x["object"]) for x in t}
            bridge = any(s and s != ent and ent not in s and s in objs for s in (norm(x.get("subject")) for x in t))
            n_rel = len({x["relation"] for x in t})
            rows.append(dict(entity_id=r["entity_id"], pregunta=preg, n_relations=n_rel,
                             type="bridge" if bridge else ("multi" if n_rel >= 2 else "single")))
    return pd.DataFrame(rows).drop_duplicates(["entity_id", "pregunta"])


def main():
    models, _ = complete_models(verbose=False)
    d = load(models)
    qt = question_types(d)
    qt.to_csv(RESULTS / "E21_question_types.csv", index=False)
    x = d.merge(qt, on=["entity_id", "pregunta"]).assign(s=lambda t: 100 * t.score_gpt)
    rng = np.random.default_rng(SEED)
    rows = []
    for m, g in x.groupby("model"):
        cell = g.groupby(["entity_id", "region", "type"]).s.agg(["sum", "size"]).reset_index()
        ents = cell.entity_id.unique()
        cell["i"] = cell.entity_id.map({e: i for i, e in enumerate(ents)})

        def stat(w):
            c = cell.assign(S=cell["sum"] * w[cell.i.values], N=cell["size"] * w[cell.i.values])
            mean = c.groupby(["type", "region"])[["S", "N"]].sum().pipe(lambda t: t.S / t.N).unstack()
            out = {}
            for ty in TYPES:
                for r in REGIONS:
                    out[f"{ty}_{r}"] = mean.loc[ty, r]
                out[f"{ty}_gap_eur"] = mean.loc[ty, "LATAM"] - mean.loc[ty, "Europe"]
                out[f"{ty}_gap_usa"] = mean.loc[ty, "LATAM"] - mean.loc[ty, "USA"]
            return out

        est = stat(np.ones(len(ents)))
        boots = pd.DataFrame([stat(rng.multinomial(len(ents), np.full(len(ents), 1 / len(ents))).astype(float))
                              for _ in range(B)])
        row = dict(model=MODEL_NAMES.get(m, m), **{f"n_{ty}": int((g.type == ty).sum()) for ty in TYPES}, **est)
        for k in [f"{ty}_gap_{r}" for ty in TYPES for r in ["eur", "usa"]]:
            row[f"{k}_lo"], row[f"{k}_hi"] = boots[k].quantile([0.025, 0.975])
            row[f"{k}_p"] = min(1.0, 2 * min((boots[k] <= 0).mean(), (boots[k] >= 0).mean()))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "E21_multihop_models.csv", index=False)
    print(qt.merge(d.drop_duplicates(["entity_id", "pregunta"])[["entity_id", "pregunta", "region"]])
          .groupby(["type", "region"]).size().unstack())
    print(out.set_index("model").round(2).T.to_string())


if __name__ == "__main__":
    main()
