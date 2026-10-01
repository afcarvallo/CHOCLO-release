"""Triplets recovered from CHOCLO Q/A pairs (reviewer: KG statistics, relations, multi-hop).

The original triplets were not preserved. For a stratified sample of entities we ask
gpt-4o-2024-08-06 (the model used to build CHOCLO) to express the facts stated by each
question and its reference answer as subject-relation-object triplets, question by question.

Outputs: src/results/E20_triplets_*.csv  (cache: src/cache/triplets_qa/, resumable)
Usage:   python src/triplets_from_qa.py [--per-cell 100]
"""
import argparse
import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from scipy import stats

from common import CACHE, RESULTS, complete_models, load, normalize_text

OUT = CACHE / "triplets_qa"
MODEL = "gpt-4o-2024-08-06"
SEED = 13
PROMPT = ('You are an information extraction system. Below are question-answer pairs about the entity "{entity}" '
          "(country: {country}, category: {category}). For each pair, express the facts stated by the question "
          "together with its answer as subject-relation-object triplets. Use only information stated in the pair; "
          "do not add external knowledge. Use short relation names in snake_case (e.g., is_a, located_in, made_with, "
          "born_in, celebrated_in). Return a JSON object "
          '{{"pairs": [{{"id": <pair id>, "triplets": [{{"subject": ..., "relation": ..., "object": ...}}]}}]}}.\n\n{pairs}')
_lock = threading.Lock()


def sample(d, per_cell):
    path = OUT / "sample.csv"
    if path.exists():
        return pd.read_csv(path)
    ents = d.drop_duplicates("entity_id")[["entity_id", "region", "categoria", "entidad", "pais"]]
    s = (ents.groupby(["region", "categoria"], group_keys=False)
         .apply(lambda g: g.sample(min(len(g), per_cell), random_state=SEED)).reset_index(drop=True))
    s.to_csv(path, index=False)
    return s


def extract(groups, key, workers=12):
    path = OUT / f"triplets_{MODEL}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["entity_id"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [g for g in groups if g["entity_id"] not in done]
    print(f"extraction: {len(done)} cached, {len(todo)} to run", flush=True)

    def call(g):
        pairs = "\n".join(f"[{i}] Q: {q}\n    A: {a}" for i, (q, a) in enumerate(g["qa"]))
        body = {"model": MODEL, "temperature": 0, "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": PROMPT.format(entity=g["entidad"], country=g["pais"],
                                                                       category=g["categoria"], pairs=pairs)}]}
        for attempt in range(8):
            r = requests.post("https://api.openai.com/v1/chat/completions", json=body, timeout=120,
                              headers={"Authorization": f"Bearer {key}"})
            if r.status_code == 200:
                d = r.json()
                try:
                    out = json.loads(d["choices"][0]["message"]["content"]).get("pairs", [])
                except (json.JSONDecodeError, AttributeError):
                    out = []
                return dict(entity_id=g["entity_id"], pairs=out, usage=d["usage"])
            time.sleep(min(60, 2 ** attempt))
        raise RuntimeError(r.text[:200])

    with ThreadPoolExecutor(workers) as ex, open(path, "a", encoding="utf-8") as f:
        for n, fut in enumerate(as_completed([ex.submit(call, g) for g in todo]), 1):
            try:
                rec = fut.result()
            except Exception as e:
                print("error", str(e)[:120])
                continue
            with _lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n % 250 == 0:
                print(f"  {n}/{len(todo)}", flush=True)
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def norm(x):
    return " ".join(normalize_text(x))


def analyses(d, s, groups, recs):
    by_id = {g["entity_id"]: g for g in groups}
    rows = []
    for r in recs:
        g = by_id.get(r["entity_id"])
        if not g:
            continue
        for p in r["pairs"]:
            try:
                i = int(p.get("id"))
            except (TypeError, ValueError):
                continue
            if not 0 <= i < len(g["qa"]):
                continue
            trip = [t for t in p.get("triplets", []) if isinstance(t, dict) and t.get("relation") and t.get("object")]
            rows.append(dict(entity_id=r["entity_id"], pregunta=g["qa"][i][0], n_triplets=len(trip),
                             n_relations=len({t["relation"] for t in trip}), triplets=trip))
    q = pd.DataFrame(rows).merge(d.drop_duplicates(["entity_id", "pregunta"])[
        ["entity_id", "pregunta", "region", "categoria", "dificultad"]], on=["entity_id", "pregunta"])

    # Graph statistics per entity and relation inventory.
    ent = q.groupby(["entity_id", "region", "categoria"]).agg(
        questions=("pregunta", "size"), triplets=("n_triplets", "sum"),
        relations=("triplets", lambda ts: len({t["relation"] for tl in ts for t in tl}))).reset_index()
    all_trip = [(r.region, r.categoria, t) for r in q.itertuples() for t in r.triplets]
    rel = Counter(t["relation"] for _, _, t in all_trip)
    top = pd.DataFrame([(k, v, v / len(all_trip)) for k, v in rel.most_common(25)], columns=["relation", "count", "share"])
    rel_cat = pd.DataFrame([(c, ", ".join(f"{k} ({v})" for k, v in Counter(t["relation"] for rr, cc, t in all_trip if cc == c).most_common(5)))
                            for c in sorted({c for _, c, _ in all_trip})], columns=["category", "top_relations"])

    # Connectivity: triplet objects that name another CHOCLO entity.
    names = {}
    for r in d.drop_duplicates("entity_id").itertuples():
        k = norm(r.entidad)
        if len(k) >= 4:
            names.setdefault(k, set()).add(r.region)
    own = s.set_index("entity_id").entidad.map(norm).to_dict()
    q["n_links"] = [sum(1 for t in r.triplets if norm(t["object"]) in names and norm(t["object"]) != own.get(r.entity_id))
                    for r in q.itertuples()]
    ent["links"] = ent.entity_id.map(q.groupby("entity_id").n_links.sum())
    ent["has_link"] = ent.links > 0

    # Structure of questions: triplets and relations per question by difficulty; recorded counts where available.
    struct = q.groupby("dificultad")[["n_triplets", "n_relations"]].mean()
    struct["multi_relation"] = q.assign(m=q.n_relations >= 2).groupby("dificultad").m.mean()

    # Multi-relation questions and model performance, all regions (judge score, all models).
    models, _ = complete_models(verbose=False)
    sc = d[d.entity_id.isin(s.entity_id)].groupby(["entity_id", "pregunta", "region"]).score_gpt.mean().reset_index()
    q2 = q.merge(sc, on=["entity_id", "pregunta", "region"])
    q2["relations_bin"] = pd.cut(q2.n_relations, [0, 1, 2, 100], labels=["1", "2", "3+"])
    perf = q2.groupby(["relations_bin", "region"], observed=True).score_gpt.mean().unstack().mul(100)
    perf["LATAM-Europe"] = perf.LATAM - perf.Europe
    perf["LATAM-USA"] = perf.LATAM - perf.USA
    perf["questions"] = q2.groupby("relations_bin", observed=True).size()

    by_region = ent.groupby("region")[["triplets", "relations", "links", "has_link"]].mean()
    by_region["entities"] = ent.groupby("region").size()
    tests = dict(
        triplets_latam_vs_others_p=stats.mannwhitneyu(ent[ent.region == "LATAM"].triplets, ent[ent.region != "LATAM"].triplets).pvalue,
        links_latam_vs_others_p=stats.chi2_contingency(pd.crosstab(ent.region == "LATAM", ent.has_link))[1])
    summary = dict(entities=len(ent), questions=len(q), triplets=len(all_trip), distinct_relations=len(rel),
                   relations_top25_share=top["count"].sum() / len(all_trip),
                   triplets_per_question=q.n_triplets.mean(), **tests)

    by_region.reset_index().to_csv(RESULTS / "E20_triplets_by_region.csv", index=False)
    ent.groupby(["categoria", "region"])[["triplets", "has_link"]].mean().unstack().to_csv(RESULTS / "E20_triplets_by_category.csv")
    top.to_csv(RESULTS / "E20_triplets_top_relations.csv", index=False)
    rel_cat.to_csv(RESULTS / "E20_triplets_relations_by_category.csv", index=False)
    struct.reset_index().to_csv(RESULTS / "E20_triplets_question_structure.csv", index=False)
    perf.reset_index().to_csv(RESULTS / "E20_triplets_multirelation_performance.csv", index=False)
    pd.Series(summary).to_csv(RESULTS / "E20_triplets_summary.csv")
    q.drop(columns=["triplets"]).to_csv(RESULTS / "E20_triplets_questions.csv", index=False)
    print(pd.Series(summary).round(3).to_string())
    print(by_region.round(2).to_string())
    print(top.head(15).to_string(index=False))
    print(struct.round(2))
    print(perf.round(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=100)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    models, _ = complete_models(verbose=False)
    d = load(models)
    s = sample(d, args.per_cell)
    ref = d[d.model == d.model.iloc[0]].drop_duplicates(["entity_id", "pregunta"])
    ref = ref[ref.entity_id.isin(s.entity_id)]
    groups = [dict(entity_id=e, entidad=g.entidad.iloc[0], pais=g.pais.iloc[0], categoria=g.categoria.iloc[0],
                   qa=list(zip(g.pregunta, g.respuesta))) for e, g in ref.groupby("entity_id")]
    print("entities:", len(groups), "| questions:", sum(len(g["qa"]) for g in groups), flush=True)
    recs = extract(groups, os.environ["OPENAI_API_KEY"])
    analyses(d, s, groups, recs)


if __name__ == "__main__":
    main()
