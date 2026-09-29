"""Entity embeddings with OpenAI text-embedding-3-large for the embedding-based predictor.

Two templates per entity (region|category|name):
  contextual: entity name, category and country
  name:       entity name only (ablation)
Embeddings are stored as float32 arrays in src/cache/embeddings/<template>.npy with the
row order given by src/cache/embeddings/entities.csv. Re-running resumes from the cache.

Usage:  python src/embeddings.py
"""
import json
import os
import time

import numpy as np
import pandas as pd
import requests

from common import CACHE, complete_models, load

URL = "https://api.openai.com/v1/embeddings"
MODEL = "text-embedding-3-large"
OUT = CACHE / "embeddings"
BATCH = 256

CATEGORY_ES = {"dish": "plato típico", "fauna": "fauna", "flora": "flora", "geography": "geografía",
               "object": "objeto cultural", "public_figure": "figura pública", "tradition": "tradición"}

TEMPLATES = {
    "contextual": "Entidad: {entidad}. Categoría: {categoria}. País: {pais}. "
                  "Describe todo lo que se sabe sobre esta entidad.",
    "name": "Describe todo lo que se sabe sobre {entidad}.",
}


def embed(texts, key):
    wait = 2
    while True:
        r = requests.post(URL, headers={"Authorization": f"Bearer {key}"},
                          json={"model": MODEL, "input": texts}, timeout=120)
        if r.status_code == 200:
            d = r.json()["data"]
            return np.array([x["embedding"] for x in sorted(d, key=lambda x: x["index"])], dtype=np.float32)
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(wait)
            wait = min(wait * 2, 60)
            continue
        raise RuntimeError(f"{r.status_code}: {r.text[:300]}")


def entities():
    path = OUT / "entities.csv"
    if path.exists():
        return pd.read_csv(path)
    models, _ = complete_models(verbose=False)
    d = load(models[:1])
    # One country label per entity (the most frequent one among its questions).
    e = (d.groupby(["entity_id", "region", "categoria", "entidad"]).pais
         .agg(lambda s: s.value_counts().index[0]).reset_index())
    OUT.mkdir(parents=True, exist_ok=True)
    e.to_csv(path, index=False)
    return e


def main():
    key = os.environ["OPENAI_API_KEY"]
    e = entities()
    for name, tpl in TEMPLATES.items():
        texts = [tpl.format(entidad=r.entidad, categoria=CATEGORY_ES[r.categoria], pais=r.pais)
                 for r in e.itertuples()]
        path, part = OUT / f"{name}.npy", OUT / f"{name}.partial.npy"
        if path.exists():
            print(name, "done")
            continue
        done = np.load(part) if part.exists() else np.zeros((0, 3072), np.float32)
        print(name, "cached", len(done), "of", len(texts), "| example:", texts[0], flush=True)
        chunks = [done]
        for i in range(len(done), len(texts), BATCH):
            chunks.append(embed(texts[i:i + BATCH], key))
            if (i // BATCH) % 20 == 0:
                np.save(part, np.concatenate(chunks))
                print(f"  {name} {i + BATCH}/{len(texts)}", flush=True)
        arr = np.concatenate(chunks)
        assert len(arr) == len(texts)
        np.save(path, arr)
        if part.exists():
            part.unlink()
        (OUT / f"{name}.json").write_text(json.dumps({"model": MODEL, "template": tpl, "n": len(arr)}))


if __name__ == "__main__":
    main()
