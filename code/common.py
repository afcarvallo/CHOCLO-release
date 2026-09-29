"""Shared loading and normalization for CHOCLO score files.

Score files live in src/data/ as Score_<model>_<region>.xlsx with one row per
question. A model is kept for analysis only if it has all three regions and
every region covers the full benchmark question set (see `complete_models`).
"""
import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def _load_dotenv(path=ROOT.parent / ".env"):
    """Read KEY=VALUE lines from the repo's .env (git-ignored) without overriding the environment."""
    if path.exists():
        for line in path.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and not line.lstrip().startswith("#"):
                os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()
DATA = ROOT / "data"
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
FIGURES = ROOT.parent / "figures"
FIGURES.mkdir(exist_ok=True)

REGIONS = {"latam": "LATAM", "europa": "Europe", "eeuu": "USA"}
METRICS = {"score_f1": "Lexical", "score_embedding": "Embedding", "score_gpt": "LLM judge"}
# Lexical overlap score shipped with the original score files; kept only because the
# embedding-based predictor of the submission was trained on it (see experiments.probe_controls).
LEGACY_LEXICAL = "score_lexico"
RAW_METRICS = [LEGACY_LEXICAL, "score_embedding", "score_gpt"]
DIFFICULTY = {"FÁCIL": "easy", "INTERMEDIA": "medium", "DIFÍCIL": "hard"}
MODEL_NAMES = {
    "Deepseek_v31": "DeepSeek-V3.1",
    "GPT4o_mini": "GPT-4o-mini",
    "qwen25_7b": "Qwen2.5-7B",
    "Llama38b": "Llama-3-8B",
    "Gemma_4b": "Gemma-3-4B",
    "GPT35turbo": "GPT-3.5-Turbo",
    "GPT5mini": "GPT-5-mini",
}
# A region file must contain at least this share of the benchmark questions.
MIN_COVERAGE = 0.99

COUNTRY_FIX = {
    "mexico": "México", "méxico": "México", "peru": "Perú", "perú": "Perú",
    "panama": "Panamá", "panamá": "Panamá", "republica dominicana": "República Dominicana",
    # Europe files use some capital cities as the country field.
    "berlín": "Alemania", "londres": "Reino Unido", "parís": "Francia", "roma": "Italia",
}


def normalize_country(s):
    s = str(s).strip()
    key = s.lower()
    if key in COUNTRY_FIX:
        return COUNTRY_FIX[key]
    return " ".join(w if w in {"de", "del", "y"} else w.capitalize() for w in key.split())


def _read(path):
    CACHE.mkdir(exist_ok=True)
    pq = CACHE / (path.name.split(".")[0] + ".parquet")
    if pq.exists() and pq.stat().st_mtime >= path.stat().st_mtime:
        return pd.read_parquet(pq)
    d = pd.read_excel(path, sheet_name=0) if path.suffix == ".xlsx" else pd.read_csv(path)
    keep = ["entidad", "pais", "categoria", "dificultad", "pregunta", "respuesta", "respuesta_gpt5",
            "score_lexico", "score_embedding", "score_gpt"]
    d = d[keep].copy()
    for c in ["entidad", "pais", "categoria", "dificultad", "pregunta", "respuesta", "respuesta_gpt5"]:
        d[c] = d[c].astype(str)
    for c in RAW_METRICS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d.to_parquet(pq)
    return d


def inventory():
    """Return {model: {region: path}} for all Score_<model>_<region>.xlsx / .csv files."""
    inv = {}
    files = [f for pat in ("Score_*.xlsx", "Score_*.csv", "Score_*.csv.gz") for f in DATA.glob(pat)]
    for p in sorted(files):
        m = re.match(r"Score_(.+)_(latam|europa|eeuu)$", p.name.split(".")[0])
        if m:
            inv.setdefault(m.group(1), {})[m.group(2)] = p
    return inv


def complete_models(verbose=True):
    """Models with all three regions and full question coverage in each."""
    inv = inventory()
    # Reference question sets: the union over all files per region.
    ref = {}
    for model, files in inv.items():
        for region, p in files.items():
            ref.setdefault(region, set()).update(_read(p).pregunta)
    keep, report = [], []
    for model, files in inv.items():
        cov = {r: (len(set(_read(files[r]).pregunta) & ref[r]) / len(ref[r]) if r in files else 0.0)
               for r in REGIONS}
        ok = all(c >= MIN_COVERAGE for c in cov.values())
        report.append({"model": model, **{REGIONS[r]: round(c, 3) for r, c in cov.items()}, "included": ok})
        if ok:
            keep.append(model)
    if verbose:
        print(pd.DataFrame(report).to_string(index=False))
    return keep, pd.DataFrame(report)


def load(models=None):
    """Long question-level frame for the given (default: complete) models."""
    if models is None:
        models, _ = complete_models(verbose=False)
    inv = inventory()
    frames = []
    for model in models:
        for region, p in inv[model].items():
            d = _read(p)
            d["model"] = MODEL_NAMES.get(model, model)
            d["region"] = REGIONS[region]
            frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["pais"] = d.pais.map(normalize_country)
    d["dificultad"] = d.dificultad.map(DIFFICULTY).fillna(d.dificultad)
    d["entity_id"] = d.region + "|" + d.categoria + "|" + d.entidad
    f1 = lexical_scores(d[["respuesta", "respuesta_gpt5"]])
    d = d.merge(f1, on=["respuesta", "respuesta_gpt5"], how="left")
    return d


def lexical_scores(pairs):
    """Token F1 per (reference, answer) pair, cached on disk.

    score_f1: normalised (lower-case, no punctuation, no diacritics, no Spanish stop words).
    score_f1_raw: lower-case and whitespace tokenisation only.
    """
    path = CACHE / "lexical_f1.parquet"
    cached = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["respuesta", "respuesta_gpt5"])
    pairs = pairs.drop_duplicates()
    todo = pairs.merge(cached[["respuesta", "respuesta_gpt5"]], how="left", indicator=True)
    todo = todo[todo._merge == "left_only"].drop(columns="_merge")
    if len(todo):
        todo["score_f1"] = [token_f1(a, b) for a, b in zip(todo.respuesta, todo.respuesta_gpt5)]
        todo["score_f1_raw"] = [token_f1(a, b, normalize=False) for a, b in zip(todo.respuesta, todo.respuesta_gpt5)]
        cached = pd.concat([cached, todo], ignore_index=True)
        CACHE.mkdir(exist_ok=True)
        cached.to_parquet(path)
    return cached


def entity_scores(d):
    """Entity-level K_m(e): mean of each metric over the entity's questions."""
    cols = list(dict.fromkeys(list(METRICS) + RAW_METRICS))
    g = d.groupby(["model", "region", "categoria", "pais", "entity_id"], as_index=False)[cols].mean()
    return g


_PUNCT = re.compile(r"[^\w\s]")


def normalize_text(s):
    s = _PUNCT.sub(" ", str(s).lower())
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn").split()


SPANISH_STOP = set("""el la los las un una unos unas de del al a en y o que es son por con para su sus se lo
como mas más fue era ha han sin sobre entre este esta estos estas ese esa""".split())


def token_f1(ref, pred, normalize=True):
    """SQuAD-style token F1 between reference and predicted answer."""
    from collections import Counter
    if normalize:
        a, b = normalize_text(ref), normalize_text(pred)
    else:
        a, b = str(ref).lower().split(), str(pred).lower().split()
    if normalize:
        a = [t for t in a if t not in SPANISH_STOP] or a
        b = [t for t in b if t not in SPANISH_STOP] or b
    common = sum((Counter(a) & Counter(b)).values())
    if common == 0 or not a or not b:
        return 0.0
    p, r = common / len(b), common / len(a)
    return 2 * p * r / (p + r)


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
