"""Wikipedia page views per CHOCLO entity (reviewer NCxe: Mallen et al. popularity measure).

Entity names are resolved to Spanish Wikipedia titles with the MediaWiki API
(exact title with case variants, following redirects; names without an exact
match are left unresolved). Page views over the last 60 days are then summed from the
MediaWiki PageViewInfo API (50 titles per request; the per-article REST endpoint is
rate-limited for this volume). Results are cached in src/cache/pageviews/.

Usage:  python src/pageviews.py
"""
import json
import re
import threading
import time

import pandas as pd
import requests

from common import CACHE, RESULTS, complete_models, load

API = "https://es.wikipedia.org/w/api.php"
PV = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/es.wikipedia/all-access/user/"
      "{title}/monthly/2024010100/2024123100")
HEADERS = {"User-Agent": "CHOCLO-benchmark-research/1.0 (anonymous submission; contact via the paper)"}
OUT = CACHE / "pageviews_v2"
_session = threading.local()


def session():
    if not hasattr(_session, "s"):
        _session.s = requests.Session()
        _session.s.headers.update(HEADERS)
    return _session.s


def get(url, method="GET", **params):
    """Request with polite retrying: honours Retry-After and never gives up on 429/5xx."""
    wait = 2
    while True:
        try:
            if method == "POST":
                r = session().post(url, data=params, timeout=60)
            else:
                r = session().get(url, params=params or None, timeout=60)
            if r.status_code == 200:
                d = r.json()
                if isinstance(d, dict) and d.get("error", {}).get("code") == "maxlag":
                    time.sleep(float(r.headers.get("Retry-After", 5)))
                    continue
                return d
            if r.status_code == 404:
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(float(r.headers.get("Retry-After", wait)))
                wait = min(wait * 2, 120)
                continue
            return None
        except requests.RequestException:
            time.sleep(wait)
            wait = min(wait * 2, 120)


def variants(name):
    """Case variants of an entity name; the parenthetical qualifier stays lower-case."""
    name = name.strip()
    m = re.match(r"^(.*?)(\s*\(.*\))?$", name)
    base, qual = m.group(1), m.group(2) or ""
    cands = [name, base[:1].upper() + base[1:] + qual,
             " ".join(w if w.lower() in {"de", "del", "la", "las", "los", "el", "y", "en"} else w[:1].upper() + w[1:]
                      for w in base.split()) + qual]
    return list(dict.fromkeys(c for c in cands if c))


def resolve_exact(names):
    """Map each candidate title to its canonical existing article title (or None)."""
    res = {}
    for i in range(0, len(names), 50):
        batch = names[i:i + 50]
        d = get(API, method="POST", action="query", titles="|".join(batch), redirects=1, prop="pageprops",
                ppprop="disambiguation", format="json", formatversion=2, maxlag=5)
        time.sleep(0.2)
        if not d:
            continue
        q = d.get("query", {})
        norm = {n["from"]: n["to"] for n in q.get("normalized", [])}
        redir = {r["from"]: r["to"] for r in q.get("redirects", [])}
        pages = {p["title"]: p for p in q.get("pages", [])}
        for t in batch:
            cur = redir.get(norm.get(t, t), norm.get(t, t))
            p = pages.get(cur)
            ok = p is not None and not p.get("missing") and not p.get("invalid")
            res[t] = cur if ok and "disambiguation" not in p.get("pageprops", {}) else None
    return res


def resolve_search(name):
    d = get(API, action="query", list="search", srsearch=name, srlimit=1, format="json", formatversion=2, maxlag=5)
    time.sleep(0.2)
    hits = (d or {}).get("query", {}).get("search", [])
    return hits[0]["title"] if hits else None


def views_batch(titles):
    """Sum of daily page views over the last 60 days for up to 50 titles (None if unavailable)."""
    out = {}
    params = dict(action="query", prop="pageviews", titles="|".join(titles), pvipdays=60, format="json",
                  formatversion=2, maxlag=5)
    while True:
        d = get(API, method="POST", **params) or {}
        for p in d.get("query", {}).get("pages", []):
            pv = p.get("pageviews")
            if pv is not None:
                vals = [v for v in pv.values() if v is not None]
                out[p["title"]] = out.get(p["title"], 0) + sum(vals) if vals else out.get(p["title"])
        if "continue" not in d:
            break
        params.update(d["continue"])
    time.sleep(1.0)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    models, _ = complete_models(verbose=False)
    d = load(models[:1])
    ents = d.drop_duplicates("entity_id")[["entity_id", "region", "categoria", "entidad"]].reset_index(drop=True)
    names = sorted(ents.entidad.unique())

    tpath = OUT / "titles.json"
    titles = json.loads(tpath.read_text()) if tpath.exists() else {}
    todo = [n for n in names if n not in titles]
    print("names:", len(names), "to resolve:", len(todo))
    cand = {n: variants(n) for n in todo}
    flat = sorted({c for cs in cand.values() for c in cs})
    epath = OUT / "exact.json"
    exact = json.loads(epath.read_text()) if epath.exists() else {}
    flat = [c for c in flat if c not in exact]
    print("candidate titles to check:", len(flat), flush=True)
    for i in range(0, len(flat), 500):
        exact.update(resolve_exact(flat[i:i + 500]))
        epath.write_text(json.dumps(exact, ensure_ascii=False))
        if (i // 500) % 20 == 0:
            print(f"  exact {i}/{len(flat)}", flush=True)
    for n in todo:
        hit = next((exact.get(c) for c in cand[n] if exact.get(c)), None)
        titles[n] = {"title": hit, "method": "exact" if hit else None}
    for n in todo:
        if not titles[n]["title"]:
            titles[n]["method"] = "none"
    # Only exact title matches (after case normalisation and redirects) are used, so no
    # full-text search fallback is attempted.
    print("exact matches:", sum(bool(titles[n]["title"]) for n in todo), "of", len(todo), flush=True)
    tpath.write_text(json.dumps(titles, ensure_ascii=False))

    vpath = OUT / "views_60d.json"
    vcache = json.loads(vpath.read_text()) if vpath.exists() else {}
    tl = sorted({v["title"] for v in titles.values() if v["title"]} - set(vcache))
    print("titles to fetch views:", len(tl), flush=True)
    for i in range(0, len(tl), 50):
        batch = tl[i:i + 50]
        got = views_batch(batch)
        for t in batch:
            vcache[t] = got.get(t)
        if (i // 50) % 40 == 0:
            vpath.write_text(json.dumps(vcache, ensure_ascii=False))
            print(f"  views {i + 50}/{len(tl)}", flush=True)
    vpath.write_text(json.dumps(vcache, ensure_ascii=False))

    ents["wiki_title"] = ents.entidad.map(lambda n: titles[n]["title"])
    ents["match"] = ents.entidad.map(lambda n: titles[n]["method"])
    ents["pageviews_60d"] = ents.wiki_title.map(lambda t: vcache.get(t) if t else None)
    ents.to_csv(RESULTS / "E13_pageviews.csv", index=False)
    print(ents.groupby("region").match.value_counts().unstack())


if __name__ == "__main__":
    main()
