"""Evaluate a recent model on a stratified CHOCLO subsample (reviewer QYTU).

The new model answers a stratified subsample; the GPT-5-mini judge then scores the
new model's answers and, with the same judge prompt, the existing answers of the
models in the paper, so that all models are compared under identical judging.

Usage:  python src/openai_eval.py [--per-cell 150] [--model gpt-5.5]
Results are cached in src/cache/openai/*.jsonl, so the script can be resumed.
"""
import argparse
import json
import re
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

from common import CACHE, RESULTS, complete_models, load

URL = "https://api.openai.com/v1/chat/completions"
JUDGE_MODEL = "gpt-5-mini"
SEED = 13

ANSWER_PROMPT = ("Responde la siguiente pregunta de forma breve y precisa, en español. "
                 "Si no sabes la respuesta, da tu mejor estimación.\nPregunta: {question}")

JUDGE_PROMPT = (
    "You are evaluating answers to factual questions. Given the question, the reference answer, and a "
    "candidate answer, rate how correct the candidate answer is with respect to the reference, on a scale "
    "from 0 to 1: 1 = fully correct and equivalent in meaning; values in between = partially correct (some "
    "required facts correct, others missing or wrong); 0 = incorrect or unrelated. Judge meaning, not "
    "wording: paraphrases, synonyms, and differences in spelling or accents are acceptable. Do not use your "
    "own knowledge to reward information that is not in the reference answer.\n"
    "Question: {question}\nReference answer: {reference}\nCandidate answer: {answer}\n"
    "Return only the score as a number between 0 and 1.")

_lock = threading.Lock()


def call(model, prompt, key, url=URL, **params):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], **params}
    for attempt in range(8):
        try:
            r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=120)
            if r.status_code == 200:
                d = r.json()
                return d["choices"][0]["message"]["content"], d["usage"]
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 2 ** attempt))
                continue
            raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
        except requests.RequestException:
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError("too many retries")


def run(jobs, path, key, workers, url=URL):
    """jobs: list of dicts with 'id', 'model', 'prompt', 'params'. Appends results to a jsonl cache."""
    done = set()
    if path.exists():
        with open(path) as f:
            done = {json.loads(line)["id"] for line in f}
    todo = [j for j in jobs if j["id"] not in done]
    print(f"{path.name}: {len(done)} cached, {len(todo)} to run")
    with ThreadPoolExecutor(workers) as ex, open(path, "a") as out:
        futs = {ex.submit(call, j["model"], j["prompt"], key, url, **j["params"]): j for j in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            j = futs[fut]
            try:
                text, usage = fut.result()
                rec = {"id": j["id"], "text": text, "usage": usage}
            except Exception as e:  # keep going; failed ids are retried on the next run
                print("error", j["id"], str(e)[:120])
                continue
            with _lock:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
            if n % 500 == 0:
                print(f"  {n}/{len(todo)}", flush=True)


def read(path):
    with open(path) as f:
        return pd.DataFrame([json.loads(line) for line in f])


def parse_score(text):
    m = re.search(r"\d+(?:[.,]\d+)?", str(text))
    return float(np.clip(float(m.group().replace(",", ".")), 0, 1)) if m else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--per-cell", type=int, default=150)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    key = os.environ["OPENAI_API_KEY"]
    out_dir = CACHE / "openai"
    out_dir.mkdir(parents=True, exist_ok=True)

    models, _ = complete_models(verbose=False)
    d = load(models)
    ref = d[d.model == d.model.iloc[0]].drop_duplicates(["region", "entity_id", "pregunta"])
    sample = (ref.groupby(["region", "categoria", "dificultad"], group_keys=False)
              .apply(lambda g: g.sample(min(len(g), args.per_cell), random_state=SEED)))
    sample = sample[["region", "pais", "categoria", "dificultad", "entidad", "entity_id", "pregunta", "respuesta"]]
    sample = sample.reset_index(drop=True)
    sample["qid"] = [f"q{i}" for i in range(len(sample))]
    sample.to_csv(RESULTS / "openai_subsample.csv", index=False)
    print("subsample:", len(sample))

    # 1) New model answers.
    ans_path = out_dir / f"answers_{args.model}.jsonl"
    run([{"id": q, "model": args.model, "prompt": ANSWER_PROMPT.format(question=p),
          "params": {"reasoning_effort": "none", "temperature": 0}}
         for q, p in zip(sample.qid, sample.pregunta)], ans_path, key, args.workers)
    new = read(ans_path).rename(columns={"id": "qid", "text": "answer"})[["qid", "answer"]]
    new["model"] = args.model

    # 2) Existing answers of the paper's models on the same questions.
    old = d.merge(sample[["qid", "region", "entity_id", "pregunta"]], on=["region", "entity_id", "pregunta"])
    old = old.drop_duplicates(["qid", "model"])[["qid", "model", "respuesta_gpt5", "score_gpt"]]
    old = old.rename(columns={"respuesta_gpt5": "answer", "score_gpt": "original_judge"})
    answers = pd.concat([new, old], ignore_index=True).merge(sample, on="qid")

    # 3) Judge every answer with the same prompt.
    judge_path = out_dir / "judge_gpt-5-mini.jsonl"
    answers["jid"] = answers.model + "|" + answers.qid
    run([{"id": j, "model": JUDGE_MODEL,
          "prompt": JUDGE_PROMPT.format(question=r.pregunta, reference=r.respuesta, answer=r.answer),
          "params": {"reasoning_effort": "minimal"}}
         for j, r in zip(answers.jid, answers.itertuples())], judge_path, key, args.workers)
    js = read(judge_path).rename(columns={"id": "jid"})
    js["judge"] = js.text.map(parse_score)
    answers = answers.merge(js[["jid", "judge"]], on="jid", how="left")
    answers.to_csv(RESULTS / "E12_openai_subsample_scores.csv", index=False)
    print(answers.groupby(["model", "region"]).judge.mean().unstack().mul(100).round(2))


if __name__ == "__main__":
    main()
