"""Annotation app for the human–judge agreement study (reviewer NCxe W6).

Each annotator logs in with a pseudonym, scores model answers against the reference
answer, and every answer is saved immediately to annotations/<annotator>.jsonl, so
annotators can close the browser and resume later. All annotators see the same items
(needed for inter-annotator agreement), in an order shuffled per annotator. The model
name and the judge score are never shown.

Usage (from the repo root):
    python src/annotation_app/app.py                 # 120 items, http://localhost:8000
    python src/annotation_app/app.py --n 100 --port 8000
Runs on localhost only by default (use --host 0.0.0.0 to share on a local network).
Export all answers:  http://localhost:8000/export  (CSV)
"""
import argparse
import csv
import io
import json
import random
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from flask import Flask, Response, redirect, render_template_string, request, session, url_for

HERE = Path(__file__).resolve().parent
SAMPLE = HERE.parent / "results" / "annotation" / "judge_human_agreement_sample.csv"
KEY = HERE.parent / "results" / "annotation" / "judge_human_agreement_key.csv"
OUT = HERE / "annotations"
SCORES = [("1", "Correcta", "Equivalente en significado a la referencia (aunque use otras palabras)."),
          ("0.75", "Casi correcta", "Correcta en lo esencial; falta un detalle menor o sobra algo irrelevante."),
          ("0.5", "Parcialmente correcta", "Algunos datos pedidos son correctos y otros faltan o están mal."),
          ("0.25", "Mayormente incorrecta", "Solo un elemento menor coincide con la referencia."),
          ("0", "Incorrecta", "No coincide con la referencia o no responde la pregunta."),
          ("na", "No puedo juzgar", "La pregunta o la referencia son ambiguas o erróneas.")]

app = Flask(__name__)
app.secret_key = "choclo-annotation"  # only used to keep the pseudonym in the browser session
_lock = threading.Lock()
ITEMS: list = []


def load_items(n, seed=13):
    """Balanced subset: the same n items for everyone, stratified by region x model."""
    d = pd.read_csv(SAMPLE)
    key = pd.read_csv(KEY)[["item_id", "model"]]
    d = d.merge(key, on="item_id")
    cells = d.groupby(["region", "model"])
    per_cell = max(1, n // cells.ngroups)
    sub = cells.apply(lambda g: g.sample(min(len(g), per_cell), random_state=seed)).reset_index(drop=True)
    sub = sub.sort_values("item_id")
    return [dict(item_id=int(r.item_id), region=r.region, category=str(r.categoria).replace("_", " "),
                 entity=r.entidad, question=r.pregunta, reference=r.respuesta, answer=r.model_answer)
            for r in sub.itertuples()]


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_-]", "", name)[:40]


def order_for(annotator):
    ids = [it["item_id"] for it in ITEMS]
    random.Random(annotator).shuffle(ids)
    return ids


def answers_of(annotator):
    path = OUT / f"{annotator}.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["item_id"]] = rec  # later lines overwrite earlier ones (edits)
    return done


def save_answer(annotator, rec):
    OUT.mkdir(exist_ok=True)
    with _lock, open(OUT / f"{annotator}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


BASE = """<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CHOCLO – anotación</title><style>
:root{--bg:#fcfcfb;--card:#fff;--ink:#1f1f1d;--muted:#5c5c58;--line:#e2e2de;--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--card:#232322;--ink:#f2f2ef;--muted:#b8b7ae;--line:#3a3a38;--accent:#3987e5}}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:760px;margin:0 auto;padding:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:12px 0}
.label{font-size:13px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px}
.q{font-size:18px;font-weight:600}.meta{font-size:13px;color:var(--muted)}
.opt{display:flex;gap:10px;align-items:flex-start;border:1px solid var(--line);border-radius:8px;padding:10px;margin:6px 0;cursor:pointer}
.opt:hover{border-color:var(--accent)}.opt input{margin-top:4px}.opt small{display:block;color:var(--muted)}
.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden}.bar>div{height:100%;background:var(--accent)}
button,.btn{font:inherit;padding:10px 16px;border-radius:8px;border:1px solid var(--accent);background:var(--accent);color:#fff;cursor:pointer;text-decoration:none;display:inline-block}
.btn.secondary{background:transparent;color:var(--accent)}
input[type=text],textarea{font:inherit;width:100%;box-sizing:border-box;padding:8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--ink)}
.row{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;align-items:center}
</style></head><body><main>{% block body %}{% endblock %}</main></body></html>"""

LOGIN = BASE.replace("{% block body %}{% endblock %}", """
<h1>CHOCLO · Evaluación de respuestas</h1>
<div class="card">
<p>Verás preguntas sobre entidades culturales, la <b>respuesta de referencia</b> del benchmark y la
<b>respuesta de un modelo de lenguaje</b>. Tu tarea es indicar qué tan correcta es la respuesta del modelo
<b>en comparación con la referencia</b>. No uses tu propio conocimiento para premiar información que no está en la referencia.
Diferencias de redacción, sinónimos, tildes o mayúsculas no importan.</p>
<p>Tus respuestas se guardan automáticamente; puedes cerrar y volver cuando quieras con el mismo identificador.
Usa un <b>seudónimo o iniciales</b>, no tu nombre completo.</p>
<form method="post"><div class="label">Identificador</div>
<input type="text" name="annotator" required pattern="[A-Za-z0-9_-]{2,40}" placeholder="p. ej. A1">
<p><button type="submit">Comenzar o continuar</button></p></form></div>""")

ITEM = BASE.replace("{% block body %}{% endblock %}", """
<div class="row"><div class="meta">Anotador: <b>{{ annotator }}</b> · {{ n_done }} de {{ total }} respondidas</div>
<a class="meta" href="{{ url_for('logout') }}">Salir</a></div>
<div class="bar"><div style="width: {{ 100 * n_done // total }}%"></div></div>
<div class="card"><div class="meta">Ítem {{ pos + 1 }} de {{ total }} · {{ item.category }} · {{ item.entity }}</div>
<div class="label" style="margin-top:8px">Pregunta</div><div class="q">{{ item.question }}</div></div>
<div class="card"><div class="label">Respuesta de referencia (correcta)</div><div>{{ item.reference }}</div></div>
<div class="card"><div class="label">Respuesta del modelo (la que evalúas)</div><div>{{ item.answer }}</div></div>
<form method="post" class="card">
<div class="q" style="font-size:16px;margin-bottom:8px">¿Qué tan correcta es la <u>respuesta del modelo</u>, tomando la <u>respuesta de referencia</u> como la correcta?</div>
{% for value, name, desc in scores %}
<label class="opt"><input type="radio" name="score" value="{{ value }}" required {% if prev and prev.score == value %}checked{% endif %}>
<span><b>{{ loop.index }}. {{ name }}</b><small>{{ desc }}</small></span></label>
{% endfor %}
<div class="label" style="margin-top:10px">Comentario (opcional)</div>
<textarea name="comment" rows="2">{{ prev.comment if prev else '' }}</textarea>
<input type="hidden" name="pos" value="{{ pos }}">
<div class="row" style="margin-top:12px">
{% if pos > 0 %}<a id="back" class="btn secondary" href="{{ url_for('item', pos=pos - 1) }}">← Anterior</a>{% else %}<span></span>{% endif %}
<button type="submit">Guardar y seguir →</button></div>
<p class="meta">Teclado: <b>1–6</b> guarda la opción y pasa al siguiente ítem · <b>←</b> vuelve al anterior · para comentar, haz clic en el cuadro, escribe, presiona <b>Esc</b> y luego el número.</p>
</form>
<script>
let sent = false;
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'TEXTAREA' || e.metaKey || e.ctrlKey || e.altKey || sent) {
    if (e.key === 'Escape' && e.target.tagName === 'TEXTAREA') e.target.blur();  // Esc leaves the comment box
    return;
  }
  const r = document.querySelectorAll('input[name=score]');
  const k = parseInt(e.key);
  if (k >= 1 && k <= r.length) {            // number key: choose, save and go to the next item
    e.preventDefault();
    r[k - 1].checked = true;
    sent = true;
    document.forms[0].submit();
  } else if (e.key === 'Enter' && document.querySelector('input[name=score]:checked')) {
    e.preventDefault(); sent = true; document.forms[0].submit();
  } else if (e.key === 'ArrowLeft') {
    const back = document.getElementById('back'); if (back) { e.preventDefault(); window.location = back.href; }
  }
});
window.addEventListener('load', () => { if (document.activeElement) document.activeElement.blur(); });
</script>""")

DONE = BASE.replace("{% block body %}{% endblock %}", """
<div class="card"><h2>¡Listo, {{ annotator }}!</h2><p>Respondiste los {{ total }} ítems. Tus respuestas están guardadas.</p>
<p>Si quieres revisar alguna, puedes volver atrás.</p>
<a class="btn secondary" href="{{ url_for('item', pos=total - 1) }}">← Revisar respuestas</a></div>""")


@app.after_request
def no_cache(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        name = safe_name(request.form.get("annotator", ""))
        if len(name) >= 2:
            session["annotator"] = name
            done = answers_of(name)
            order = order_for(name)
            first_open = next((i for i, iid in enumerate(order) if iid not in done), None)
            return redirect(url_for("done") if first_open is None else url_for("item", pos=first_open))
    return render_template_string(LOGIN)


@app.route("/item/<int:pos>", methods=["GET", "POST"])
def item(pos):
    annotator = session.get("annotator")
    if not annotator:
        return redirect(url_for("login"))
    order = order_for(annotator)
    pos = max(0, min(pos, len(order) - 1))
    by_id = {it["item_id"]: it for it in ITEMS}
    it = by_id[order[pos]]
    if request.method == "POST":
        save_answer(annotator, dict(item_id=it["item_id"], annotator=annotator, score=request.form["score"],
                                    comment=request.form.get("comment", "").strip(),
                                    timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds")))
        done = answers_of(annotator)
        nxt = next((i for i in range(pos + 1, len(order)) if order[i] not in done), None)
        if nxt is None:
            nxt = next((i for i, iid in enumerate(order) if iid not in done), None)
        return redirect(url_for("done") if nxt is None else url_for("item", pos=nxt))
    done = answers_of(annotator)
    return render_template_string(ITEM, annotator=annotator, item=it, pos=pos, total=len(order),
                                  n_done=len([i for i in order if i in done]), prev=done.get(it["item_id"]),
                                  scores=SCORES)


@app.route("/done")
def done():
    annotator = session.get("annotator")
    if not annotator:
        return redirect(url_for("login"))
    return render_template_string(DONE, annotator=annotator, total=len(ITEMS))


@app.route("/logout")
def logout():
    session.pop("annotator", None)
    return redirect(url_for("login"))


@app.route("/export")
def export():
    """All annotators' latest answers as one CSV (item_id, annotator, score, comment, timestamp)."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["item_id", "annotator", "score", "comment", "timestamp"])
    w.writeheader()
    for path in sorted(OUT.glob("*.jsonl")):
        for rec in answers_of(path.stem).values():
            w.writerow({k: rec.get(k, "") for k in w.fieldnames})
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=human_judge_annotations.csv"})


@app.route("/status")
def status():
    rows = [(p.stem, len(answers_of(p.stem))) for p in sorted(OUT.glob("*.jsonl"))]
    return {"items": len(ITEMS), "annotators": {a: n for a, n in rows}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120, help="number of items (balanced by region x model)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to let others on the same network connect")
    args = ap.parse_args()
    ITEMS.extend(load_items(args.n))
    OUT.mkdir(exist_ok=True)
    (OUT / "items.json").write_text(json.dumps(ITEMS, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(ITEMS)} items. Open http://localhost:{args.port}  (export: /export, progress: /status)")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
