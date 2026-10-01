"""Extract hidden states of an evaluated open-weight model for every CHOCLO entity (GPU machine).

Three inputs per entity:
  * contextual / name: the Spanish prompts used for the OpenAI embeddings, in the model's chat template;
  * entity: the entity name alone, as in KEEN (Gottesman & Geva, 2024), which probes the
    representation of the subject tokens.
At several relative depths we keep (a) the hidden state of the last token (for "entity", the last
entity token) and (b) the mean over non-special tokens (for "entity", the mean over entity tokens). Arrays are saved as float16 in src/gpu_probe/hidden/<model>/.
Resumable: finished (template, chunk) files are skipped.

Usage:
  python src/gpu_probe/extract_hidden.py --model google/gemma-3-4b-it --tag gemma
  python src/gpu_probe/extract_hidden.py --model Qwen/Qwen2.5-7B-Instruct --tag qwen
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
CATEGORY_ES = {"dish": "plato típico", "fauna": "fauna", "flora": "flora", "geography": "geografía",
               "object": "objeto cultural", "public_figure": "figura pública", "tradition": "tradición"}
TEMPLATES = {
    "contextual": "Entidad: {entidad}. Categoría: {categoria}. País: {pais}. "
                  "Describe todo lo que se sabe sobre esta entidad.",
    "name": "Describe todo lo que se sabe sobre {entidad}.",
    "entity": "{entidad}",
}
RAW = {"entity"}  # fed without chat template, with the tokenizer's special tokens (e.g., BOS)
DEPTHS = [0.25, 0.5, 0.75, 1.0]
CHUNK = 4096


def load_model(name):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(torch_dtype=torch.bfloat16, device_map="auto")
    try:
        model = AutoModelForCausalLM.from_pretrained(name, **kw)
    except (ValueError, KeyError):
        # Gemma-3-4B is a multimodal checkpoint in recent transformers versions.
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(name, **kw)
    model.eval()
    return tok, model


def chat(tok, text):
    """Wrap the prompt in the model's chat template, as when the model answered the benchmark."""
    msgs = [{"role": "user", "content": text}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return text


@torch.no_grad()
def encode(tok, model, texts, batch, special):
    n_layers = None
    last, mean = [], []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True, add_special_tokens=special).to(model.device)
        out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states  # tuple: embeddings + one per layer
        if n_layers is None:
            n_layers = len(hs) - 1
            idx = [max(1, round(d * n_layers)) for d in DEPTHS]
        ids = enc["input_ids"]
        keep = enc["attention_mask"].bool() & ~torch.isin(ids, torch.tensor(tok.all_special_ids, device=ids.device))
        mask = keep.unsqueeze(-1).to(hs[0].dtype)
        last.append(torch.stack([hs[j][:, -1, :] for j in idx], 1).float().cpu().numpy().astype(np.float16))
        mean.append(torch.stack([(hs[j] * mask).sum(1) / mask.sum(1) for j in idx], 1)
                    .float().cpu().numpy().astype(np.float16))
    return np.concatenate(last), np.concatenate(mean), idx, n_layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="only the first N entities (smoke test)")
    args = ap.parse_args()
    meta = pd.read_csv(HERE / "bundle" / "entities.csv")
    if args.limit:
        meta = meta.head(args.limit)
    out = HERE / "hidden" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    tok, model = load_model(args.model)
    for tname, tpl in TEMPLATES.items():
        if (out / f"{tname}.npz").exists():
            continue
        fmt = (lambda t: t) if tname in RAW else (lambda t: chat(tok, t))
        texts = [fmt(tpl.format(entidad=r.entidad, categoria=CATEGORY_ES[r.categoria], pais=r.pais))
                 for r in meta.itertuples()]
        for c in range(0, len(texts), CHUNK):
            f = out / f"{tname}_{c // CHUNK:03d}.npz"
            if f.exists():
                continue
            last, mean, idx, n_layers = encode(tok, model, texts[c:c + CHUNK], args.batch, tname in RAW)
            np.savez(f, last=last, mean=mean, layers=np.array(idx), n_layers=n_layers)
            print(f"{args.tag} {tname} {min(c + CHUNK, len(texts))}/{len(texts)}", flush=True)
    for tname in TEMPLATES:
        parts = sorted(out.glob(f"{tname}_*.npz"))
        if not parts:
            continue
        z = [np.load(p) for p in parts]
        np.savez(out / f"{tname}.npz", last=np.concatenate([a["last"] for a in z]),
                 mean=np.concatenate([a["mean"] for a in z]), layers=z[0]["layers"], n_layers=z[0]["n_layers"])
        for p in parts:
            p.unlink()
    print("done:", out)


if __name__ == "__main__":
    main()
