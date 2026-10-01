"""Probe the evaluated models' own hidden states (reviewer: "probing should use the internal states
of the model that performed the task"). Same protocol as src/probe.py (E14), so results are
directly comparable with Table 6 of the paper.

Inputs:  src/gpu_probe/bundle/ (entities, targets, page views) and src/gpu_probe/hidden/<tag>/*.npz
Outputs: src/gpu_probe/results/E24_hidden_probe_<tag>.csv and E24_hidden_probe_<tag>_transfer.csv
Usage:   python src/gpu_probe/probe_hidden.py --tag gemma --target Gemma-3-4B
         python src/gpu_probe/probe_hidden.py --tag qwen  --target Qwen2.5-7B
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
SEED = 13
REGIONS = ["LATAM", "Europe", "USA"]
RNG = np.random.default_rng(SEED)


def within_group_spearman(y, pred, groups, gm):
    ry, rp = y - gm, pred - gm
    vals = [stats.spearmanr(ry[groups == k], rp[groups == k])[0]
            for k in np.unique(groups) if (groups == k).sum() > 30]
    return np.nan if np.all(np.isnan(vals)) else np.nanmean(vals)


def evaluate(name, y, pred, meta, gm):
    rows = []
    for r in REGIONS + ["All"]:
        m = np.ones(len(y), bool) if r == "All" else (meta.region == r).to_numpy()
        yt, pt = y[m], pred[m]
        rows.append(dict(predictor=name, region=r, n=int(m.sum()), target_sd=yt.std(), mae=np.abs(yt - pt).mean(),
                         r2=1 - ((yt - pt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum(),
                         spearman=stats.spearmanr(yt, pt)[0],
                         within_group_spearman=within_group_spearman(yt, pt, meta.group.to_numpy()[m], gm[m])))
    return rows


def permute_within(F, groups):
    idx = np.arange(len(F))
    for g in np.unique(groups):
        m = np.where(groups == g)[0]
        idx[m] = RNG.permutation(m)
    return F[idx]


def fit_predict(kind, F, y, tr, te):
    # Hidden states have a few very large dimensions, so features are standardized on the training fold.
    sc = StandardScaler().fit(F[tr])
    Xtr, Xte = sc.transform(F[tr]), sc.transform(F[te])
    if kind == "ridge":
        return np.clip(Ridge(alpha=10.0 * Xtr.shape[1] / 3072).fit(Xtr, y[tr]).predict(Xte), 0, 1)
    pca = PCA(512, random_state=SEED).fit(Xtr)
    Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
    est = MLPRegressor(hidden_layer_sizes=(512, 128), alpha=1e-3, batch_size=256, learning_rate_init=1e-3,
                       max_iter=80, early_stopping=True, n_iter_no_change=6, random_state=SEED).fit(Xtr, y[tr])
    return np.clip(est.predict(Xte), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--target", required=True, help="evaluated model whose scores are the main target")
    ap.add_argument("--no-mlp", action="store_true")
    args = ap.parse_args()
    b = HERE / "bundle"
    meta = pd.read_csv(b / "entities.csv")
    meta["group"] = meta.region + "|" + meta.categoria
    meta["name_key"] = meta.entidad.map(lambda s: re.sub(r"[^\w]+", " ", str(s).lower()).strip())
    groups = meta.group.to_numpy()
    T = pd.read_csv(b / "targets.csv").set_index("entity_id").reindex(meta.entity_id)
    pv = np.log10(pd.read_csv(b / "pageviews.csv").set_index("entity_id")
                  .reindex(meta.entity_id).pageviews_60d.to_numpy(float) + 1)
    feats = {}
    for tname in ["contextual", "name", "entity"]:
        z = np.load(HERE / "hidden" / args.tag / f"{tname}.npz")
        assert len(z["last"]) == len(meta), "hidden states do not cover all entities (run without --limit)"
        for pool in ["last", "mean"]:
            for j, layer in enumerate(z["layers"]):
                feats[(tname, pool, int(layer), int(z["n_layers"]))] = z[pool][:, j, :].astype(np.float32)
    folds = list(GroupKFold(5).split(meta, groups=meta.name_key))
    out, own_preds = [], {}
    targets = [args.target] + [m for m in T.columns if m != args.target]
    for model in targets:
        y = T[model].to_numpy()
        ok = ~np.isnan(y)
        fk = [(tr[ok[tr]], te[ok[te]]) for tr, te in folds]
        preds = {"category x region mean": np.full(len(y), np.nan)}
        for tr, te in fk:
            mu = pd.Series(y[tr]).groupby(groups[tr]).mean()
            preds["category x region mean"][te] = pd.Series(groups[te]).map(mu).to_numpy()
        gm = preds["category x region mean"]
        X = pd.get_dummies(meta.group).assign(pv=pv).to_numpy(float)
        k = "page views + category x region (GBM)"
        preds[k] = np.full(len(y), np.nan)
        for tr, te in fk:
            preds[k][te] = np.clip(HistGradientBoostingRegressor(max_iter=300, random_state=SEED)
                                   .fit(X[tr], y[tr]).predict(X[te]), 0, 1)
        for (tname, pool, layer, nl), F in feats.items():
            # Ridge (linear, as in KEEN) for every input; MLP and the permutation control for the two main inputs.
            # (MLP only at the middle layer, fixed a priori, to keep the runtime reasonable.)
            mid = layer == max(1, round(0.5 * nl))
            kinds = ["ridge"] + ([] if args.no_mlp or model != args.target or tname not in ("contextual", "entity")
                                 or not mid else ["mlp", "mlp_perm"])
            for kind in kinds:
                key = f"{kind} ({tname}, {pool}, layer {layer}/{nl})"
                preds[key] = np.full(len(y), np.nan)
                Fk = permute_within(F, groups) if kind == "mlp_perm" else F
                for tr, te in fk:
                    preds[key][te] = fit_predict("mlp" if kind.startswith("mlp") else "ridge", Fk, y, tr, te)
                print(args.tag, "->", model, key, flush=True)
        if model == args.target:
            own_preds = preds
        for name, pr in preds.items():
            for row in evaluate(name, y[ok], pr[ok], meta[ok].reset_index(drop=True), gm[ok]):
                out.append(dict(features=args.tag, model=model, **row))
    res = pd.DataFrame(out)
    (HERE / "results").mkdir(exist_ok=True)
    res.to_csv(HERE / "results" / f"E24_hidden_probe_{args.tag}.csv", index=False)
    # Transfer: predictors trained on the target model's scores evaluated on the other models' scores.
    rows = []
    for key, pr in own_preds.items():
        if not key.startswith(("ridge", "mlp (")):
            continue
        for m in T.columns:
            y = T[m].to_numpy()
            ok = ~np.isnan(y) & ~np.isnan(pr)
            gm = pd.Series(y[ok]).groupby(groups[ok]).transform("mean").to_numpy()
            rows.append(dict(features=args.tag, predictor=key, trained_on=args.target, evaluated_on=m,
                             within_group_spearman=within_group_spearman(y[ok], pr[ok], groups[ok], gm)))
    pd.DataFrame(rows).to_csv(HERE / "results" / f"E24_hidden_probe_{args.tag}_transfer.csv", index=False)
    s = res[(res.region == "All") & (res.model == args.target)].sort_values("r2", ascending=False)
    print(s[["predictor", "r2", "within_group_spearman"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
