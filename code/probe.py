"""Embedding-based predictor of entity-level scores, designed around reviewer NCxe's W3.

Target: entity-level mean LLM-judge score of each evaluated model (one predictor per model).
Features: text-embedding-3-large embeddings of the entity (src/embeddings.py), with a
contextual template (entity, category, country) and a name-only ablation.
Protocol: 5-fold CV grouped by normalised entity name (a name never appears in both train
and test, even across regions).

Controls (each answers one point of W3):
  * group means: category x region mean, and page views + category x region (GBM);
  * capacity control: the same MLP trained on embeddings permuted *within* category x region,
    which keeps group information and destroys entity-specific information;
  * within-group Spearman: correlation after subtracting the training-fold category x region
    mean from targets and predictions;
  * per-region R^2 and MAE / target SD (lower MAE alone may just mean a less variable target);
  * cross-model transfer: a predictor trained on one model's scores evaluated on the others.

Usage:  python src/probe.py
"""
import re

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor

from common import CACHE, RESULTS, complete_models, entity_scores, load

EMB = CACHE / "embeddings"
SEED = 13
REGIONS = ["LATAM", "Europe", "USA"]
RNG = np.random.default_rng(SEED)


def targets():
    models, _ = complete_models(verbose=False)
    e = entity_scores(load(models))
    return e.groupby(["entity_id", "model"]).score_gpt.mean().unstack()


def within_group_spearman(y, pred, groups, group_means):
    ry, rp = y - group_means, pred - group_means
    vals = [stats.spearmanr(ry[groups == k], rp[groups == k])[0]
            for k in np.unique(groups) if (groups == k).sum() > 30]
    # Constant predictions within a group (e.g., the group-mean baseline) give NaN.
    return np.nan if np.all(np.isnan(vals)) else np.nanmean(vals)


def evaluate(name, y, pred, meta, gm):
    rows = []
    for r in REGIONS + ["All"]:
        m = np.ones(len(y), bool) if r == "All" else (meta.region == r).to_numpy()
        yt, pt = y[m], pred[m]
        rows.append(dict(predictor=name, region=r, n=m.sum(), target_sd=yt.std(), mae=np.abs(yt - pt).mean(),
                         mae_over_sd=np.abs(yt - pt).mean() / yt.std(),
                         r2=1 - ((yt - pt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum(),
                         spearman=stats.spearmanr(yt, pt)[0],
                         within_group_spearman=within_group_spearman(yt, pt, meta.group.to_numpy()[m], gm[m])))
    return rows


def permute_within(F, groups):
    """Rows of F shuffled within each group: group-level structure kept, entity identity broken."""
    idx = np.arange(len(F))
    for g in np.unique(groups):
        m = np.where(groups == g)[0]
        idx[m] = RNG.permutation(m)
    return F[idx]


def fit_predict(kind, F, y, tr, te):
    if kind == "ridge":
        est = Ridge(alpha=10.0).fit(F[tr], y[tr])
        return np.clip(est.predict(F[te]), 0, 1)
    pca = PCA(512, random_state=SEED).fit(F[tr])
    Xtr, Xte = pca.transform(F[tr]), pca.transform(F[te])
    est = MLPRegressor(hidden_layer_sizes=(512, 128), alpha=1e-3, batch_size=256, learning_rate_init=1e-3,
                       max_iter=80, early_stopping=True, n_iter_no_change=6, random_state=SEED).fit(Xtr, y[tr])
    return np.clip(est.predict(Xte), 0, 1)


def main():
    meta = pd.read_csv(EMB / "entities.csv")
    meta["group"] = meta.region + "|" + meta.categoria
    meta["name_key"] = meta.entidad.map(lambda s: re.sub(r"[^\w]+", " ", str(s).lower()).strip())
    groups = meta.group.to_numpy()
    T = targets().reindex(meta.entity_id)
    feats = {t: np.load(EMB / f"{t}.npy") for t in ["contextual", "name"] if (EMB / f"{t}.npy").exists()}
    feats["contextual, permuted within category x region"] = permute_within(feats["contextual"], groups)
    pv = None
    if (RESULTS / "E13_pageviews.csv").exists():
        p = pd.read_csv(RESULTS / "E13_pageviews.csv").set_index("entity_id")
        pv = np.log10(p.reindex(meta.entity_id).pageviews_60d.to_numpy(float) + 1)

    folds = list(GroupKFold(5).split(meta, groups=meta.name_key))
    out, all_preds = [], {}
    for model in T.columns:
        y = T[model].to_numpy()
        ok = ~np.isnan(y)
        fk = [(tr[ok[tr]], te[ok[te]]) for tr, te in folds]
        preds = {"global mean": np.full(len(y), np.nan), "category x region mean": np.full(len(y), np.nan)}
        for tr, te in fk:
            preds["global mean"][te] = y[tr].mean()
            mu = pd.Series(y[tr]).groupby(groups[tr]).mean()
            preds["category x region mean"][te] = pd.Series(groups[te]).map(mu).to_numpy()
        gm = preds["category x region mean"]
        if pv is not None:
            X = pd.get_dummies(meta.group).assign(pv=pv).to_numpy(float)
            k = "page views + category x region (GBM)"
            preds[k] = np.full(len(y), np.nan)
            for tr, te in fk:
                gb = HistGradientBoostingRegressor(max_iter=300, random_state=SEED).fit(X[tr], y[tr])
                preds[k][te] = np.clip(gb.predict(X[te]), 0, 1)
        for tname, F in feats.items():
            for kind in ["ridge", "mlp"]:
                if kind == "ridge" and "permuted" in tname:
                    continue
                key = f"{kind} ({tname})"
                preds[key] = np.full(len(y), np.nan)
                for tr, te in fk:
                    preds[key][te] = fit_predict(kind, F, y, tr, te)
                print(model, key, "done", flush=True)
        all_preds[model] = preds
        for name, pr in preds.items():
            for row in evaluate(name, y[ok], pr[ok], meta[ok].reset_index(drop=True), gm[ok]):
                out.append(dict(model=model, **row))
    res = pd.DataFrame(out)
    res.to_csv(RESULTS / "E14_probe.csv", index=False)

    # Cross-model transfer of the contextual MLP predictor (within-group Spearman).
    tr_rows = []
    for src in T.columns:
        pr = all_preds[src]["mlp (contextual)"]
        for tgt in T.columns:
            y = T[tgt].to_numpy()
            ok = ~np.isnan(y) & ~np.isnan(pr)
            gm = all_preds[tgt]["category x region mean"]
            tr_rows.append(dict(trained_on=src, evaluated_on=tgt, spearman=stats.spearmanr(y[ok], pr[ok])[0],
                                within_group_spearman=within_group_spearman(y[ok], pr[ok], groups[ok], gm[ok])))
    transfer = pd.DataFrame(tr_rows)
    transfer.to_csv(RESULTS / "E14_probe_transfer.csv", index=False)
    print(res[res.region == "All"].round(3).to_string(index=False))
    print(transfer.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
