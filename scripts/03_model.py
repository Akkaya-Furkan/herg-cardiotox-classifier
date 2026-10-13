"""Day 2: scaffold vs random split, RandomForest and LightGBM, evaluation, SHAP.

Reads data/processed/features.npz (from 02_prepare.py). Writes:
  results/metrics.json                 CV and held-out metrics per split x model, applicability domain
  results/band_sensitivity.csv         how the excluded band around pIC50 5 changes the scaffold-split score
  results/shap_top20.csv
  figures/fig1_roc_pr_calibration.png
  figures/fig2_shap_summary.png
  models/lightgbm_scaffold.joblib      model for P04's safety flag (features: see feature_names)
"""
import importlib.util
import json
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import shap
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss, matthews_corrcoef,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold, KFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RES, FIG, MOD = ROOT / "results", ROOT / "figures", ROOT / "models"
SEED, NJ = 42, 8

_spec = importlib.util.spec_from_file_location("prep", ROOT / "scripts" / "02_prepare.py")
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)


def scaffold_split(scaffolds, frac_test=0.2, seed=SEED):
    """Whole scaffold groups go to one side: shuffled groups fill train, the rest is test."""
    groups = pd.Series(np.arange(len(scaffolds))).groupby(pd.Series(scaffolds)).apply(list).tolist()
    np.random.RandomState(seed).shuffle(groups)
    n_train = len(scaffolds) - int(len(scaffolds) * frac_test)
    tr, te = [], []
    for g in groups:
        (tr if len(tr) + len(g) <= n_train else te).extend(g)
    return np.array(tr), np.array(te)


def random_split(n, frac_test=0.2, seed=SEED):
    perm = np.random.RandomState(seed).permutation(n)
    k = int(n * frac_test)
    return perm[k:], perm[:k]


def make_rf():
    return RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", n_jobs=NJ, random_state=SEED)


def make_lgb():
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.3, n_jobs=NJ, random_state=SEED, verbose=-1)


MODELS = {"RandomForest": make_rf, "LightGBM": make_lgb}


def scores(y, p):
    return {"ROC_AUC": roc_auc_score(y, p), "PR_AUC": average_precision_score(y, p),
            "MCC": matthews_corrcoef(y, p >= 0.5), "Brier": brier_score_loss(y, p)}


def rounded(d):
    return {k: round(float(v), 4) for k, v in d.items()}


def main():
    for p in (RES, FIG, MOD):
        p.mkdir(exist_ok=True)
    z = np.load(ROOT / "data" / "processed" / "features.npz", allow_pickle=True)
    X, y, scaf, names = z["X"], z["y"], z["scaffold"], list(z["feature_names"])
    splits = {"scaffold": scaffold_split(scaf), "random": random_split(len(y))}
    out = {"n": int(len(y)), "positive_fraction": round(float(y.mean()), 4), "splits": {}, "results": {}}
    for s, (tr, te) in splits.items():
        out["splits"][s] = {"train": len(tr), "test": len(te), "test_positive_fraction": round(float(y[te].mean()), 4),
                            "scaffolds_shared_train_test": len(set(scaf[tr]) & set(scaf[te]))}

    preds, fitted = {}, {}
    for s, (tr, te) in splits.items():
        # CV inside the training set: scaffold-grouped folds for the scaffold split, shuffled folds otherwise
        folds = (GroupKFold(n_splits=5).split(tr, groups=scaf[tr]) if s == "scaffold"
                 else KFold(n_splits=5, shuffle=True, random_state=SEED).split(tr))
        folds = list(folds)
        for name, make in MODELS.items():
            cv = pd.DataFrame([scores(y[tr][b], make().fit(X[tr][a], y[tr][a]).predict_proba(X[tr][b])[:, 1])
                               for a, b in folds])
            model = make().fit(X[tr], y[tr])
            p = model.predict_proba(X[te])[:, 1]
            out["results"][f"{s}/{name}"] = {"cv_mean": rounded(cv.mean()), "cv_sd": rounded(cv.std()),
                                             "test": rounded(scores(y[te], p))}
            preds[s, name], fitted[s, name] = (y[te], p), model
            print(s, name, out["results"][f"{s}/{name}"]["test"], flush=True)

    # applicability domain: held-out scaffold-test compounds binned by Tanimoto to their nearest training neighbour
    tr, te = splits["scaffold"]
    smiles = pd.read_csv(ROOT / "data" / "processed" / "herg_compounds.csv").query("label >= 0").smiles.values
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]
    train_fps = [fps[i] for i in tr]
    nn = np.array([max(DataStructs.BulkTanimotoSimilarity(fps[i], train_fps)) for i in te])
    yt, pt = preds["scaffold", "LightGBM"]
    ad = []
    for lo, hi in [(0, .4), (.4, .5), (.5, .6), (.6, .7), (.7, 1.01)]:
        k = (nn >= lo) & (nn < hi)
        row = {"nn_tanimoto": f"{lo:.1f}-{min(hi, 1):.1f}", "n": int(k.sum())}
        if k.sum() >= 30 and len(set(yt[k])) == 2:
            row["ROC_AUC"] = round(float(roc_auc_score(yt[k], pt[k])), 4)
        ad.append(row)
    out["applicability_domain"] = ad
    out["test_nn_tanimoto_median"] = round(float(np.median(nn)), 4)

    # sensitivity to the excluded band around the pIC50 5 threshold (LightGBM, scaffold split)
    allc = pd.read_csv(ROOT / "data" / "processed" / "herg_compounds.csv")
    sens = []
    for band in (0.0, 0.3, 0.5):
        c = allc[(allc.pIC50 - prep.THRESHOLD).abs() >= band].reset_index(drop=True) if band else allc
        yb = (c.pIC50 >= prep.THRESHOLD).astype(int).values
        Xb = prep.featurise(c.smiles)
        a, b = scaffold_split(c.scaffold.fillna("").values)
        r = scores(yb[b], make_lgb().fit(Xb[a], yb[a]).predict_proba(Xb[b])[:, 1])
        sens.append({"band": band, "n": len(c), "positive_fraction": round(float(yb.mean()), 4), **rounded(r)})
        print("band", sens[-1], flush=True)
    pd.DataFrame(sens).to_csv(RES / "band_sensitivity.csv", index=False)
    out["band_sensitivity"] = sens

    colours = {"RandomForest": "#1f77b4", "LightGBM": "#d62728"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for (s, name), (yt_, p_) in preds.items():
        ls = "-" if s == "scaffold" else "--"
        f, t, _ = roc_curve(yt_, p_)
        ax[0].plot(f, t, ls, c=colours[name], label=f"{name}, {s} split (AUC {roc_auc_score(yt_, p_):.3f})")
        pr, rc, _ = precision_recall_curve(yt_, p_)
        ax[1].plot(rc, pr, ls, c=colours[name], label=f"{name}, {s} split (AP {average_precision_score(yt_, p_):.3f})")
        if s == "scaffold":
            frac, mean_p = calibration_curve(yt_, p_, n_bins=10, strategy="quantile")
            ax[2].plot(mean_p, frac, "o-", c=colours[name], label=f"{name} (Brier {brier_score_loss(yt_, p_):.3f})")
    ax[0].plot([0, 1], [0, 1], ":k")
    ax[2].plot([0, 1], [0, 1], ":k")
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (held-out test)")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall (held-out test)")
    ax[2].set(xlabel="Mean predicted probability", ylabel="Observed fraction of blockers", title="Calibration (scaffold test)")
    for a in ax:
        a.legend(fontsize=7)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_roc_pr_calibration.png", dpi=200)

    model = fitted["scaffold", "LightGBM"]
    idx = np.random.RandomState(SEED).choice(te, size=min(800, len(te)), replace=False)
    sv = np.asarray(shap.TreeExplainer(model).shap_values(X[idx]))
    if sv.ndim == 3:
        sv = sv[..., 1]
    plt.figure(figsize=(8, 6.5))
    shap.summary_plot(sv, X[idx], feature_names=names, max_display=20, show=False)
    plt.gcf().tight_layout()
    plt.gcf().savefig(FIG / "fig2_shap_summary.png", dpi=200)
    top = pd.Series(np.abs(sv).mean(0), index=names).sort_values(ascending=False).head(20)
    top.rename("mean_abs_shap").to_csv(RES / "shap_top20.csv", index_label="feature")
    out["shap_top10"] = [f"{k} ({v:.3f})" for k, v in top.head(10).items()]

    joblib.dump({"model": model, "feature_names": names, "threshold_pIC50": prep.THRESHOLD,
                 "featurise": "scripts/02_prepare.py:featurise"}, MOD / "lightgbm_scaffold.joblib")
    (RES / "metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("splits", "applicability_domain", "shap_top10")}, indent=2))


if __name__ == "__main__":
    main()
