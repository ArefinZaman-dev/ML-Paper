#!/usr/bin/env python3
"""
Road crash severity (Bangladesh newspaper dataset) - full benchmark.

Run:
    pip install pandas numpy scikit-learn scipy matplotlib openpyxl
    pip install imbalanced-learn xgboost lightgbm        # optional but recommended
    python run_experiments.py --data Data_Collection_Recap_CLEANED_v1.xlsx

Quick test (1-2 minutes):  python run_experiments.py --quick
Everything is written to ./results/ . Send those CSV/PNG files back.

Target  : 3 classes from Deaths ONLY  (0 = Non-fatal, 1-2 = Severe, >=3 = Extremely severe)
Leakage : Deaths and Injured are NEVER used as input features.
"""
import argparse, os, sys, time, warnings, platform
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, recall_score, cohen_kappa_score,
                             roc_auc_score, average_precision_score, confusion_matrix)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

# ---------- optional libraries ----------
HAVE_SMOTE = True  # built-in SMOTE implementation below (no imblearn needed)


def smote_fit_resample(X, y, k, rng):
    """Standard SMOTE (Chawla et al. 2002) on the encoded matrix; oversample all classes to the majority size."""
    from sklearn.neighbors import NearestNeighbors
    cls, cnt = np.unique(y, return_counts=True)
    tgt = cnt.max(); Xs, ys = [X], [y]
    for c, n in zip(cls, cnt):
        if n == tgt:
            continue
        Xc = X[y == c]
        kk = min(k, len(Xc) - 1)
        nn = NearestNeighbors(n_neighbors=kk + 1).fit(Xc)
        nbr = nn.kneighbors(Xc, return_distance=False)[:, 1:]
        base = rng.integers(0, len(Xc), tgt - n)
        pick = nbr[base, rng.integers(0, kk, tgt - n)]
        lam = rng.random((tgt - n, 1))
        Xs.append(Xc[base] + lam * (Xc[pick] - Xc[base])); ys.append(np.full(tgt - n, c))
    return np.vstack(Xs), np.concatenate(ys)
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False
try:
    from lightgbm import LGBMClassifier
    HAVE_LGBM = True
except Exception:
    HAVE_LGBM = False

# ---------- args ----------
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="Data_Collection_Recap_CLEANED_v1.xlsx")
ap.add_argument("--out", default="results")
ap.add_argument("--quick", action="store_true", help="1 CV repeat, fewer trees (smoke test)")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
SEED = args.seed
N_REPEATS = 1 if args.quick else 5
N_TREES = 100 if args.quick else 300
LABELS = ["Non-fatal", "Severe", "Extremely severe"]
CLASSES = [0, 1, 2]


def P(name):
    return os.path.join(args.out, name)


def log(*a):
    print(*a, flush=True)


# =====================================================================
# 1. LOAD + FEATURES
# =====================================================================
raw = pd.read_excel(args.data, sheet_name="Cleaned_Data")
df = raw.copy()
df["Crash_Date"] = pd.to_datetime(df["Crash_Date"], errors="coerce")
df["Deaths"] = pd.to_numeric(df["Deaths"], errors="coerce")


def sev(d):
    if pd.isna(d):
        return np.nan
    return 0 if d == 0 else (1 if d <= 2 else 2)


df["y"] = df["Deaths"].apply(sev)               # computed here, NOT read from the Excel formula
df = df[df["y"].notna()].copy()
df["y"] = df["y"].astype(int)
df["Year"] = df["Crash_Date"].dt.year
df["Month"] = df["Crash_Date"].dt.month_name().fillna("Unknown")
df["Weekday"] = df["Crash_Date"].dt.day_name().fillna("Unknown")
for c in ["Time_Slot", "Road_Type", "Division", "District", "Newspaper"]:
    df[c] = df[c].fillna("Unknown").astype(str)
df["Vehicle_1"] = df["Vehicle_1"].fillna("Unknown").astype(str)
df["Vehicle_2"] = df["Vehicle_2"].fillna("None (single vehicle)").astype(str)
df["Pedestrian"] = df["Pedestrian"].fillna("No").astype(str)

HEAVY = {"Bus", "Truck", "Covered van", "Tractor/Trolley", "Train"}
LIGHT = {"Motorcycle", "Rickshaw", "Battery rickshaw (easy-bike)", "Auto-rickshaw", "Bicycle",
         "CNG auto-rickshaw", "Human hauler"}
v = df[["Vehicle_1", "Vehicle_2"]]
df["n_vehicles"] = (df["Vehicle_1"].ne("Unknown").astype(int)
                    + df["Vehicle_2"].ne("None (single vehicle)").astype(int))
df["heavy"] = v.isin(HEAVY).any(axis=1).astype(int)
df["light"] = v.isin(LIGHT).any(axis=1).astype(int)
df["flagged"] = df["Review_Flag"].notna()
df = df.reset_index(drop=True)
NUM = ["n_vehicles", "heavy", "light"]

FEATURE_SETS = {
    "core":                {"cat": ["Time_Slot", "Division", "Road_Type", "Vehicle_1", "Vehicle_2",
                                    "Pedestrian", "Month", "Weekday"], "num": NUM},
    "core+District":       {"cat": ["Time_Slot", "Division", "District", "Road_Type", "Vehicle_1",
                                    "Vehicle_2", "Pedestrian", "Month", "Weekday"], "num": NUM},
    "core+Newspaper":      {"cat": ["Time_Slot", "Division", "Road_Type", "Vehicle_1", "Vehicle_2",
                                    "Pedestrian", "Month", "Weekday", "Newspaper"], "num": NUM},
    "vehicles_only":       {"cat": ["Vehicle_1", "Vehicle_2", "Pedestrian"], "num": NUM},
    "context_only":        {"cat": ["Time_Slot", "Division", "Road_Type", "Month", "Weekday"], "num": []},
}
MAIN_FS = "core"

log(f"Rows used: {len(df)} | class counts: {df['y'].value_counts().sort_index().to_dict()}")
log(f"SMOTE={HAVE_SMOTE}  XGB={HAVE_XGB}  LGBM={HAVE_LGBM}")

# =====================================================================
# 2. MODELS / IMBALANCE STRATEGIES
# =====================================================================
MODELS = ["LR", "DT", "RF", "ET", "HGB"]
if HAVE_XGB:
    MODELS.append("XGB")
if HAVE_LGBM:
    MODELS.append("LGBM")
STRATS = ["none", "classweight", "ROS", "RUS"] + (["SMOTE"] if HAVE_SMOTE else [])


def get_model(name, seed):
    if name == "LR":
        return LogisticRegression(max_iter=3000)
    if name == "DT":
        return DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=seed)
    if name == "RF":
        return RandomForestClassifier(n_estimators=N_TREES, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    if name == "ET":
        return ExtraTreesClassifier(n_estimators=N_TREES, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    if name == "HGB":
        return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200, random_state=seed)
    if name == "XGB":
        return XGBClassifier(n_estimators=N_TREES, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, eval_metric="mlogloss", n_jobs=-1, random_state=seed,
                             verbosity=0)
    if name == "LGBM":
        return LGBMClassifier(n_estimators=N_TREES, learning_rate=0.05, num_leaves=15, subsample=0.8,
                              colsample_bytree=0.8, n_jobs=-1, random_state=seed, verbose=-1)
    if name == "Dummy_majority":
        return DummyClassifier(strategy="prior")
    raise ValueError(name)


def resample(X, y, kind, rng):
    cls, cnt = np.unique(y, return_counts=True)
    if kind == "ROS":
        tgt = cnt.max()
        idx = np.concatenate([np.concatenate([np.where(y == c)[0],
                              rng.choice(np.where(y == c)[0], tgt - n, replace=True)])
                              for c, n in zip(cls, cnt)])
        return X[idx], y[idx]
    if kind == "RUS":
        tgt = cnt.min()
        idx = np.concatenate([rng.choice(np.where(y == c)[0], tgt, replace=False) for c in cls])
        return X[idx], y[idx]
    if kind == "SMOTE":
        k = max(1, min(5, cnt.min() - 1))
        return smote_fit_resample(X, y, k, rng)
    return X, y


def full_proba(model, X):
    p = model.predict_proba(X)
    out = np.zeros((X.shape[0], 3))
    for j, c in enumerate(model.classes_):
        out[:, int(c)] = p[:, j]
    return out


def fit(train, fs, mname, strat, seed):
    cat, num = FEATURE_SETS[fs]["cat"], FEATURE_SETS[fs]["num"]
    enc = ColumnTransformer(
        [("c", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5, sparse_output=False), cat)]
        + ([("n", "passthrough", num)] if num else []))
    X = enc.fit_transform(train[cat + num])
    y = train["y"].values
    rng = np.random.default_rng(seed)
    sw = None
    if strat in ("ROS", "RUS", "SMOTE"):
        X, y = resample(X, y, strat, rng)
    elif strat == "classweight":
        sw = compute_sample_weight("balanced", y)
    model = get_model(mname, seed)
    if mname == "LR" and strat == "classweight":
        model.set_params(class_weight="balanced"); sw = None
    if mname in ("DT", "RF", "ET") and strat == "classweight":
        model.set_params(class_weight="balanced"); sw = None
    if mname == "LGBM" and strat == "classweight":
        model.set_params(class_weight="balanced"); sw = None
    if sw is not None:
        model.fit(X, y, sample_weight=sw)
    else:
        model.fit(X, y)
    return enc, model, cat + num


def predict(fitted, test):
    enc, model, cols = fitted
    X = enc.transform(test[cols])
    proba = full_proba(model, X)
    return proba.argmax(1), proba


def metrics(y, pred, proba):
    rec = recall_score(y, pred, labels=CLASSES, average=None, zero_division=0)
    f1c = f1_score(y, pred, labels=CLASSES, average=None, zero_division=0)
    m = {"acc": accuracy_score(y, pred),
         "bal_acc": balanced_accuracy_score(y, pred),
         "macro_f1": f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0),
         "weighted_f1": f1_score(y, pred, labels=CLASSES, average="weighted", zero_division=0),
         "mcc": matthews_corrcoef(y, pred),
         "gmean": float(np.prod(rec) ** (1 / 3)),
         "qwk": cohen_kappa_score(y, pred, weights="quadratic"),
         "mae_ord": float(np.mean(np.abs(y - pred)))}
    for c, n in zip(CLASSES, LABELS):
        m[f"recall_{n}"] = rec[c]
        m[f"f1_{n}"] = f1c[c]
    try:
        m["auroc_macro"] = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=CLASSES)
    except Exception:
        m["auroc_macro"] = np.nan
    m["ap_nonfatal"] = average_precision_score((y == 0).astype(int), proba[:, 0]) if (y == 0).any() else np.nan
    return m


def cv_eval(data, fs, mname, strat, n_repeats=N_REPEATS, tag=""):
    rows = []
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    for k, (tr, te) in enumerate(rskf.split(data, data["y"])):
        f = fit(data.iloc[tr], fs, mname, strat, SEED + k)
        pred, proba = predict(f, data.iloc[te])
        r = metrics(data.iloc[te]["y"].values, pred, proba)
        r.update(model=mname, strategy=strat, featureset=fs, fold=k, tag=tag)
        rows.append(r)
    return pd.DataFrame(rows)


def summarize(d, by):
    num = [c for c in d.columns if c not in by + ["fold", "tag"] and pd.api.types.is_numeric_dtype(d[c])]
    g = d.groupby(by)[num]
    s = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd"))
    return s.reset_index()


# =====================================================================
# 3. MAIN GRID  (repeated stratified 5-fold CV)
# =====================================================================
t0 = time.time()
grid = []
combos = [("Dummy_majority", "none")] + [(m, s) for m in MODELS for s in STRATS]
for i, (m, s) in enumerate(combos, 1):
    grid.append(cv_eval(df, MAIN_FS, m, s))
    log(f"[CV {i}/{len(combos)}] {m:15s} {s:12s} macroF1={grid[-1]['macro_f1'].mean():.3f}  ({time.time()-t0:.0f}s)")
cv = pd.concat(grid, ignore_index=True)
cv.to_csv(P("01_cv_fold_metrics.csv"), index=False)
cv_sum = summarize(cv, ["model", "strategy", "featureset"]).sort_values("macro_f1_mean", ascending=False)
cv_sum.to_csv(P("02_cv_summary.csv"), index=False)

cand = cv_sum[cv_sum["model"] != "Dummy_majority"].iloc[0]
BEST_M, BEST_S = cand["model"], cand["strategy"]
log(f"\nBEST (by mean macro-F1): {BEST_M} + {BEST_S}  macroF1={cand['macro_f1_mean']:.3f}")

# heatmap
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hm = cv_sum[cv_sum["model"] != "Dummy_majority"].pivot(index="model", columns="strategy", values="macro_f1_mean")
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(hm.values, cmap="viridis")
    ax.set_xticks(range(hm.shape[1])); ax.set_xticklabels(hm.columns)
    ax.set_yticks(range(hm.shape[0])); ax.set_yticklabels(hm.index)
    for (i, j), val in np.ndenumerate(hm.values):
        ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="w", fontsize=8)
    ax.set_title("Macro-F1 (5x5 CV): model x imbalance strategy")
    plt.colorbar(im); plt.tight_layout(); plt.savefig(P("03_heatmap_macroF1.png"), dpi=300); plt.close()
except Exception as e:
    log("heatmap skipped:", e)

# =====================================================================
# 4. STATISTICAL TESTS (paired Wilcoxon on per-fold macro-F1)
# =====================================================================
st = []
for m in MODELS:
    base = cv[(cv.model == m) & (cv.strategy == "none")].sort_values("fold")["macro_f1"].values
    for s in STRATS:
        if s == "none":
            continue
        x = cv[(cv.model == m) & (cv.strategy == s)].sort_values("fold")["macro_f1"].values
        try:
            w = stats.wilcoxon(x, base)
            p = w.pvalue
        except Exception:
            p = np.nan
        st.append({"model": m, "strategy_vs_none": s, "mean_diff_macroF1": float(np.mean(x - base)), "wilcoxon_p": p})
dum = cv[cv.model == "Dummy_majority"].sort_values("fold")["macro_f1"].values
bestv = cv[(cv.model == BEST_M) & (cv.strategy == BEST_S)].sort_values("fold")["macro_f1"].values
st.append({"model": f"{BEST_M}+{BEST_S}", "strategy_vs_none": "vs Dummy_majority",
           "mean_diff_macroF1": float(np.mean(bestv - dum)), "wilcoxon_p": stats.wilcoxon(bestv, dum).pvalue})
pd.DataFrame(st).to_csv(P("04_stat_tests.csv"), index=False)

# =====================================================================
# 5. OUT-OF-FOLD PREDICTIONS, CONFUSION MATRIX, BOOTSTRAP CI (best config)
# =====================================================================
oof_pred = np.zeros(len(df), int); oof_proba = np.zeros((len(df), 3))
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
for k, (tr, te) in enumerate(skf.split(df, df["y"])):
    f = fit(df.iloc[tr], MAIN_FS, BEST_M, BEST_S, SEED + k)
    oof_pred[te], oof_proba[te] = predict(f, df.iloc[te])
pd.DataFrame({"Record_ID": df["Record_ID"], "y_true": df["y"], "y_pred": oof_pred,
              "p_nonfatal": oof_proba[:, 0], "p_severe": oof_proba[:, 1], "p_extreme": oof_proba[:, 2]}
             ).to_csv(P("05_oof_predictions_best.csv"), index=False)
cm = confusion_matrix(df["y"], oof_pred, labels=CLASSES)
pd.DataFrame(cm, index=[f"true_{l}" for l in LABELS], columns=[f"pred_{l}" for l in LABELS]).to_csv(P("06_confusion_matrix_best.csv"))
try:
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5, 4.3))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(LABELS, rotation=20); ax.set_yticks(range(3)); ax.set_yticklabels(LABELS)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f"{val}\n({cmn[i, j]:.0%})", ha="center", va="center", fontsize=8,
                color="w" if cmn[i, j] > .5 else "k")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"{BEST_M}+{BEST_S} (OOF, 5-fold)")
    plt.tight_layout(); plt.savefig(P("06_confusion_matrix_best.png"), dpi=300); plt.close()
except Exception as e:
    log("cm plot skipped:", e)

rng = np.random.default_rng(SEED)
bs = []
yv = df["y"].values
for _ in range(1000):
    i = rng.integers(0, len(df), len(df))
    bs.append([f1_score(yv[i], oof_pred[i], average="macro", labels=CLASSES, zero_division=0),
               balanced_accuracy_score(yv[i], oof_pred[i]), matthews_corrcoef(yv[i], oof_pred[i])])
bs = np.array(bs)
pd.DataFrame({"metric": ["macro_f1", "bal_acc", "mcc"], "mean": bs.mean(0),
              "ci_low_2.5": np.percentile(bs, 2.5, 0), "ci_high_97.5": np.percentile(bs, 97.5, 0)}
             ).to_csv(P("07_bootstrap_ci_best.csv"), index=False)

# =====================================================================
# 6. TEMPORAL SPLIT + LEAVE-ONE-NEWSPAPER-OUT (all model x strategy)
# =====================================================================
tr_t = df[df["Year"] <= 2024]; te_t = df[df["Year"] >= 2025]
log(f"\nTemporal split: train={len(tr_t)} (<=2024), test={len(te_t)} (>=2025); test classes={te_t['y'].value_counts().sort_index().to_dict()}")
rows = []
for m, s in combos:
    f = fit(tr_t, MAIN_FS, m, s, SEED)
    pred, proba = predict(f, te_t)
    r = metrics(te_t["y"].values, pred, proba); r.update(model=m, strategy=s, n_train=len(tr_t), n_test=len(te_t)); rows.append(r)
pd.DataFrame(rows).sort_values("macro_f1", ascending=False).to_csv(P("08_temporal_split.csv"), index=False)

top_np = df["Newspaper"].value_counts().head(5).index.tolist()
rows = []
for npname in top_np:
    tr_n = df[df["Newspaper"] != npname]; te_n = df[df["Newspaper"] == npname]
    for m, s in combos:
        f = fit(tr_n, MAIN_FS, m, s, SEED)
        pred, proba = predict(f, te_n)
        r = metrics(te_n["y"].values, pred, proba)
        r.update(model=m, strategy=s, held_out=npname, n_test=len(te_n)); rows.append(r)
lono = pd.DataFrame(rows)
lono.to_csv(P("09_leave_one_newspaper_out.csv"), index=False)
lono_avg = lono.groupby(["model", "strategy"])[["macro_f1", "bal_acc", "mcc", "gmean"]].mean().reset_index()
lono_avg.sort_values("macro_f1", ascending=False).to_csv(P("09b_lono_average.csv"), index=False)

# =====================================================================
# 7. FEATURE-SET ABLATION + SENSITIVITY (best config)
# =====================================================================
ab = pd.concat([cv_eval(df, fs, BEST_M, BEST_S, tag="ablation") for fs in FEATURE_SETS], ignore_index=True)
summarize(ab, ["model", "strategy", "featureset"]).to_csv(P("10_ablation_featuresets.csv"), index=False)

sens = []
for name, sub in [("all labelled rows", df),
                  ("excluding flagged rows", df[~df["flagged"]]),
                  ("year >= 2020 only", df[df["Year"] >= 2020])]:
    d = cv_eval(sub.reset_index(drop=True), MAIN_FS, BEST_M, BEST_S, tag=name)
    d["subset"] = name; d["n_rows"] = len(sub); sens.append(d)
sens = pd.concat(sens)
summarize(sens, ["subset", "n_rows", "model", "strategy"]).to_csv(P("11_sensitivity_subsets.csv"), index=False)

# =====================================================================
# 8. PERMUTATION IMPORTANCE (grouped by original column, best config)
# =====================================================================
feat_cols = FEATURE_SETS[MAIN_FS]["cat"] + FEATURE_SETS[MAIN_FS]["num"]
imp = {c: [] for c in feat_cols}
rng = np.random.default_rng(SEED)
for k, (tr, te) in enumerate(skf.split(df, df["y"])):
    f = fit(df.iloc[tr], MAIN_FS, BEST_M, BEST_S, SEED + k)
    test = df.iloc[te].copy()
    base = f1_score(test["y"], predict(f, test)[0], average="macro", labels=CLASSES, zero_division=0)
    for c in feat_cols:
        drops = []
        for _ in range(10):
            t2 = test.copy(); t2[c] = rng.permutation(t2[c].values)
            drops.append(base - f1_score(t2["y"], predict(f, t2)[0], average="macro", labels=CLASSES, zero_division=0))
        imp[c].append(np.mean(drops))
imp_df = pd.DataFrame({"feature": list(imp), "macroF1_drop_mean": [np.mean(v) for v in imp.values()],
                       "macroF1_drop_sd_over_folds": [np.std(v) for v in imp.values()]}
                      ).sort_values("macroF1_drop_mean", ascending=False)
imp_df.to_csv(P("12_permutation_importance.csv"), index=False)
try:
    fig, ax = plt.subplots(figsize=(6, 4))
    d = imp_df.iloc[::-1]
    ax.barh(d["feature"], d["macroF1_drop_mean"], xerr=d["macroF1_drop_sd_over_folds"])
    ax.set_xlabel("Drop in macro-F1 when permuted"); ax.set_title(f"Permutation importance ({BEST_M}+{BEST_S})")
    plt.tight_layout(); plt.savefig(P("12_permutation_importance.png"), dpi=300); plt.close()
except Exception as e:
    log("imp plot skipped:", e)

# =====================================================================
# 9. ENVIRONMENT LOG
# =====================================================================
import sklearn
with open(P("00_environment.txt"), "w") as fh:
    fh.write(f"python {platform.python_version()}\nsklearn {sklearn.__version__}\npandas {pd.__version__}\n"
             f"numpy {np.__version__}\nseed {SEED}\nCV repeats {N_REPEATS}\nSMOTE {HAVE_SMOTE} XGB {HAVE_XGB} LGBM {HAVE_LGBM}\n"
             f"rows {len(df)}\nbest {BEST_M}+{BEST_S}\nmodels {MODELS}\nstrategies {STRATS}\n")
log(f"\nDONE in {time.time()-t0:.0f}s. Files are in ./{args.out}/")
