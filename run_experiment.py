from __future__ import annotations

import argparse
import io
import json
import platform
import sys
import time
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import sklearn
import xgboost
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

TITLE = "Cost-Sensitive AI Decision Intelligence: A Data-Driven Framework for Utility-Optimized Decisions under Class Imbalance"
UCI_ID = 350
UCI_ZIP = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--cost-ratios", nargs="+", type=float, default=[2, 3, 5, 8, 10])
    p.add_argument("--central-ratio", type=float, default=5.0)
    p.add_argument("--risk-aversion", type=float, default=0.25)
    return p.parse_args()


def load_data(cache_dir="data"):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    cache = cache_dir / "uci_default_credit_clean.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        return df[FEATURES], df["target"].astype(int)

    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=UCI_ID)
        X = ds.data.features.copy()
        y = ds.data.targets.iloc[:, 0].copy()
        X.columns = FEATURES
    except Exception:
        r = requests.get(UCI_ZIP, timeout=60)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            xls = [n for n in z.namelist() if n.lower().endswith(".xls")][0]
            with z.open(xls) as f:
                raw = pd.read_excel(f, header=1, engine="xlrd")
        if "ID" in raw.columns:
            raw = raw.drop(columns=["ID"])
        y = raw.iloc[:, -1].astype(int)
        X = raw.iloc[:, :-1].copy()
        X.columns = FEATURES

    X = X.apply(pd.to_numeric, errors="raise")
    y = pd.to_numeric(y, errors="raise").astype(int).rename("target")
    out = X.copy()
    out["target"] = y.to_numpy()
    out.to_csv(cache, index=False)
    return X, y


def split4(X, y, seed):
    Xtr, Xt, ytr, yt = train_test_split(X, y, test_size=0.40, stratify=y, random_state=seed)
    Xcal, Xr, ycal, yr = train_test_split(Xt, yt, test_size=0.625, stratify=yt, random_state=seed + 1)
    Xval, Xte, yval, yte = train_test_split(Xr, yr, test_size=0.60, stratify=yr, random_state=seed + 2)
    return (Xtr, ytr), (Xcal, ycal), (Xval, yval), (Xte, yte)


def make_models(y_train, seed):
    yv = np.asarray(y_train)
    ratio = float((yv == 0).sum() / max((yv == 1).sum(), 1))
    common = dict(
        n_estimators=450, max_depth=4, learning_rate=0.04, min_child_weight=3,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=2.0, reg_alpha=0.05,
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        random_state=seed, n_jobs=-1, verbosity=0,
    )
    return {
        "LogisticRegression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=seed)),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=4, max_features="sqrt",
            n_jobs=-1, random_state=seed,
        ),
        "XGBoost": XGBClassifier(**common),
        "XGBoostWeighted": XGBClassifier(**common, scale_pos_weight=ratio),
    }


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_platt(raw_prob, y, seed):
    model = LogisticRegression(C=1e6, max_iter=1000, random_state=seed)
    model.fit(logit(raw_prob).reshape(-1, 1), np.asarray(y, int))
    return model


def apply_platt(model, raw_prob):
    return model.predict_proba(logit(raw_prob).reshape(-1, 1))[:, 1]


def prob_metrics(y, p):
    p = np.clip(np.asarray(p, float), 1e-8, 1 - 1e-8)
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
    }


def threshold_grid():
    return np.unique(np.r_[np.linspace(0.01, 0.99, 197), 0.5])


def pred(p, t):
    return (np.asarray(p) >= t).astype(int)


def counts(y, yp):
    y = np.asarray(y, int); yp = np.asarray(yp, int)
    return {
        "tn": int(((y == 0) & (yp == 0)).sum()),
        "fp": int(((y == 0) & (yp == 1)).sum()),
        "fn": int(((y == 1) & (yp == 0)).sum()),
        "tp": int(((y == 1) & (yp == 1)).sum()),
    }


def cost(y, p, t, c_fp=1.0, c_fn=5.0):
    c = counts(y, pred(p, t))
    return float(c_fp * c["fp"] + c_fn * c["fn"])


def optimize_f1(y, p):
    best = (0.5, -1.0)
    for t in threshold_grid():
        s = f1_score(y, pred(p, t), zero_division=0)
        if s > best[1]:
            best = (float(t), float(s))
    return best


def optimize_cost(y, p, ratio):
    best = (0.5, float("inf"))
    for t in threshold_grid():
        c = cost(y, p, t, 1.0, ratio) / len(y)
        if c < best[1]:
            best = (float(t), float(c))
    return best


def bayes_threshold(ratio):
    return 1.0 / (1.0 + float(ratio))


def naive_ref_cost(y, ratio):
    y = np.asarray(y, int)
    return min(float((y == 0).sum()), float(ratio * (y == 1).sum()))


def robust_threshold(y, p, ratios, risk_aversion):
    best_t, best_obj, best_mean, best_std = 0.5, float("inf"), None, None
    for t in threshold_grid():
        vals = []
        for r in ratios:
            vals.append(cost(y, p, t, 1.0, r) / max(naive_ref_cost(y, r), 1e-12))
        vals = np.asarray(vals, float)
        obj = vals.mean() + risk_aversion * vals.std(ddof=0)
        if obj < best_obj:
            best_t, best_obj = float(t), float(obj)
            best_mean, best_std = float(vals.mean()), float(vals.std(ddof=0))
    return best_t, {"objective": best_obj, "mean_normalized_cost": best_mean, "std_normalized_cost": best_std}


def class_metrics(y, p, t):
    yp = pred(p, t)
    c = counts(y, yp)
    specificity = c["tn"] / max(c["tn"] + c["fp"], 1)
    return {
        "threshold": t,
        "accuracy": accuracy_score(y, yp),
        "balanced_accuracy": balanced_accuracy_score(y, yp),
        "precision": precision_score(y, yp, zero_division=0),
        "recall": recall_score(y, yp, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y, yp, zero_division=0),
        **c,
    }


def bootstrap_reduction(y, p, t_base, t_new, ratio, n_boot, seed):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, int); p = np.asarray(p, float)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a = cost(y[idx], p[idx], t_base, 1.0, ratio)
        b = cost(y[idx], p[idx], t_new, 1.0, ratio)
        if a > 0:
            vals.append(100 * (a - b) / a)
    v = np.asarray(vals)
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def savefig(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def make_plots(y_all, y_test, raw_test, cal_test, all_test_probs, policy_df, y_val, p_val, chosen, figdir):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    vals = [(np.asarray(y_all) == 0).sum(), (np.asarray(y_all) == 1).sum()]
    ax.bar(["Non-default", "Default"], vals); ax.set_ylabel("Count"); ax.set_title("Class distribution")
    savefig(fig, figdir / "class_distribution.png")

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for name, p in all_test_probs.items():
        precision, recall, _ = precision_recall_curve(y_test, p)
        ax.plot(recall, precision, label=name)
    ax.axhline(float(np.mean(y_test)), ls="--", lw=1, label="Prevalence")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-recall curves"); ax.legend(fontsize=8)
    savefig(fig, figdir / "precision_recall_curves.png")

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for label, p in [("Raw", raw_test), ("Platt calibrated", cal_test)]:
        frac, meanp = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
        ax.plot(meanp, frac, marker="o", label=label)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, label="Perfect")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed rate"); ax.set_title("Calibration"); ax.legend()
    savefig(fig, figdir / "calibration_curve.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    for policy, g in policy_df.groupby("policy"):
        g = g.sort_values("cost_ratio")
        ax.plot(g["cost_ratio"], g["mean_cost"], marker="o", label=policy)
    ax.set_xlabel("C_FN / C_FP"); ax.set_ylabel("Mean decision cost"); ax.set_title("Decision-cost sensitivity"); ax.legend(fontsize=7)
    savefig(fig, figdir / "decision_cost_sensitivity.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    for policy, g in policy_df[policy_df.policy != "Default_0.50"].groupby("policy"):
        g = g.sort_values("cost_ratio")
        ax.plot(g["cost_ratio"], g["reduction_pct"], marker="o", label=policy)
    ax.axhline(0, lw=1); ax.set_xlabel("C_FN / C_FP"); ax.set_ylabel("Cost reduction vs default (%)")
    ax.set_title("Relative utility gain"); ax.legend(fontsize=7)
    savefig(fig, figdir / "cost_reduction_vs_baseline.png")

    ts = threshold_grid(); ratio = chosen["central_ratio"]
    vals = [cost(y_val, p_val, t, 1.0, ratio) / len(y_val) for t in ts]
    fig, ax = plt.subplots(figsize=(6.4, 4.8)); ax.plot(ts, vals)
    for name in ["default_t", "f1_t", "robust_t", "bayes_t", "empirical_t"]:
        ax.axvline(chosen[name], ls="--", lw=1, label=f"{name}: {chosen[name]:.3f}")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Mean validation cost"); ax.set_title(f"Threshold-cost landscape, ratio={ratio:g}")
    ax.legend(fontsize=7); savefig(fig, figdir / "threshold_cost_landscape.png")


def main():
    a = args(); np.random.seed(a.seed); started = time.time()
    out = Path("results"); tables = out / "tables"; figs = out / "figures"
    tables.mkdir(parents=True, exist_ok=True); figs.mkdir(parents=True, exist_ok=True)

    print("=" * 78); print(TITLE); print("=" * 78)
    print("[1/7] Loading UCI dataset...")
    X, y = load_data()
    print({"samples": len(y), "features": X.shape[1], "default_rate": round(float(y.mean()), 4), "missing": int(X.isna().sum().sum())})

    print("[2/7] Four-way stratified split...")
    (Xtr, ytr), (Xcal, ycal), (Xval, yval), (Xte, yte) = split4(X, y, a.seed)
    print({"train": len(ytr), "calibration": len(ycal), "validation": len(yval), "test": len(yte)})

    print("[3/7] Training models...")
    models = make_models(ytr, a.seed)
    store = {}
    model_rows = []
    val_rows = []
    for name, model in models.items():
        print("  ", name)
        model.fit(Xtr, ytr)
        pcal_raw = model.predict_proba(Xcal)[:, 1]
        calibrator = fit_platt(pcal_raw, ycal, a.seed)
        pval_raw = model.predict_proba(Xval)[:, 1]
        pte_raw = model.predict_proba(Xte)[:, 1]
        pval = apply_platt(calibrator, pval_raw)
        pte = apply_platt(calibrator, pte_raw)
        store[name] = {"val": pval, "test": pte, "test_raw": pte_raw}
        row = {"model": name}
        for k, v in prob_metrics(yte, pte_raw).items(): row[f"raw_{k}"] = v
        for k, v in prob_metrics(yte, pte).items(): row[f"calibrated_{k}"] = v
        model_rows.append(row)
        val_rows.append({"model": name, **prob_metrics(yval, pval)})

    model_df = pd.DataFrame(model_rows).sort_values("calibrated_pr_auc", ascending=False)
    val_df = pd.DataFrame(val_rows).sort_values(["pr_auc", "brier"], ascending=[False, True])
    model_df.to_csv(tables / "model_metrics_test.csv", index=False)
    val_df.to_csv(tables / "model_metrics_validation.csv", index=False)
    best = val_df[val_df.model != "XGBoostWeighted"].iloc[0].model
    print("[4/7] Selected base model from validation only:", best)

    pval = store[best]["val"]; pte = store[best]["test"]; pte_raw = store[best]["test_raw"]
    pweight = store["XGBoostWeighted"]["test"]
    f1_t, val_f1 = optimize_f1(yval, pval)
    robust_t, robust_meta = robust_threshold(yval, pval, a.cost_ratios, a.risk_aversion)
    empirical = {r: optimize_cost(yval, pval, r)[0] for r in a.cost_ratios}
    bayes = {r: bayes_threshold(r) for r in a.cost_ratios}
    print("[5/7] Thresholds:", {"f1": f1_t, "robust": robust_t, "robust_objective": robust_meta["objective"]})

    threshold_rows = [
        {"policy": "Default_0.50", "threshold": 0.5},
        {"policy": "F1_Optimized", "threshold": f1_t, "validation_f1": val_f1},
        {"policy": "Robust_MultiScenario", "threshold": robust_t, **robust_meta},
    ]
    for r in a.cost_ratios:
        threshold_rows += [
            {"policy": "Bayes_CostThreshold", "cost_ratio": r, "threshold": bayes[r]},
            {"policy": "Empirical_UtilityOptimized", "cost_ratio": r, "threshold": empirical[r]},
        ]
    pd.DataFrame(threshold_rows).to_csv(tables / "thresholds_validation.csv", index=False)

    print("[6/7] Held-out decision policy evaluation...")
    rows = []
    for r in a.cost_ratios:
        baseline = cost(yte, pte, 0.5, 1.0, r)
        policies = {
            "Default_0.50": (pte, 0.5),
            "F1_Optimized": (pte, f1_t),
            "Bayes_CostThreshold": (pte, bayes[r]),
            "Empirical_UtilityOptimized": (pte, empirical[r]),
            "Robust_MultiScenario": (pte, robust_t),
            "Weighted_XGBoost_0.50": (pweight, 0.5),
        }
        for name, (pp, t) in policies.items():
            cm = class_metrics(yte, pp, t)
            c = cost(yte, pp, t, 1.0, r)
            rows.append({
                "policy": name, "cost_ratio": r, **cm,
                "total_cost": c, "mean_cost": c / len(yte), "utility": -c / len(yte),
                "reduction_pct": 100 * (baseline - c) / max(baseline, 1e-12),
            })
    policy_df = pd.DataFrame(rows)
    policy_df.to_csv(tables / "decision_policy_costs_test.csv", index=False)

    central = float(a.central_ratio)
    if central not in a.cost_ratios:
        raise ValueError("For this one-command version, --central-ratio must be included in --cost-ratios")
    mean_b, lo_b, hi_b = bootstrap_reduction(yte, pte, 0.5, robust_t, central, a.bootstrap, a.seed)
    pd.DataFrame([{
        "comparison": "Robust_MultiScenario vs Default_0.50", "cost_ratio": central,
        "bootstrap_replicates": a.bootstrap, "mean_reduction_pct": mean_b,
        "ci_low_pct": lo_b, "ci_high_pct": hi_b,
    }]).to_csv(tables / "bootstrap_ci.csv", index=False)

    print("[7/7] Generating figures and summary...")
    make_plots(
        y, yte, pte_raw, pte,
        {k: v["test"] for k, v in store.items()},
        policy_df, yval, pval,
        {
            "central_ratio": central, "default_t": 0.5, "f1_t": f1_t, "robust_t": robust_t,
            "bayes_t": bayes[central], "empirical_t": empirical[central],
        }, figs,
    )

    center = policy_df[np.isclose(policy_df.cost_ratio, central)].sort_values("total_cost")
    base = center[center.policy == "Default_0.50"].iloc[0]
    rob = center[center.policy == "Robust_MultiScenario"].iloc[0]
    summary = f"""# Experiment Summary\n\n## Title\n{TITLE}\n\n## Dataset\nUCI Default of Credit Card Clients: {len(y):,} observations, {X.shape[1]} predictors, default prevalence {y.mean():.2%}.\n\n## Design\n60% training, 15% calibration, 10% threshold validation, 15% untouched test. Probabilities are Platt calibrated. Cost ratios are normalized scenario assumptions, not real dollars.\n\n## Selected model\n{best}\n\n## Key thresholds\n- Default: 0.5000\n- F1-optimal: {f1_t:.4f}\n- Robust multi-scenario: {robust_t:.4f}\n\n## Central scenario C_FN/C_FP={central:g}\n- Default total cost: {base.total_cost:.2f}\n- Robust total cost: {rob.total_cost:.2f}\n- Robust reduction: {rob.reduction_pct:.2f}%\n- Paired bootstrap mean reduction: {mean_b:.2f}%\n- 95% CI: [{lo_b:.2f}%, {hi_b:.2f}%]\n- Lowest-cost test policy: {center.iloc[0].policy}\n\n## Interpretation boundary\nTreat costs as normalized decision-loss scenarios unless externally validated monetary costs are added. The testable contribution is whether cost-aware decision policies reduce held-out decision loss consistently under asymmetric costs.\n"""
    (out / "summary.md").write_text(summary)
    metadata = {
        "title": TITLE, "seed": a.seed, "cost_ratios": a.cost_ratios,
        "central_ratio": central, "risk_aversion": a.risk_aversion,
        "bootstrap": a.bootstrap, "python": sys.version,
        "platform": platform.platform(), "numpy": np.__version__,
        "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__, "elapsed_seconds": round(time.time() - started, 3),
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    print("DONE")
    print("Selected model:", best)
    print("Robust threshold:", round(robust_t, 4))
    print("Bootstrap robust-vs-default cost reduction: "
          f"{mean_b:.2f}% [{lo_b:.2f}, {hi_b:.2f}]")
    print("Read results/summary.md and results/tables/decision_policy_costs_test.csv")


if __name__ == "__main__":
    main()
