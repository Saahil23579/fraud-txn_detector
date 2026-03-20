"""
============================================================
  FRAUD TRANSACTION DETECTION — Full ML Pipeline
  Dataset : fraud_dataset_10000_records.csv
  Library : scikit-learn
============================================================
"""

# ── 0. IMPORTS ───────────────────────────────────────────
import warnings
import shap
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Sampling
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)

# Tuning
from sklearn.model_selection import RandomizedSearchCV

# Evaluation
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    ConfusionMatrixDisplay,
)



# ── 1. LOAD DATA ─────────────────────────────────────────
print("\n[1/7] Loading & inspecting data ...")
df = pd.read_csv("data/fraud_dataset_10000_records.csv")  ## check the source path file

print(f"  Shape       : {df.shape}")
print(f"  Missing vals: {df.isnull().sum().sum()}")
print(f"  Fraud rate  : {df['fraud'].mean()*100:.2f}%")
print(df.head(10).to_string())


# ── 2. FEATURE ENGINEERING ───────────────────────────────
print("\n[2/7] Feature engineering ...")

fe = df.copy()

# ── Temporal features
fe["is_night"]    = (fe["hour"] < 6).astype(int)          # 0–5 AM
fe["is_evening"]  = ((fe["hour"] >= 18) & (fe["hour"] < 24)).astype(int)
fe["hour_sin"]    = np.sin(2 * np.pi * fe["hour"] / 24)   # cyclic encoding
fe["hour_cos"]    = np.cos(2 * np.pi * fe["hour"] / 24)

# ── Amount features
fe["amount_log"]    = np.log1p(fe["amount"])
fe["amount_bucket"] = pd.cut(
    fe["amount"],
    bins=[0, 1000, 2000, 3000, 4000, 5000],
    labels=[0, 1, 2, 3, 4],
).astype(int)

# ── High-risk combos (domain rules → binary flags)
fe["mobile_night"]  = ((fe["device"] == "Mobile") & (fe["is_night"] == 1)).astype(int)
fe["wallet_night"]  = ((fe["payment_method"] == "Wallet") & (fe["is_night"] == 1)).astype(int)
fe["high_amt_mobile"] = ((fe["amount"] > 3000) & (fe["device"] == "Mobile")).astype(int)

# ── Encode categoricals
le = LabelEncoder()
for col in ["country", "payment_method", "device"]:
    fe[col + "_enc"] = le.fit_transform(fe[col])

# ── Final feature set
FEATURES = [
    "amount", "amount_log", "amount_bucket",
    "hour", "hour_sin", "hour_cos",
    "is_night", "is_evening",
    "country_enc", "payment_method_enc", "device_enc",
    "mobile_night", "wallet_night", "high_amt_mobile",
]
TARGET = "fraud"

X = fe[FEATURES]
y = fe[TARGET]

print(f"  Feature count : {len(FEATURES)}")
print(f"  Features      : {FEATURES}")


# ── 3. TRAIN / TEST SPLIT ────────────────────────────────
print("\n[3/7] Splitting data (80/20 stratified) ...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Train : {X_train.shape[0]} rows  |  Fraud = {y_train.sum()} ({y_train.mean()*100:.1f}%)")
print(f"  Test  : {X_test.shape[0]} rows   |  Fraud = {y_test.sum()} ({y_test.mean()*100:.1f}%)")

# ── 4. SMOTE — HANDLE CLASS IMBALANCE ───────────────────
print("\n[4/7] Applying SMOTE to training set ...")

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"  Before SMOTE : {dict(pd.Series(y_train).value_counts())}")
print(f"  After  SMOTE : {dict(pd.Series(y_train_res).value_counts())}")


# ── 5. MODEL TRAINING ────────────────────────────────────
print("\n[5/7] Training & evaluating models ...")

scaler = StandardScaler()

# ── 5a. Baseline models for comparison ──────────────────
baseline_models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ]),
    "Decision Tree": Pipeline([
        ("clf", DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42)),
    ]),
    "Random Forest (baseline)": Pipeline([
        ("clf", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)),
    ]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in baseline_models.items():
    auc_scores = cross_val_score(model, X_train_res, y_train_res,
                                 scoring="roc_auc", cv=cv, n_jobs=-1)
    model.fit(X_train_res, y_train_res)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    results[name] = {
        "model":   model,
        "cv_auc":  auc_scores.mean(),
        "test_auc": roc_auc_score(y_test, y_proba),
        "y_pred":  y_pred,
        "y_proba": y_proba,
    }
    print(f"  {name:<35}  CV-AUC={auc_scores.mean():.4f}  Test-AUC={roc_auc_score(y_test, y_proba):.4f}")

# ── 5b. Tuned Random Forest via RandomizedSearch ────────
print("\n  Hyperparameter tuning — Random Forest ...")

rf_param_grid = {
    "n_estimators":      [100, 200, 300],
    "max_depth":         [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":      ["sqrt", "log2"],
    "class_weight":      ["balanced", "balanced_subsample"],
}

rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=rf_param_grid,
    n_iter=30,
    scoring="roc_auc",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=0,
)
rf_search.fit(X_train_res, y_train_res)
best_rf = rf_search.best_estimator_

y_pred_rf  = best_rf.predict(X_test)
y_proba_rf = best_rf.predict_proba(X_test)[:, 1]

results["Random Forest (tuned)"] = {
    "model":    best_rf,
    "cv_auc":   rf_search.best_score_,
    "test_auc": roc_auc_score(y_test, y_proba_rf),
    "y_pred":   y_pred_rf,
    "y_proba":  y_proba_rf,
}
print(f"  {'Random Forest (tuned)':<35}  CV-AUC={rf_search.best_score_:.4f}  Test-AUC={roc_auc_score(y_test, y_proba_rf):.4f}")
print(f"  Best params: {rf_search.best_params_}")

# ── 5c. Gradient Boosting ───────────────────────────────
print("\n  Training Gradient Boosting ...")

gb = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.05,
    max_depth=4, subsample=0.8, random_state=42
)
gb.fit(X_train_res, y_train_res)
y_pred_gb  = gb.predict(X_test)
y_proba_gb = gb.predict_proba(X_test)[:, 1]

results["Gradient Boosting"] = {
    "model":    gb,
    "cv_auc":   cross_val_score(gb, X_train_res, y_train_res, scoring="roc_auc", cv=cv).mean(),
    "test_auc": roc_auc_score(y_test, y_proba_gb),
    "y_pred":   y_pred_gb,
    "y_proba":  y_proba_gb,
}
print(f"  {'Gradient Boosting':<35}  Test-AUC={roc_auc_score(y_test, y_proba_gb):.4f}")


# ── 5d. Voting Ensemble ─────────────────────────────────
print("\n  Training Soft Voting Ensemble ...")

ensemble = VotingClassifier(
    estimators=[
        ("rf",  best_rf),
        ("gb",  gb),
        ("lr", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression( max_iter=3000,
                solver="saga",
                class_weight="balanced",
                random_state=42)),
    ]))
],
    voting="soft",
    n_jobs=-1,
)
ensemble.fit(X_train_res, y_train_res)
y_pred_ens  = ensemble.predict(X_test)
y_proba_ens = ensemble.predict_proba(X_test)[:, 1]

results["Voting Ensemble"] = {
    "model":    ensemble,
    "cv_auc":   cross_val_score(ensemble, X_train_res, y_train_res, scoring="roc_auc", cv=cv).mean(),
    "test_auc": roc_auc_score(y_test, y_proba_ens),
    "y_pred":   y_pred_ens,
    "y_proba":  y_proba_ens,
}
print(f"  {'Voting Ensemble':<35}  Test-AUC={roc_auc_score(y_test, y_proba_ens):.4f}")

# ── 6. EVALUATION & THRESHOLD OPTIMISATION ───────────────
print("\n[6/7] Detailed evaluation of best model ...")

# Pick model with highest test AUC
best_name = max(results, key=lambda k: results[k]["test_auc"])
best      = results[best_name]
print(f"\n  Best model : {best_name}  (Test AUC = {best['test_auc']:.4f})\n")

# ── Classification report at default 0.5 threshold
print("  --- Classification Report (threshold = 0.50) ---")
print(classification_report(y_test, best["y_pred"],
                            target_names=["Legitimate", "Fraud"]))

# ── Find optimal threshold via F1-score on fraud class
precisions, recalls, thresholds = precision_recall_curve(y_test, best["y_proba"])
f1_scores  = 2 * precisions * recalls / (precisions + recalls + 1e-8)
opt_idx    = np.argmax(f1_scores[:-1])
opt_thresh = thresholds[opt_idx]

y_pred_opt = (best["y_proba"] >= opt_thresh).astype(int)

print(f"\n  --- Classification Report (threshold = {opt_thresh:.3f}, F1-optimal) ---")
print(classification_report(y_test, y_pred_opt,
                            target_names=["Legitimate", "Fraud"]))

print(f"  Optimal Threshold : {opt_thresh:.4f}")
print(f"  AUC-ROC           : {best['test_auc']:.4f}")
print(f"  Avg Precision     : {average_precision_score(y_test, best['y_proba']):.4f}")


# ── 7. VISUALISATIONS ────────────────────────────────────
print("\n[7/7] Generating visualisations ...")

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"teal": "#00B4D8", "red": "#E63946", "navy": "#0D1B2A",
          "gold": "#FFB703", "green": "#2DC653", "purple": "#7C3AED"}

# ── Fig 1 — Model Comparison ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Model Comparison", fontsize=15, fontweight="bold")

names  = list(results.keys())
aucs   = [results[n]["test_auc"] for n in names]
colors = [COLORS["teal"] if n != best_name else COLORS["gold"] for n in names]

axes[0].barh(names, aucs, color=colors, edgecolor="white")
axes[0].set_xlim(0.5, 1.0)
axes[0].set_xlabel("Test AUC-ROC")
axes[0].set_title("AUC-ROC by Model")
for i, v in enumerate(aucs):
    axes[0].text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=10)

# PR Curve comparison
for name, res in results.items():
    p, r, _ = precision_recall_curve(y_test, res["y_proba"])
    ap      = average_precision_score(y_test, res["y_proba"])
    axes[1].plot(r, p, label=f"{name} (AP={ap:.3f})",
                 lw=1.8 if name == best_name else 1.2)
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curves")
axes[1].legend(fontsize=8, loc="upper right")
axes[1].axhline(y=y_test.mean(), linestyle="--", color="gray", label="Baseline")

plt.tight_layout()
plt.savefig("1_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved → 1_model_comparison.png")


# ── Fig 2 — Confusion Matrices ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle(f"Confusion Matrix — {best_name}", fontsize=14, fontweight="bold")

for ax, (thresh, preds, title) in zip(axes, [
    (0.50, best["y_pred"],  "Default Threshold (0.50)"),
    (opt_thresh, y_pred_opt, f"Optimal Threshold ({opt_thresh:.3f})"),
]):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Legit", "Fraud"],
                yticklabels=["Legit", "Fraud"],
                linewidths=0.5)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("2_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved → 2_confusion_matrices.png")


# ── Fig 3 — ROC Curves ───────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    ax.plot(fpr, tpr, label=f"{name} (AUC={res['test_auc']:.3f})",
            lw=2 if name == best_name else 1.2)
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.5)")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("3_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved → 3_roc_curves.png")


# ── Fig 4 — Feature Importance ───────────────────────────
if hasattr(best["model"], "feature_importances_"):
    importances = best["model"].feature_importances_
else:
    importances = np.abs(best["model"].named_steps["clf"].feature_importances_) \
                  if hasattr(best["model"], "named_steps") else None

if importances is not None:
    fi_df = pd.DataFrame({"feature": FEATURES, "importance": importances}) \
              .sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors  = [COLORS["gold"] if fi_df["importance"].iloc[i] >= fi_df["importance"].quantile(0.75)
               else COLORS["teal"] for i in range(len(fi_df))]
    ax.barh(fi_df["feature"], fi_df["importance"], color=colors, edgecolor="white")
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title(f"Feature Importance — {best_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("4_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → 4_feature_importance.png")


# ── Fig 5 — Threshold vs Precision / Recall / F1 ─────────
thresholds_range = np.linspace(0.01, 0.99, 200)
prec_list, rec_list, f1_list = [], [], []

for t in thresholds_range:
    y_t = (best["y_proba"] >= t).astype(int)
    tp  = ((y_t == 1) & (y_test == 1)).sum()
    fp  = ((y_t == 1) & (y_test == 0)).sum()
    fn  = ((y_t == 0) & (y_test == 1)).sum()
    p   = tp / (tp + fp + 1e-8)
    r   = tp / (tp + fn + 1e-8)
    f1  = 2 * p * r / (p + r + 1e-8)
    prec_list.append(p); rec_list.append(r); f1_list.append(f1)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(thresholds_range, prec_list, label="Precision", color=COLORS["teal"],  lw=2)
ax.plot(thresholds_range, rec_list,  label="Recall",    color=COLORS["red"],   lw=2)
ax.plot(thresholds_range, f1_list,   label="F1 Score",  color=COLORS["gold"],  lw=2)
ax.axvline(opt_thresh, color="gray", linestyle="--", label=f"Optimal ({opt_thresh:.3f})")
ax.set_xlabel("Decision Threshold"); ax.set_ylabel("Score")
ax.set_title("Threshold Sensitivity Analysis", fontsize=13, fontweight="bold")
ax.legend(); ax.set_xlim(0, 1)
plt.tight_layout()
plt.savefig("5_threshold_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved → 5_threshold_analysis.png")


# ── Fig 6 — SHAP Explainability ──────────────────────────
print("\n  Computing SHAP values (this may take ~30s) ...")

explainer = shap.TreeExplainer(best["model"])
X_sample  = X_test.sample(300, random_state=42)
shap_vals = explainer.shap_values(X_sample)

# For binary classification shap returns list[class0, class1]
sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("SHAP Explainability — Fraud Class", fontsize=14, fontweight="bold")

plt.sca(axes[0])
shap.summary_plot(sv, X_sample, feature_names=FEATURES,
                  plot_type="bar", show=False, color=COLORS["teal"])
axes[0].set_title("Mean |SHAP| — Feature Impact")

plt.sca(axes[1])
shap.summary_plot(sv, X_sample, feature_names=FEATURES, show=False)
axes[1].set_title("SHAP Beeswarm — Direction & Magnitude")

plt.tight_layout()
plt.savefig("6_shap_explainability.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved → 6_shap_explainability.png")


# ── REAL-TIME SCORING FUNCTION ────────────────────────────
def predict_fraud(transaction: dict,
                  model=best["model"],
                  threshold: float = opt_thresh) -> dict:
    """
    Score a single transaction in real time.

    Parameters
    ----------
    transaction : dict with keys:
        country, amount, payment_method, device, hour
    model       : trained sklearn model
    threshold   : decision threshold (default = F1-optimal)

    Returns
    -------
    dict with fraud_probability, decision, risk_tier, explanation
    """
    row = transaction.copy()

    # Feature engineering (mirrors training)
    row["is_night"]       = int(row["hour"] < 6)
    row["is_evening"]     = int(18 <= row["hour"] < 24)
    row["hour_sin"]       = np.sin(2 * np.pi * row["hour"] / 24)
    row["hour_cos"]       = np.cos(2 * np.pi * row["hour"] / 24)
    row["amount_log"]     = np.log1p(row["amount"])
    row["amount_bucket"]  = min(int(row["amount"] // 1000), 4)
    row["mobile_night"]   = int(row["device"] == "Mobile" and row["is_night"])
    row["wallet_night"]   = int(row["payment_method"] == "Wallet" and row["is_night"])
    row["high_amt_mobile"]= int(row["amount"] > 3000 and row["device"] == "Mobile")

    # Encode categoricals with same mapping used in training
    country_map   = {"Germany": 0, "India": 1, "Singapore": 2, "UAE": 3, "UK": 4, "USA": 5}
    pm_map        = {"Credit Card": 0, "Debit Card": 1, "NetBanking": 2, "UPI": 3, "Wallet": 4}
    device_map    = {"Laptop": 0, "Mobile": 1, "Tablet": 2}

    row["country_enc"]        = country_map.get(row["country"], 0)
    row["payment_method_enc"] = pm_map.get(row["payment_method"], 0)
    row["device_enc"]         = device_map.get(row["device"], 0)

    X_new = pd.DataFrame([row])[FEATURES]
    prob  = model.predict_proba(X_new)[0][1]

    if prob >= 0.70:
        risk_tier = "🔴 HIGH RISK"
        decision  = "BLOCK"
    elif prob >= threshold:
        risk_tier = "🟡 MEDIUM RISK"
        decision  = "STEP-UP AUTH"
    else:
        risk_tier = "🟢 LOW RISK"
        decision  = "APPROVE"

    return {
        "fraud_probability": round(float(prob), 4),
        "decision":          decision,
        "risk_tier":         risk_tier,
        "threshold_used":    round(threshold, 4),
    }

# ── DEMO SCORING ─────────────────────────────────────────
print("\n" + "=" * 60)
print("  REAL-TIME SCORING DEMO")
print("=" * 60)

demo_transactions = [
    {"country": "USA",       "amount": 150.0,  "payment_method": "Credit Card", "device": "Laptop", "hour": 14},
    {"country": "Germany",   "amount": 4800.0, "payment_method": "Wallet",      "device": "Mobile", "hour": 2},
    {"country": "India",     "amount": 2500.0, "payment_method": "UPI",         "device": "Tablet", "hour": 10},
    {"country": "UK",        "amount": 4200.0, "payment_method": "Debit Card",  "device": "Mobile", "hour": 3},
    {"country": "Singapore", "amount": 800.0,  "payment_method": "NetBanking",  "device": "Laptop", "hour": 9},
]

for i, txn in enumerate(demo_transactions, 1):
    result = predict_fraud(txn)
    print(f"\n  Transaction {i}: {txn['country']} | ${txn['amount']:,.0f} | "
          f"{txn['payment_method']} | {txn['device']} | Hour={txn['hour']}")
    print(f"  → Fraud Prob = {result['fraud_probability']:.4f}  |  "
          f"{result['risk_tier']}  |  Decision = {result['decision']}")


# ── FINAL SUMMARY ─────────────────────────────────────────
print("\n" + "=" * 60)
print("  PIPELINE SUMMARY")
print("=" * 60)
print(f"  Best Model          : {best_name}")
print(f"  Test AUC-ROC        : {best['test_auc']:.4f}")
print(f"  Avg Precision (AP)  : {average_precision_score(y_test, best['y_proba']):.4f}")
print(f"  Optimal Threshold   : {opt_thresh:.4f}")
print(f"\n  Classification Report @ Optimal Threshold:")
print(classification_report(y_test, y_pred_opt,
                            target_names=["Legitimate", "Fraud"]))
print("  Saved artefacts:")
for f in ["1_model_comparison.png", "2_confusion_matrices.png",
          "3_roc_curves.png", "4_feature_importance.png",
          "5_threshold_analysis.png", "6_shap_explainability.png"]:
    print(f"    • {f}")
print("\n  Done ✓")




# Exposing Model as A Service
import joblib, json
joblib.dump(best["model"], "fraud_model.pkl")
json.dump({"threshold": float(opt_thresh), "best_model_name": best_name, "features": FEATURES},
          open("model_meta.json","w"), indent=2)


