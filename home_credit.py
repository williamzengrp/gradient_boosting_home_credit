import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


DATA_DIR   = "data"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_FOLDS       = 5
RANDOM_STATE  = 42
OPTUNA_TRIALS = 50
TARGET        = "TARGET"


def load_main_table() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "application_train.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot find {path}.\n"
            "Download 'application_train.csv' from:\n"
            "  https://www.kaggle.com/c/home-credit-default-risk/data\n"
            f"and place it in the '{DATA_DIR}/' folder."
        )
    df = pd.read_csv(path)
    print(f"[1] Loaded application_train.csv: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def load_supplementary_tables() -> dict[str, pd.DataFrame]:
    tables = {}
    names = ["bureau", "bureau_balance", "previous_application",
             "installments_payments", "credit_card_balance", "POS_CASH_balance"]
    for name in names:
        fpath = os.path.join(DATA_DIR, f"{name}.csv")
        if os.path.exists(fpath):
            tables[name] = pd.read_csv(fpath)
            print(f"    Loaded {name}.csv: {tables[name].shape}")
    return tables


def run_eda(df: pd.DataFrame) -> None:
    print("\n[2] EDA")


    target_counts = df[TARGET].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    target_counts.plot.bar(ax=axes[0], color=["steelblue","tomato"], edgecolor="k")
    axes[0].set_title("Target Distribution")
    axes[0].set_xticklabels(["No Default (0)", "Default (1)"], rotation=0)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():,}", (p.get_x() + 0.15, p.get_height() + 100))

    imbalance_pct = target_counts[1] / len(df) * 100
    print(f"    Default rate: {imbalance_pct:.2f}%  ({target_counts[1]:,} / {len(df):,})")


    missing = df.isnull().mean().sort_values(ascending=False).head(30)
    missing[missing > 0].plot.barh(ax=axes[1], color="salmon")
    axes[1].set_title("Top Features by Missing Rate")
    axes[1].set_xlabel("Missing Fraction")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_target_missing.png"), dpi=150)
    plt.close()
    print(f"    Features with > 50% missing: "
          f"{(df.isnull().mean() > 0.5).sum()}")


    num_cols = df.select_dtypes(include=np.number).drop(columns=[TARGET, "SK_ID_CURR"],
                                                         errors="ignore")
    top20 = num_cols.var().nlargest(20).index
    fig, ax = plt.subplots(figsize=(14, 11))
    corr = num_cols[top20].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", cmap="coolwarm",
                linewidths=0.3, ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlation Heatmap — Top 20 Numeric Features (by variance)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_correlation_heatmap.png"), dpi=150)
    plt.close()
    print("    EDA plots saved.")


def engineer_bureau_features(bureau: pd.DataFrame) -> pd.DataFrame:
    agg = bureau.groupby("SK_ID_CURR").agg(
        bureau_count         = ("SK_ID_BUREAU", "count"),
        bureau_active_count  = ("CREDIT_ACTIVE", lambda x: (x == "Active").sum()),
        bureau_credit_sum    = ("AMT_CREDIT_SUM", "sum"),
        bureau_debt_sum      = ("AMT_CREDIT_SUM_DEBT", "sum"),
        bureau_overdue_sum   = ("AMT_CREDIT_SUM_OVERDUE", "sum"),
        bureau_days_overdue  = ("CREDIT_DAY_OVERDUE", "max"),
    ).reset_index()
    agg["bureau_debt_ratio"] = agg["bureau_debt_sum"] / (agg["bureau_credit_sum"] + 1)
    return agg


def engineer_prev_app_features(prev: pd.DataFrame) -> pd.DataFrame:
    agg = prev.groupby("SK_ID_CURR").agg(
        prev_app_count       = ("SK_ID_PREV", "count"),
        prev_approved_count  = ("NAME_CONTRACT_STATUS",
                                lambda x: (x == "Approved").sum()),
        prev_refused_count   = ("NAME_CONTRACT_STATUS",
                                lambda x: (x == "Refused").sum()),
        prev_amt_credit_mean = ("AMT_CREDIT", "mean"),
        prev_days_decision_mean = ("DAYS_DECISION", "mean"),
    ).reset_index()
    agg["prev_approval_rate"] = (agg["prev_approved_count"] /
                                 agg["prev_app_count"].clip(lower=1))
    return agg


KNOWN_LEAKAGE_COLS = []

def preprocess(df: pd.DataFrame, tables: dict = None) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()


    if tables:
        if "bureau" in tables:
            bureau_feats = engineer_bureau_features(tables["bureau"])
            df = df.merge(bureau_feats, on="SK_ID_CURR", how="left")
            print(f"    Merged bureau features. Shape: {df.shape}")
        if "previous_application" in tables:
            prev_feats = engineer_prev_app_features(tables["previous_application"])
            df = df.merge(prev_feats, on="SK_ID_CURR", how="left")
            print(f"    Merged prev_app features. Shape: {df.shape}")


    drop_cols = ["SK_ID_CURR", TARGET] + KNOWN_LEAKAGE_COLS
    y = df[TARGET]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])


    cat_cols = X.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        n_unique = X[col].nunique()
        if n_unique <= 2:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        else:
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True, dtype=np.int8)
            X = pd.concat([X.drop(columns=[col]), dummies], axis=1)

    print(f"    Preprocessed shape: X={X.shape}, y distribution: "
          f"{y.value_counts().to_dict()}")
    return X, y


def build_logreg_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   LogisticRegression(max_iter=500, random_state=RANDOM_STATE,
                                       class_weight="balanced", C=0.1)),
    ])


def cv_auc(model, X: pd.DataFrame, y: pd.Series,
           n_folds: int = N_FOLDS) -> tuple[float, float]:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    return scores.mean(), scores.std()


def train_lgbm_cv(X: pd.DataFrame, y: pd.Series,
                  params: dict = None) -> tuple[lgb.Booster, list[float]]:
    default_params = {
        "objective":       "binary",
        "metric":          "auc",
        "learning_rate":   0.05,
        "num_leaves":      63,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":    5,
        "reg_alpha":       0.1,
        "reg_lambda":      0.1,
        "class_weight":    "balanced",
        "verbosity":       -1,
        "random_state":    RANDOM_STATE,
    }
    if params:
        default_params.update(params)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(y))
    fold_aucs  = []
    final_model = None

    print(f"\n[LightGBM] {N_FOLDS}-fold CV …")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            default_params, dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        preds = model.predict(X_val)
        oof_preds[val_idx] = preds
        auc = roc_auc_score(y_val, preds)
        fold_aucs.append(auc)
        print(f"  Fold {fold+1}: AUC = {auc:.4f}")
        final_model = model

    oof_auc = roc_auc_score(y, oof_preds)
    print(f"  OOF AUC: {oof_auc:.4f}  (mean folds: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f})")
    return final_model, fold_aucs, oof_preds


def tune_lgbm(X: pd.DataFrame, y: pd.Series) -> dict:
    if not OPTUNA_AVAILABLE:
        print("  Optuna not installed — skipping tuning.")
        return {}

    print(f"\n[Optuna] Tuning LightGBM for {OPTUNA_TRIALS} trials …")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            "objective":        "binary",
            "metric":           "auc",
            "verbosity":        -1,
            "learning_rate":    trial.suggest_float("learning_rate",  0.01, 0.1,  log=True),
            "num_leaves":       trial.suggest_int("num_leaves",       16,   256),
            "min_child_samples":trial.suggest_int("min_child_samples",5,    100),
            "feature_fraction": trial.suggest_float("feature_fraction",0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction",0.4, 1.0),
            "bagging_freq":     trial.suggest_int("bagging_freq",     1,    10),
            "reg_alpha":        trial.suggest_float("reg_alpha",      1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda",     1e-4, 10.0, log=True),
            "random_state":     RANDOM_STATE,
        }
        aucs = []
        for tr_idx, val_idx in skf.split(X, y):
            dtrain = lgb.Dataset(X.iloc[tr_idx], label=y.iloc[tr_idx])
            dval   = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx])
            m = lgb.train(params, dtrain, num_boost_round=300,
                          valid_sets=[dval],
                          callbacks=[lgb.early_stopping(30, verbose=False),
                                     lgb.log_evaluation(-1)])
            aucs.append(roc_auc_score(y.iloc[val_idx], m.predict(X.iloc[val_idx])))
        return np.mean(aucs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    print(f"  Best AUC: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params


def compare_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    print("\n[Model Comparison] 5-fold CV AUC …")
    rows = []


    lr = build_logreg_pipeline()
    mu, sd = cv_auc(lr, X, y)
    print(f"  Logistic Regression:  {mu:.4f} ± {sd:.4f}")
    rows.append({"model": "Logistic Regression", "cv_auc_mean": mu, "cv_auc_std": sd})


    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   RandomForestClassifier(n_estimators=100, max_depth=8,
                                           class_weight="balanced",
                                           random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    mu, sd = cv_auc(rf, X, y)
    print(f"  Random Forest:        {mu:.4f} ± {sd:.4f}")
    rows.append({"model": "Random Forest", "cv_auc_mean": mu, "cv_auc_std": sd})


    lgbm_clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                   class_weight="balanced",
                                   verbosity=-1, random_state=RANDOM_STATE)
    pipe_lgbm = Pipeline([("imputer", SimpleImputer(strategy="median")),
                           ("model",  lgbm_clf)])
    mu, sd = cv_auc(pipe_lgbm, X, y)
    print(f"  LightGBM:             {mu:.4f} ± {sd:.4f}")
    rows.append({"model": "LightGBM", "cv_auc_mean": mu, "cv_auc_std": sd})


    xgb_clf = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05,
                                  max_depth=6, use_label_encoder=False,
                                  eval_metric="auc", verbosity=0,
                                  random_state=RANDOM_STATE, n_jobs=-1)
    pipe_xgb = Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("model",  xgb_clf)])
    mu, sd = cv_auc(pipe_xgb, X, y)
    print(f"  XGBoost:              {mu:.4f} ± {sd:.4f}")
    rows.append({"model": "XGBoost", "cv_auc_mean": mu, "cv_auc_std": sd})

    if CATBOOST_AVAILABLE:
        cat_clf = cb.CatBoostClassifier(iterations=300, learning_rate=0.05,
                                         depth=6, verbose=0,
                                         random_state=RANDOM_STATE,
                                         auto_class_weights="Balanced")
        pipe_cat = Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("model",  cat_clf)])
        mu, sd = cv_auc(pipe_cat, X, y)
        print(f"  CatBoost:             {mu:.4f} ± {sd:.4f}")
        rows.append({"model": "CatBoost", "cv_auc_mean": mu, "cv_auc_std": sd})

    return pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False)


def plot_roc_pr(y_true: np.ndarray, y_prob: np.ndarray, label: str, fname: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    axes[0].plot(fpr, tpr, label=f"AUC = {auc_val:.4f}", color="steelblue")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.7)
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].set_title(f"ROC Curve — {label}")
    axes[0].legend()

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    axes[1].plot(rec, prec, color="darkorange")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall Curve — {label}")

    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           label: str, fname: str, threshold: float = 0.5):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Default","Default"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {label} (threshold={threshold})")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


def plot_feature_importance(model: lgb.Booster, feature_names: list[str], fname: str,
                             top_n: int = 30):
    imp = pd.Series(model.feature_importance(importance_type="gain"),
                    index=feature_names).nlargest(top_n)
    fig, ax = plt.subplots(figsize=(9, 8))
    imp[::-1].plot.barh(ax=ax, color="steelblue", edgecolor="k", linewidth=0.3)
    ax.set_title(f"LightGBM Feature Importance (top {top_n}, gain)")
    ax.set_xlabel("Gain")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


def plot_shap(model: lgb.Booster, X_sample: pd.DataFrame, fname: str):
    if not SHAP_AVAILABLE:
        print("  SHAP not installed — skipping SHAP plot.")
        return
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    vals = shap_values[1] if isinstance(shap_values, list) else shap_values
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(vals, X_sample, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved SHAP plot: {fname}")


def plot_model_comparison(comp_df: pd.DataFrame, fname: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = sns.color_palette("husl", len(comp_df))
    bars = ax.barh(comp_df["model"], comp_df["cv_auc_mean"],
                   xerr=comp_df["cv_auc_std"], color=colors,
                   edgecolor="k", linewidth=0.5, capsize=4)
    ax.set_xlabel("5-Fold CV AUC")
    ax.set_title("Model Comparison — Cross-Validation AUC")
    ax.set_xlim(0.5, 1.0)
    for bar, (_, row) in zip(bars, comp_df.iterrows()):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{row['cv_auc_mean']:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


def make_submission(model: lgb.Booster, fname: str = "submission.csv"):
    test_path = os.path.join(DATA_DIR, "application_test.csv")
    if not os.path.exists(test_path):
        print("  application_test.csv not found — skipping submission file.")
        return
    test_df   = pd.read_csv(test_path)
    sk_ids    = test_df["SK_ID_CURR"]
    test_X, _ = preprocess(test_df.assign(**{TARGET: 0}))

    train_cols = model.feature_name()
    test_X = test_X.reindex(columns=train_cols, fill_value=0)
    preds = model.predict(test_X)
    submission = pd.DataFrame({"SK_ID_CURR": sk_ids, "TARGET": preds})
    out_path = os.path.join(OUTPUT_DIR, fname)
    submission.to_csv(out_path, index=False)
    print(f"  Kaggle submission saved: {out_path}")


def main():
    print("=" * 60)
    print("  HOME CREDIT DEFAULT RISK — GRADIENT BOOSTING PIPELINE")
    print("=" * 60)


    df = load_main_table()
    tables = load_supplementary_tables()


    run_eda(df)


    print("\n[3] Preprocessing …")
    X, y = preprocess(df, tables=tables if tables else None)


    X_imp = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X),
        columns=X.columns
    )


    comp_df = compare_models(X_imp, y)
    print("\n  Model Comparison Table:")
    print(comp_df.to_string(index=False))
    comp_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
    plot_model_comparison(comp_df, os.path.join(OUTPUT_DIR, "model_comparison.png"))


    best_params = tune_lgbm(X_imp, y) if OPTUNA_AVAILABLE else {}


    print("\n[LightGBM] Full cross-validation with best params …")
    lgbm_model, fold_aucs, oof_preds = train_lgbm_cv(X_imp, y, params=best_params)


    threshold = 0.5
    y_pred    = (oof_preds >= threshold).astype(int)


    plot_roc_pr(y.values, oof_preds, "LightGBM OOF",
                os.path.join(OUTPUT_DIR, "roc_pr_lgbm.png"))
    plot_confusion_matrix(y.values, y_pred, "LightGBM OOF",
                          os.path.join(OUTPUT_DIR, "confusion_matrix.png"), threshold)
    plot_feature_importance(lgbm_model, X_imp.columns.tolist(),
                            os.path.join(OUTPUT_DIR, "feature_importance.png"))


    print("\n  Classification Report (OOF, threshold=0.5):")
    print(classification_report(y, y_pred, target_names=["No Default", "Default"]))


    sample_size = min(2000, len(X_imp))
    X_sample    = X_imp.sample(sample_size, random_state=RANDOM_STATE)
    plot_shap(lgbm_model, X_sample, os.path.join(OUTPUT_DIR, "shap_summary.png"))


    make_submission(lgbm_model)


    oof_auc = roc_auc_score(y, oof_preds)
    print(f"\n{'='*60}")
    print(f"  Final LightGBM OOF AUC : {oof_auc:.4f}")
    print(f"  Fold AUCs              : {[round(a,4) for a in fold_aucs]}")
    print(f"  All outputs saved to   : {OUTPUT_DIR}/")
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
