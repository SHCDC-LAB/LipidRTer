import os
import contextlib
from math import sqrt
import numpy as np
import pandas as pd
from scipy.stats import zscore

from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict, permutation_test_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def calculate_basic_metrics(y_true, y_pred, relative_mode="scale_by_max"):
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    if relative_mode == "scale_by_max":
        denom = np.max(np.abs(y_true)) or 1.0
        mre = mae / denom
        medre = medae / denom
    else:
        eps = max(1e-8, 1e-6 * np.nanmedian(np.abs(y_true)))
        rel = np.abs(y_true - y_pred) / np.clip(np.abs(y_true), eps, None)
        mre = np.mean(rel)
        medre = np.median(rel)

    return {"R2": r2, "RMSE": rmse, "MedAE": medae, "MAE": mae, "MRE": mre, "MedRE": medre}

def choose_strategy(n, p, random_state=42):
    if n < 30:
        return {
            "cv": LeaveOneOut(), "cv_name": "Leave-one-out CV",
            "n_permutations": 10, "top_n_features": min(8, p, max(3, n // 4)),
            "shap_max_samples": min(n, 30), "shap_background": min(n, 20),
            "ad_method": "top_features",
        }
    if n < 80:
        return {
            "cv": KFold(n_splits=5, shuffle=True, random_state=random_state), "cv_name": "5-fold CV",
            "n_permutations": 20, "top_n_features": min(15, p, max(5, n // 3)),
            "shap_max_samples": min(n, 60), "shap_background": min(n, 30),
            "ad_method": "top_features",
        }
    if n < 200:
        return {
            "cv": KFold(n_splits=5, shuffle=True, random_state=random_state), "cv_name": "5-fold CV",
            "n_permutations": 50, "top_n_features": min(25, p, max(10, n // 4)),
            "shap_max_samples": min(n, 100), "shap_background": min(n, 50),
            "ad_method": "top_features",
        }
    return {
        "cv": KFold(n_splits=5, shuffle=True, random_state=random_state), "cv_name": "5-fold CV",
        "n_permutations": 100, "top_n_features": min(40, p, max(15, n // 5)),
        "shap_max_samples": min(n, 200), "shap_background": min(n, 80),
        "ad_method": "pca",
    }

def calculate_q2(model, X, y, cv):
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        y_cv_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
    return r2_score(y, y_cv_pred), y_cv_pred

def run_fast_permutation_test(model, X, y, n_permutations=100, random_state=42):
    """
    Fast fixed-model permutation test.

    This mode does NOT refit the model. It keeps the fitted predictions fixed and
    permutes y to build a quick null distribution. It is suitable for interactive
    Streamlit use and screening, but the exact CV refitting mode is more rigorous
    for final reporting.
    """
    rng = np.random.default_rng(random_state)
    y_pred = model.predict(X)
    score = r2_score(y, y_pred)
    perm_scores = np.empty(int(n_permutations), dtype=float)

    for i in range(int(n_permutations)):
        y_perm = rng.permutation(y)
        perm_scores[i] = r2_score(y_perm, y_pred)

    pvalue = (np.sum(perm_scores >= score) + 1) / (len(perm_scores) + 1)
    return score, perm_scores, pvalue

def run_exact_permutation_test(model, X, y, cv, n_permutations, random_state=42, n_jobs=1):
    """Exact sklearn permutation_test_score with CV refitting."""
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        score, perm_scores, pvalue = permutation_test_score(
            model, X, y, scoring="r2", cv=cv,
            n_permutations=int(n_permutations), random_state=random_state, n_jobs=n_jobs
        )
    return score, perm_scores, pvalue

def run_permutation_test(
    model,
    X,
    y,
    cv,
    n_permutations,
    random_state=42,
    mode="fast",
    n_jobs=1,
):
    """Unified permutation test interface.

    mode='fast': fixed fitted model, no CV refit, fast and interactive.
    mode='exact': sklearn permutation_test_score, CV refit each permutation.
    """
    mode = str(mode).lower()
    if mode in {"fast", "quick", "fixed"}:
        return run_fast_permutation_test(model, X, y, n_permutations, random_state)
    if mode in {"exact", "high_precision", "high-precision", "cv"}:
        return run_exact_permutation_test(model, X, y, cv, n_permutations, random_state, n_jobs=n_jobs)
    raise ValueError(f"Unknown permutation mode: {mode}")

def run_feature_importance(model, X, y, voting_features, display_names, n_repeats=3, random_state=42):
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        perm_imp = permutation_importance(
            model, X, y, scoring="r2", n_repeats=n_repeats,
            random_state=random_state, n_jobs=-1
        )
    return pd.DataFrame({
        "Feature_renamed": voting_features,
        "Feature_original": display_names,
        "Importance_mean": perm_imp.importances_mean,
        "Importance_std": perm_imp.importances_std,
    }).sort_values("Importance_mean", ascending=False)

def restore_display_names(voting_features, mapping_file):
    mapping = pd.read_csv(mapping_file)
    rename_back = dict(zip(mapping["renamed_column"], mapping["original_column"]))
    return [rename_back.get(c, c) for c in voting_features]

def calculate_ad_williams(X_top, y, y_pred, n, ad_method="top_features", random_state=42):
    X_ad = X_top.copy()
    X_ad = X_ad.replace([np.inf, -np.inf], np.nan)
    X_ad = X_ad.fillna(X_ad.median())

    nonzero_var_cols = X_ad.columns[X_ad.var(axis=0) > 1e-12]
    X_ad = X_ad[nonzero_var_cols]

    if X_ad.shape[1] < 2:
        raise ValueError("AD analysis failed: fewer than 2 valid non-zero variance features.")

    X_scaled = StandardScaler().fit_transform(X_ad)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    if ad_method == "pca" or X_scaled.shape[1] >= n:
        max_components = min(n - 2, X_scaled.shape[1], 40)
        max_components = max(2, max_components)
        pca = PCA(n_components=max_components, random_state=random_state)
        X_ad_space = pca.fit_transform(X_scaled)
        ad_note = f"PCA space, components={max_components}, explained variance={pca.explained_variance_ratio_.sum():.3f}"
    else:
        X_ad_space = X_scaled
        ad_note = f"Top-feature space, features={X_ad_space.shape[1]}"

    X_design = np.column_stack([np.ones(X_ad_space.shape[0]), X_ad_space])

    try:
        H = X_design @ np.linalg.pinv(X_design.T @ X_design, rcond=1e-6) @ X_design.T
        leverage = np.diag(H)
    except np.linalg.LinAlgError:
        XtX = X_design.T @ X_design
        ridge = 1e-6 * np.eye(XtX.shape[0])
        H = X_design @ np.linalg.inv(XtX + ridge) @ X_design.T
        leverage = np.diag(H)

    residuals = y - y_pred
    std_residuals = zscore(residuals)
    std_residuals = np.nan_to_num(std_residuals, nan=0.0)

    k = X_design.shape[1] - 1
    h_star = 3 * (k + 1) / n
    inside_ad = (leverage < h_star) & (np.abs(std_residuals) < 3)

    ad_df = pd.DataFrame({
        "RT_observed": y, "RT_predicted": y_pred, "Residual": residuals,
        "Standardized_residual": std_residuals, "Leverage": leverage, "Inside_AD": inside_ad,
    })
    return ad_df, h_star, ad_note
