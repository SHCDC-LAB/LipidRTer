# src/plotting.py
# ============================================================
# Publication-ready plotting utilities for RT prediction Streamlit pipeline.
# All functions return matplotlib Figure objects for st.pyplot(fig) and
# optionally save publication-ready PNG/PDF files.
# ============================================================

from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# Global publication style
# ============================================================

def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.grid": False,
    })


set_publication_style()


# ============================================================
# Save helper
# ============================================================

def save_publication_figure(fig, filename, output_dir="outputs/figures", dpi=600, formats=("png", "pdf")):
    """Save matplotlib figure as publication-ready PNG/PDF."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_name = str(filename).replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_")
    saved_paths = {}
    for fmt in formats:
        out_path = output_dir / f"{clean_name}.{fmt}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)
        saved_paths[fmt] = out_path
    return saved_paths


def polish_axes(ax):
    """Apply consistent publication formatting to axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", direction="out", length=4, width=1.0, colors="black")
    return ax


def _safe_min_max(a, b=None):
    a = np.asarray(a, dtype=float)
    if b is not None:
        b = np.asarray(b, dtype=float)
        values = np.concatenate([a.ravel(), b.ravel()])
    else:
        values = a.ravel()
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0, 1.0
    return float(np.min(values)), float(np.max(values))


def _identity_limits(y_true, y_pred):
    min_val, max_val = _safe_min_max(y_true, y_pred)
    span = max_val - min_val
    if span <= 0:
        span = 1.0
    padding = span * 0.06
    return min_val - padding, max_val + padding


# ============================================================
# Permutation test plot
# ============================================================

def plot_permutation(perm_scores, score, pvalue, output_dir="outputs/figures", filename="permutation_test", save=True):
    """Plot and optionally save permutation test histogram."""
    set_publication_style()
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    perm_scores = np.asarray(perm_scores, dtype=float)
    perm_scores = perm_scores[np.isfinite(perm_scores)]
    bins = min(20, max(5, len(perm_scores))) if len(perm_scores) else 5
    ax.hist(perm_scores, bins=bins, edgecolor="black", linewidth=0.8, alpha=0.85)
    ax.axvline(score, linestyle="--", linewidth=2, color="black", label="Observed score")
    ax.set_xlabel("Permutation R² score")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Permutation Test (p = {pvalue:.4f})")
    ax.legend(frameon=False)
    polish_axes(ax)
    fig.tight_layout()
    if save:
        save_publication_figure(fig, filename=filename, output_dir=output_dir)
    return fig


# ============================================================
# Williams plot / Applicability Domain
# ============================================================

def plot_williams(ad_df, h_star, output_dir="outputs/figures", filename="williams_plot", save=True):
    """Williams plot for applicability domain analysis."""
    set_publication_style()
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    inside_mask = ad_df["Inside_AD"].astype(bool).values
    ax.scatter(ad_df.loc[inside_mask, "Leverage"], ad_df.loc[inside_mask, "Standardized_residual"],
               s=42, alpha=0.78, edgecolor="black", linewidth=0.4, label="Inside AD")
    ax.scatter(ad_df.loc[~inside_mask, "Leverage"], ad_df.loc[~inside_mask, "Standardized_residual"],
               s=55, alpha=0.9, marker="^", edgecolor="black", linewidth=0.5, label="Outside AD")
    ax.axhline(3, linestyle="--", linewidth=1.4, color="black")
    ax.axhline(-3, linestyle="--", linewidth=1.4, color="black")
    ax.axhline(0, linestyle="-", linewidth=1.0, color="black", alpha=0.8)
    ax.axvline(h_star, linestyle="--", linewidth=1.4, color="black")
    ax.set_xlabel("Leverage")
    ax.set_ylabel("Standardized residuals")
    ax.set_title("Williams Plot")
    ax.legend(frameon=False, loc="best")
    polish_axes(ax)
    fig.tight_layout()
    if save:
        save_publication_figure(fig, filename=filename, output_dir=output_dir)
    return fig


# ============================================================
# Observed vs predicted plot
# ============================================================

def plot_observed_vs_predicted(y_true, y_pred, output_dir="outputs/figures", filename="observed_vs_predicted",
                               title="Observed vs Predicted RT", y_label="Predicted RT", save=True):
    """Scatter plot of observed vs predicted RT."""
    set_publication_style()
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=300)
    ax.scatter(y_true, y_pred, s=42, alpha=0.78, edgecolor="black", linewidth=0.4)
    min_lim, max_lim = _identity_limits(y_true, y_pred)
    ax.plot([min_lim, max_lim], [min_lim, max_lim], linestyle="--", linewidth=1.5, color="black")
    ax.set_xlim(min_lim, max_lim)
    ax.set_ylim(min_lim, max_lim)
    ax.set_xlabel("Observed RT")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    polish_axes(ax)
    fig.tight_layout()
    if save:
        save_publication_figure(fig, filename=filename, output_dir=output_dir)
    return fig


# ============================================================
# Residual plot
# ============================================================

def plot_residuals(y_pred, residuals, output_dir="outputs/figures", filename="residual_plot", title="Residual Plot", save=True):
    """Residuals vs predicted RT plot."""
    set_publication_style()
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    ax.scatter(y_pred, residuals, s=42, alpha=0.78, edgecolor="black", linewidth=0.4)
    ax.axhline(0, linestyle="--", linewidth=1.5, color="black")
    ax.set_xlabel("Predicted RT")
    ax.set_ylabel("Residuals")
    ax.set_title(title)
    polish_axes(ax)
    fig.tight_layout()
    if save:
        save_publication_figure(fig, filename=filename, output_dir=output_dir)
    return fig


# ============================================================
# SHAP summary dot plot
# ============================================================

def plot_shap_summary(shap_values, X_shap, output_dir="outputs/figures", filename="shap_summary_dot", save=True):
    """SHAP summary dot plot."""
    set_publication_style()
    import shap
    plt.close("all")
    fig = plt.figure(figsize=(7, 5.5), dpi=300)
    shap.summary_plot(shap_values, X_shap, max_display=min(20, X_shap.shape[1]), show=False, plot_size=None)
    plt.title("SHAP Summary Plot", fontsize=13)
    plt.tight_layout()
    fig = plt.gcf()
    if save:
        save_publication_figure(fig, filename=filename, output_dir=output_dir)
    return fig


# ============================================================
# SHAP bar plot
# ============================================================

def plot_shap_bar(shap_values, X_shap=None, output_dir="outputs/figures", filename="shap_bar", save=True):
    """SHAP mean absolute value bar plot."""
    set_publication_style()
    import shap
    plt.close("all")
    fig = plt.figure(figsize=(7, 5.5), dpi=300)
    shap.plots.bar(shap_values, max_display=20 if X_shap is None else min(20, X_shap.shape[1]), show=False)
    plt.title("SHAP Feature Importance", fontsize=13)
    plt.tight_layout()
    fig = plt.gcf()
    if save:
        save_publication_figure(fig, filename=filename, output_dir=output_dir)
    return fig


# ============================================================
# Feature importance bar plot
# ============================================================

def plot_feature_importance(importance_df, output_dir="outputs/figures", filename="feature_importance", top_n=20, save=True):
    """Plot top feature importance."""
    set_publication_style()
    df = importance_df.copy().head(top_n)
    feature_col = "Feature_original" if "Feature_original" in df.columns else "Feature"
    df = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    ax.barh(df[feature_col], df["Importance_mean"], edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Permutation importance")
    ax.set_ylabel("")
    ax.set_title(f"Top {min(top_n, len(df))} Feature Importance")
    polish_axes(ax)
    fig.tight_layout()
    if save:
        save_publication_figure(fig, filename=filename, output_dir=output_dir)
    return fig


# ============================================================
# Prediction distribution plot
# ============================================================

def plot_prediction_distribution(y_pred, output_dir="outputs/figures", filename="prediction_distribution", save=True):
    """Distribution of predicted RT for unknown samples."""
    set_publication_style()
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = y_pred[np.isfinite(y_pred)]
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    bins = min(30, max(8, len(y_pred) // 2)) if len(y_pred) else 8
    ax.hist(y_pred, bins=bins, edgecolor="black", linewidth=0.8, alpha=0.85)
    ax.set_xlabel("Predicted RT")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Predicted RT")
    polish_axes(ax)
    fig.tight_layout()
    if save:
        save_publication_figure(fig, filename=filename, output_dir=output_dir)
    return fig


# ============================================================
# Fine-tuned model result plots
# ============================================================

def plot_finetuned_model_results(y_true, y_pred, y_cv_pred=None, output_dir="outputs/figures",
                                 filename_prefix="finetuned_model", save=True):
    """
    Generate and optionally save key figures for the fine-tuned model.

    Figures:
        1. Fine-tuned model observed vs predicted
        2. Fine-tuned model residuals vs predicted
        3. Cross-validated predicted vs observed, optional

    Returns
    -------
    dict
        {"observed_vs_predicted": {"fig": Figure, "paths": {"png": Path, "pdf": Path}}, ...}

    This function does not call plt.show(), so it will not interfere with Streamlit rendering.
    Use st.pyplot(result["..."]["fig"]).
    """
    set_publication_style()
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    output_dir = Path(output_dir)
    results = {}

    fig_obs = plot_observed_vs_predicted(
        y_true=y_true,
        y_pred=y_pred,
        output_dir=output_dir,
        filename=f"{filename_prefix}_observed_vs_predicted",
        title="Fine-tuned Model: Observed vs Predicted RT",
        y_label="Predicted RT",
        save=save,
    )
    results["observed_vs_predicted"] = {
        "fig": fig_obs,
        "paths": {"png": output_dir / f"{filename_prefix}_observed_vs_predicted.png",
                  "pdf": output_dir / f"{filename_prefix}_observed_vs_predicted.pdf"} if save else {},
    }

    residuals = y_true - y_pred
    fig_res = plot_residuals(
        y_pred=y_pred,
        residuals=residuals,
        output_dir=output_dir,
        filename=f"{filename_prefix}_residuals",
        title="Fine-tuned Model: Residuals vs Predicted RT",
        save=save,
    )
    results["residuals"] = {
        "fig": fig_res,
        "paths": {"png": output_dir / f"{filename_prefix}_residuals.png",
                  "pdf": output_dir / f"{filename_prefix}_residuals.pdf"} if save else {},
    }

    if y_cv_pred is not None:
        y_cv_pred = np.asarray(y_cv_pred, dtype=float)
        fig_cv = plot_observed_vs_predicted(
            y_true=y_true,
            y_pred=y_cv_pred,
            output_dir=output_dir,
            filename=f"{filename_prefix}_cv_predicted_vs_observed",
            title="Fine-tuned Model: CV Predicted vs Observed RT",
            y_label="CV Predicted RT",
            save=save,
        )
        results["cv_predicted_vs_observed"] = {
            "fig": fig_cv,
            "paths": {"png": output_dir / f"{filename_prefix}_cv_predicted_vs_observed.png",
                      "pdf": output_dir / f"{filename_prefix}_cv_predicted_vs_observed.pdf"} if save else {},
        }

    return results


# ============================================================
# Utility for Streamlit download buttons
# ============================================================

def get_saved_figure_paths(output_dir="outputs/figures"):
    """Return all saved figure paths in output directory."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []
    return sorted(list(output_dir.glob("*.png")) + list(output_dir.glob("*.pdf")))
