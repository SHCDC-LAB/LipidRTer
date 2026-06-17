import matplotlib.pyplot as plt

# ============================================================
# Publication-ready figure saving helper
# ============================================================

def _save_fig(fig, output_dir, filename):
    """
    Save matplotlib figure as PNG and PDF for publication.
    """
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{filename}.png"
    pdf_path = output_dir / f"{filename}.pdf"
    try:
        fig.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='white')
        fig.savefig(pdf_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='white')
    except Exception as e:
        import warnings
        warnings.warn(f"Could not save figure {filename}: {e}")
    return {"png": png_path, "pdf": pdf_path}


def set_publication_style():
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 600
    plt.rcParams["axes.linewidth"] = 1.2

def plot_fit_scatter(y_true, y_pred, output_dir=None, filename=None, save=False):
    fig, ax = plt.subplots(figsize=(5.8, 5.4), dpi=300)
    ax.scatter(y_true, y_pred, alpha=0.65, edgecolor="black", linewidth=0.3)
    lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    ax.plot(lims, lims, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Observed RT")
    ax.set_ylabel("Predicted RT")
    ax.set_title("Fine-tuned model fit")
    fig.tight_layout()
    if save and output_dir and filename:
        _save_fig(fig, output_dir, filename)
    return fig

def plot_prediction_hist(y_pred_unknown, output_dir=None, filename=None, save=False):
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    ax.hist(y_pred_unknown, bins=30, edgecolor="black")
    ax.set_xlabel("Predicted RT")
    ax.set_ylabel("Frequency")
    ax.set_title("Unknown sample predicted RT distribution")
    fig.tight_layout()
    if save and output_dir and filename:
        _save_fig(fig, output_dir, filename)
    return fig

def plot_permutation(perm_scores, score, pvalue, output_dir=None, filename=None, save=False):
    fig, ax = plt.subplots(figsize=(6, 4.8), dpi=300)
    ax.hist(perm_scores, bins=min(20, len(perm_scores)), edgecolor="black")
    ax.axvline(score, linestyle="--", linewidth=2)
    ax.set_xlabel("Permutation R² score")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Permutation test, p = {pvalue:.4f}")
    fig.tight_layout()
    if save and output_dir and filename:
        _save_fig(fig, output_dir, filename)
    return fig

def plot_williams(ad_df, h_star, output_dir=None, filename=None, save=False):
    fig, ax = plt.subplots(figsize=(6, 4.8), dpi=300)
    ax.scatter(
        ad_df["Leverage"], ad_df["Standardized_residual"],
        alpha=0.75, edgecolor="black", linewidth=0.4
    )
    ax.axhline(3, linestyle="--", linewidth=1.3)
    ax.axhline(-3, linestyle="--", linewidth=1.3)
    ax.axhline(0, linestyle="-", linewidth=1)
    ax.axvline(h_star, linestyle="--", linewidth=1.3)
    ax.set_xlabel("Leverage")
    ax.set_ylabel("Standardized residuals")
    ax.set_title("Williams plot")
    fig.tight_layout()
    if save and output_dir and filename:
        _save_fig(fig, output_dir, filename)
    return fig

def plot_shap_summary(shap_values, X_shap, output_dir=None, filename=None, save=False):
    import shap
    fig = plt.figure(figsize=(7, 5.5), dpi=300)
    shap.summary_plot(shap_values, X_shap, max_display=min(20, X_shap.shape[1]), show=False)
    plt.tight_layout()
    if save and output_dir and filename:
        _save_fig(fig, output_dir, filename)
    return fig

def plot_shap_bar(shap_values, X_shap, output_dir=None, filename=None, save=False):
    import shap
    fig = plt.figure(figsize=(7, 5.5), dpi=300)
    shap.plots.bar(shap_values, max_display=min(20, X_shap.shape[1]), show=False)
    plt.tight_layout()
    if save and output_dir and filename:
        _save_fig(fig, output_dir, filename)
    return fig
