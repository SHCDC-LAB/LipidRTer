from pathlib import Path
import joblib
import pandas as pd

from .descriptor_utils import process_dataset, align_features
from .evaluation_utils import (
    calculate_basic_metrics, choose_strategy, calculate_q2, run_permutation_test,
    run_feature_importance, restore_display_names, calculate_ad_williams
)
from .shap_utils import run_shap
from .config import RANDOM_STATE

def run_full_pipeline(
    train_csv: Path,
    unknown_csv: Path,
    mapping_csv: Path,
    base_model_file: Path,
    feature_list_file: Path,
    padel_jar: Path,
    output_dir: Path,
    target_col: str = "RT",
    n_jobs: int = 1,
    batch_size: int = 50000,
    padel_threads: int = 4,
    run_permutation: bool = True,
    permutation_mode: str = "fast",
    permutation_n: int | None = None,
    exact_permutation_n_jobs: int = 1,
    run_shap_flag: bool = True,
    run_feature_importance_flag: bool = True,
    precomputed_train_dp: Path | None = None,
    precomputed_unknown_dp: Path | None = None,
    progress_callback=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback("Processing training dataset")
    train_dp_file, train_df, train_invalid, train_stats = process_dataset(
        input_csv=train_csv,
        output_folder=output_dir,
        dataset_name="train",
        mapping_file=mapping_csv,
        padel_jar=padel_jar,
        n_jobs=n_jobs,
        batch_size=batch_size,
        padel_threads=padel_threads,
        precomputed_dp_file=precomputed_train_dp,
        progress_callback=progress_callback,
    )

    if progress_callback:
        progress_callback("Processing unknown dataset")
    unknown_dp_file, unknown_df, unknown_invalid, unknown_stats = process_dataset(
        input_csv=unknown_csv,
        output_folder=output_dir,
        dataset_name="unknown",
        mapping_file=mapping_csv,
        padel_jar=padel_jar,
        n_jobs=n_jobs,
        batch_size=batch_size,
        padel_threads=padel_threads,
        precomputed_dp_file=precomputed_unknown_dp,
        progress_callback=progress_callback,
    )

    if progress_callback:
        progress_callback("Loading base model and aligning features")
    voting_model = joblib.load(base_model_file)
    voting_features = joblib.load(feature_list_file)
    if not isinstance(voting_features, list):
        voting_features = list(voting_features)

    X_train = align_features(train_dp_file, voting_features)
    X_unknown = align_features(unknown_dp_file, voting_features)

    if target_col not in train_df.columns:
        raise ValueError(f"Training data does not contain target column: {target_col}")

    y_train = train_df[target_col].values

    if progress_callback:
        progress_callback("Fine-tuning model")
    voting_model.fit(X_train, y_train)
    finetuned_model_file = output_dir / "voting_model_finetuned.joblib"
    joblib.dump(voting_model, finetuned_model_file)

    if progress_callback:
        progress_callback("Evaluating fine-tuned model")
    y_pred_train = voting_model.predict(X_train)
    metrics = calculate_basic_metrics(y_train, y_pred_train)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / "fine_tuned_model_metrics.csv", index=False)

    n, p = X_train.shape
    strategy = choose_strategy(n, p, RANDOM_STATE)
    q2, y_cv_pred = calculate_q2(voting_model, X_train, y_train, strategy["cv"])
    metrics["Q2"] = q2
    pd.DataFrame([metrics]).to_csv(output_dir / "fine_tuned_model_metrics_with_q2.csv", index=False)

    display_names = restore_display_names(voting_features, mapping_csv)

    perm_result = None
    if run_permutation:
        perm_n = int(permutation_n) if permutation_n is not None else int(strategy["n_permutations"])
        if progress_callback:
            if str(permutation_mode).lower() == "exact":
                progress_callback(f"Running high-precision permutation test ({perm_n} permutations; CV refitting)")
            else:
                progress_callback(f"Running fast permutation test ({perm_n} permutations; fixed model)")
        score, perm_scores, pvalue = run_permutation_test(
            voting_model, X_train, y_train, strategy["cv"], perm_n, RANDOM_STATE,
            mode=permutation_mode, n_jobs=exact_permutation_n_jobs
        )
        perm_result = {
            "score": score,
            "perm_scores": perm_scores,
            "pvalue": pvalue,
            "mode": permutation_mode,
            "n_permutations": perm_n,
        }
        pd.DataFrame({"Permutation_R2": perm_scores}).to_csv(output_dir / "permutation_scores.csv", index=False)
        pd.DataFrame([{
            "mode": permutation_mode,
            "n_permutations": perm_n,
            "score": score,
            "pvalue": pvalue,
        }]).to_csv(output_dir / "permutation_test_summary.csv", index=False)

    if run_feature_importance_flag:
        if progress_callback:
            progress_callback("Calculating adaptive feature importance")
        importance_df = run_feature_importance(
            voting_model, X_train, y_train, voting_features, display_names,
            n_repeats=3 if n < 80 else 5, random_state=RANDOM_STATE
        )
    else:
        importance_df = pd.DataFrame({
            "Feature_renamed": voting_features,
            "Feature_original": display_names,
            "Importance_mean": [0] * len(voting_features),
            "Importance_std": [0] * len(voting_features),
        })

    importance_df.to_csv(output_dir / "permutation_feature_importance.csv", index=False)

    top_n = strategy["top_n_features"]
    top_features_renamed = importance_df.head(top_n)["Feature_renamed"].tolist()
    top_features_original = importance_df.head(top_n)["Feature_original"].tolist()

    X_top = X_train[top_features_renamed].copy()
    X_top_display = X_top.copy()
    X_top_display.columns = top_features_original

    if progress_callback:
        progress_callback("Running Williams plot and applicability domain analysis")
    ad_df, h_star, ad_note = calculate_ad_williams(
        X_top, y_train, y_pred_train, n, strategy["ad_method"], RANDOM_STATE
    )
    ad_df.to_csv(output_dir / "williams_applicability_domain.csv", index=False)

    shap_result = None
    if run_shap_flag:
        if progress_callback:
            progress_callback("Running adaptive SHAP analysis")
        X_shap, shap_values, shap_importance = run_shap(
            voting_model, X_top_display, voting_features,
            top_features_renamed, top_features_original,
            strategy["shap_max_samples"], strategy["shap_background"], RANDOM_STATE
        )
        shap_importance.to_csv(output_dir / "shap_feature_importance.csv", index=False)
        shap_result = {"X_shap": X_shap, "shap_values": shap_values, "shap_importance": shap_importance}

    if progress_callback:
        progress_callback("Predicting unknown samples")
    y_pred_unknown = voting_model.predict(X_unknown)
    pred_df = pd.DataFrame({
        "ID": unknown_df["ID"].values,
        "SMILES": unknown_df["SMILES"].values,
        "Clean_SMILES": unknown_df.get("Clean_SMILES", pd.Series([""] * len(unknown_df))).values,
        "Predicted_RT": y_pred_unknown,
    })
    pred_file = output_dir / "unknown_voting_predictions.csv"
    pred_df.to_csv(pred_file, index=False)

    return {
        "model": voting_model,
        "voting_features": voting_features,
        "X_train": X_train,
        "X_unknown": X_unknown,
        "y_train": y_train,
        "y_pred_train": y_pred_train,
        "y_cv_pred": y_cv_pred,
        "metrics": metrics,
        "strategy": strategy,
        "display_names": display_names,
        "importance_df": importance_df,
        "ad_df": ad_df,
        "h_star": h_star,
        "ad_note": ad_note,
        "perm_result": perm_result,
        "shap_result": shap_result,
        "pred_df": pred_df,
        "train_stats": train_stats,
        "unknown_stats": unknown_stats,
        "train_invalid": train_invalid,
        "unknown_invalid": unknown_invalid,
        "files": {
            "train_dp_file": train_dp_file,
            "unknown_dp_file": unknown_dp_file,
            "finetuned_model_file": finetuned_model_file,
            "pred_file": pred_file,
            "output_dir": output_dir,
        }
    }
