import numpy as np
import pandas as pd

def run_shap(model, X_top_display, voting_features, top_features_renamed, top_features_original, max_samples, background_size, random_state=42):
    import shap

    X_shap = X_top_display.copy()
    if len(X_shap) > max_samples:
        X_shap = X_shap.sample(max_samples, random_state=random_state)

    background = X_shap.sample(min(background_size, len(X_shap)), random_state=random_state)

    def predict_top_display(X):
        X = pd.DataFrame(X, columns=top_features_original)
        X_real = pd.DataFrame(0.0, index=X.index, columns=voting_features)
        for original_col, renamed_col in zip(top_features_original, top_features_renamed):
            X_real[renamed_col] = X[original_col].values
        return model.predict(X_real)

    explainer = shap.Explainer(predict_top_display, background)
    shap_values = explainer(X_shap)

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    shap_importance = pd.DataFrame({
        "Feature": X_shap.columns,
        "Mean_abs_SHAP": mean_abs_shap,
    }).sort_values("Mean_abs_SHAP", ascending=False)

    return X_shap, shap_values, shap_importance
