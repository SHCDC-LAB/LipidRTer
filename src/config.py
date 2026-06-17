from pathlib import Path

RANDOM_STATE = 42
DEFAULT_TARGET_COL = "RT"

DEFAULT_FILENAMES = {
    "train_csv": "transforming_Learning.csv",
    "unknown_csv": "unknown_data.csv",
    "mapping_csv": "DP_renamed_column.csv",
    "base_model": "voting_model.joblib",
    "feature_list": "voting_feature_names.joblib",
    "padel_jar": "PaDEL-Descriptor-2.21.jar",
}

def default_search_roots(project_root: str):
    root = Path(project_root)
    return {
        "project_root": root,
        "data": root / "data",
        "model": root / "model",
        "descriptor_mapping": root / "descriptor_mapping",
        "tools": root / "tools",
        "outputs": root / "outputs",
    }
