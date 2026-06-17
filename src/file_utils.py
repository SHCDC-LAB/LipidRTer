from pathlib import Path
import shutil
import pandas as pd

def find_file(root: Path, filename: str):
    if not root.exists():
        return None
    try:
        for p in root.rglob(filename):
            if p.is_file():
                return p
    except Exception:
        return None
    return None

def save_uploaded_file(uploaded_file, save_dir: Path, filename: str):
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / filename
    with open(out, "wb") as f:
        shutil.copyfileobj(uploaded_file, f)
    return out

def read_csv_preview(path: Path, n=5):
    return pd.read_csv(path).head(n)

def validate_columns(path: Path, required_cols):
    df = pd.read_csv(path, nrows=5)
    missing = sorted(set(required_cols) - set(df.columns))
    return missing
