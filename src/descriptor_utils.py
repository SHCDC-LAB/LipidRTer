from pathlib import Path
import math
import os
import subprocess
import pandas as pd

from .smiles_utils import require_rdkit

try:
    from rdkit import Chem
except Exception:
    Chem = None

def write_sdf_in_batches(df: pd.DataFrame, output_folder: Path, dataset_name: str, batch_size: int = 50000):
    require_rdkit()
    output_folder.mkdir(parents=True, exist_ok=True)

    n = len(df)
    n_batches = math.ceil(n / batch_size)
    sdf_files = []

    for i in range(n_batches):
        batch_df = df.iloc[i * batch_size: (i + 1) * batch_size]
        sdf_file = output_folder / f"{dataset_name}_batch{i + 1}.sdf"

        writer = Chem.SDWriter(str(sdf_file))
        written_count = 0

        for _, row in batch_df.iterrows():
            mol = row["ROMol"]
            if mol is not None:
                mol.SetProp("_Name", str(row["ID"]))
                writer.write(mol)
                written_count += 1

        writer.close()
        sdf_files.append(sdf_file)

    return sdf_files

def generate_dp_from_batches(sdf_files, output_folder: Path, dataset_name: str, padel_jar: Path, threads: int = 4, progress_callback=None):
    output_folder.mkdir(parents=True, exist_ok=True)
    dp_list = []

    if not padel_jar.exists():
        raise FileNotFoundError(f"PaDEL jar not found: {padel_jar}")

    for i, sdf_file in enumerate(sdf_files):
        dp_file = output_folder / f"{dataset_name}_DP_batch{i + 1}.csv"

        if progress_callback:
            progress_callback(f"Generating PaDEL descriptors for {sdf_file.name}")

        classpath = os.pathsep.join([
            str(padel_jar),
            str(padel_jar.parent / "lib" / "*"),
        ])

        cmd = [
            "java",
            "-cp", classpath,
            "padeldescriptor.PaDELDescriptorApp",
            "-2d",
            "-dir", str(sdf_file),
            "-file", str(dp_file),
            "-threads", str(threads),
            "-retainorder",
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"PaDEL failed for {sdf_file}. Return code: {result.returncode}\n"
                f"Command: {' '.join(cmd)}\n"
                f"Output:\n{result.stdout}"
            )

        if not dp_file.exists():
            raise RuntimeError(f"PaDEL output not created: {dp_file}")

        dp_list.append(pd.read_csv(dp_file))

    if not dp_list:
        raise RuntimeError("No PaDEL descriptor output was generated.")

    dp_merged = pd.concat(dp_list, ignore_index=True)
    dp_merged_file = output_folder / f"{dataset_name}_DP_merged.csv"
    dp_merged.to_csv(dp_merged_file, index=False)
    return dp_merged_file

def apply_fixed_dp_mapping(dp_file: Path, mapping_file: Path, output_file: Path):
    df = pd.read_csv(dp_file)
    mapping = pd.read_csv(mapping_file)

    if "original_column" not in mapping.columns or "renamed_column" not in mapping.columns:
        raise ValueError("Mapping file must contain 'original_column' and 'renamed_column' columns.")

    if "NO" not in df.columns:
        df.insert(0, "NO", [f"F{i + 1:06d}" for i in range(len(df))])

    valid_mask = mapping["original_column"].isin(df.columns)
    rename_dict = dict(
        zip(mapping.loc[valid_mask, "original_column"], mapping.loc[valid_mask, "renamed_column"])
    )
    df.rename(columns=rename_dict, inplace=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    return output_file

def align_features(dp_file: Path, voting_features: list):
    df = pd.read_csv(dp_file)
    missing = [c for c in voting_features if c not in df.columns]
    if missing:
        raise ValueError(
            f"Feature alignment failed. First 20 missing: {missing[:20]} | Total missing: {len(missing)}"
        )
    return df[voting_features].copy()

def process_dataset(
    input_csv: Path,
    output_folder: Path,
    dataset_name: str,
    mapping_file: Path,
    padel_jar: Path,
    n_jobs: int = 1,
    batch_size: int = 50000,
    padel_threads: int = 4,
    precomputed_dp_file: Path | None = None,
    progress_callback=None,
):
    from .smiles_utils import clean_dataframe_smiles

    df = pd.read_csv(input_csv)

    required_cols = {"ID", "SMILES"}
    if dataset_name == "train":
        required_cols.add("RT")

    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"{input_csv} is missing required columns: {sorted(missing_cols)}")

    if progress_callback:
        progress_callback(f"Cleaning and standardizing SMILES for {dataset_name}")

    valid_df, invalid_df, before, after = clean_dataframe_smiles(df, n_jobs=n_jobs)

    if after == 0:
        raise RuntimeError(f"No valid molecules found in {input_csv}")

    cleaned_file = output_folder / f"{dataset_name}_cleaned_valid.csv"
    invalid_file = output_folder / f"{dataset_name}_invalid_smiles.csv"
    valid_df.drop(columns=["ROMol"], errors="ignore").to_csv(cleaned_file, index=False)
    invalid_df.to_csv(invalid_file, index=False)

    if precomputed_dp_file is not None and Path(precomputed_dp_file).exists():
        if progress_callback:
            progress_callback(f"Using precomputed descriptor file for {dataset_name}: {precomputed_dp_file}")
        dp_file_renamed = Path(precomputed_dp_file)
    else:
        if progress_callback:
            progress_callback(f"Writing SDF batches for {dataset_name}")
        sdf_files = write_sdf_in_batches(valid_df, output_folder, dataset_name, batch_size=batch_size)

        dp_file = generate_dp_from_batches(
            sdf_files=sdf_files,
            output_folder=output_folder,
            dataset_name=dataset_name,
            padel_jar=padel_jar,
            threads=padel_threads,
            progress_callback=progress_callback,
        )

        dp_file_renamed = apply_fixed_dp_mapping(
            dp_file=dp_file,
            mapping_file=mapping_file,
            output_file=output_folder / f"{dataset_name}_DP_renamed.csv",
        )

    return dp_file_renamed, valid_df, invalid_df, {"input_rows": before, "valid_rows": after, "invalid_rows": before - after}
