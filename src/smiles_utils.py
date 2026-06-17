from typing import Iterable, Optional, Tuple
import pandas as pd
from joblib import Parallel, delayed

try:
    from rdkit import Chem
    from rdkit.Chem import SaltRemover
    from rdkit.Chem.MolStandardize import rdMolStandardize
except Exception as e:
    Chem = None
    SaltRemover = None
    rdMolStandardize = None
    RDKIT_IMPORT_ERROR = e
else:
    RDKIT_IMPORT_ERROR = None

def require_rdkit():
    if Chem is None:
        raise ImportError(
            "RDKit is not available. Please install rdkit before running SMILES processing. "
            f"Original error: {RDKIT_IMPORT_ERROR}"
        )

def clean_and_standardize_smiles(smiles: object) -> Tuple[Optional[str], str]:
    require_rdkit()

    if pd.isna(smiles) or str(smiles).strip() == "":
        return None, "Empty or NaN"

    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None, "Invalid SMILES"

        remover = SaltRemover.SaltRemover()
        mol = remover.StripMol(mol, dontRemoveEverything=True)

        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)

        normalizer = rdMolStandardize.Normalizer()
        mol = normalizer.normalize(mol)

        std_smiles = Chem.MolToSmiles(mol, canonical=True)
        return std_smiles, "OK"

    except Exception as e:
        return None, f"Error: {e}"

def mol_from_smiles_parallel(smiles_list: Iterable[object], n_jobs: int = 1):
    require_rdkit()

    def safe_mol(smiles: object):
        std_smiles, status = clean_and_standardize_smiles(smiles)
        if std_smiles is None:
            return None
        return Chem.MolFromSmiles(std_smiles)

    return Parallel(n_jobs=n_jobs)(
        delayed(safe_mol)(s) for s in smiles_list
    )

def clean_dataframe_smiles(df: pd.DataFrame, n_jobs: int = 1):
    require_rdkit()

    status = []
    clean = []
    for s in df["SMILES"]:
        std, msg = clean_and_standardize_smiles(s)
        clean.append(std)
        status.append(msg)

    out = df.copy()
    out["Clean_SMILES"] = clean
    out["SMILES_Status"] = status
    out["ROMol"] = mol_from_smiles_parallel(out["SMILES"], n_jobs=n_jobs)

    before = len(out)
    valid = out[out["ROMol"].notna()].reset_index(drop=True)
    invalid = out[out["ROMol"].isna()].drop(columns=["ROMol"], errors="ignore").reset_index(drop=True)

    return valid, invalid, before, len(valid)
