# src/msp_program1.py
# ============================================================
# Program 1: Add ID to MSP library and export CSV for RT update
# Based on MSP_updated.ipynb Program 1.
# ============================================================

from pathlib import Path
import csv
import os

MSP_RT_UPDATE_FIELDS = [
    "ID", "NAME", "PRECURSORMZ", "PRECURSORTYPE", "SMILES", "INCHIKEY",
    "FORMULA", "RETENTIONTIME", "CCS", "IONMODE", "COMPOUNDCLASS"
]


def detect_prefix_from_msp_filename(msp_file_path):
    """
    Detect ID prefix from MSP file name.
    Negative mode -> N000001...
    Positive mode -> P000001...
    """
    filename = os.path.basename(str(msp_file_path)).lower()

    negative_keywords = [
        "neg", "negative", "negative_mode", "neg_mode",
        "esi-", "n-", "_n_", "-n-", "(n)", "minus"
    ]
    positive_keywords = [
        "pos", "positive", "positive_mode", "pos_mode",
        "esi+", "p+", "_p_", "-p-", "(p)", "plus"
    ]

    if any(keyword in filename for keyword in negative_keywords):
        return "N"
    if any(keyword in filename for keyword in positive_keywords):
        return "P"

    raise ValueError(
        "MSP file name must contain positive/negative related words, such as 'pos' or 'neg', "
        "or provide the prefix manually."
    )


def read_msp_file(msp_file_path, encoding="utf-8"):
    """
    Read MSP file and split entries by blank line.
    """
    msp_file_path = Path(msp_file_path)
    try:
        content = msp_file_path.read_text(encoding=encoding)
    except UnicodeDecodeError:
        content = msp_file_path.read_text(encoding="latin-1")

    return content.strip().split("\n\n")


def parse_msp_entries(msp_entries, prefix):
    """
    Parse each MSP entry into a dict and add ID.
    ID format: prefix + six-digit number, e.g. P000001 / N000001.
    """
    compound_info_list = []

    for idx, entry in enumerate(msp_entries):
        lines = entry.strip().split("\n")
        compound_info = {}
        compound_id = f"{prefix}{str(idx + 1).zfill(6)}"

        for line in lines:
            if ":" in line:
                if ": " in line:
                    key, value = line.split(": ", 1)
                else:
                    key, value = line.split(":", 1)
                    value = value.strip()
                compound_info[key] = value
            else:
                values = line.split("\t")
                if len(values) == 2:
                    mass, abundance = values
                    compound_info[mass] = abundance

        compound_info["ID"] = compound_id
        compound_info_list.append(compound_info)

    return compound_info_list


def write_msp_file(compound_info_list, output_file_path):
    """
    Write MSP entries.
    ID is written immediately after NAME.
    Peak lines are written as mass<TAB>abundance.
    """
    output_file_path = Path(output_file_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    updated_msp_entries = []

    for compound_info in compound_info_list:
        updated_lines = []

        for key, value in compound_info.items():
            if key == "NAME":
                updated_lines.append(f"{key}: {value}")
                updated_lines.append(f"ID: {compound_info['ID']}")
            elif key == "Num Peaks":
                updated_lines.append(f"{key}: {value}")
            elif key != "ID":
                if key.isalpha():
                    updated_lines.append(f"{key}: {value}")
                else:
                    updated_lines.append(f"{key}\t{value}")

        updated_msp_entries.append("\n".join(updated_lines))

    output_file_path.write_text("\n\n".join(updated_msp_entries), encoding="utf-8")


def export_csv_for_rt_update(compound_info_list, output_csv_path, fields=None):
    """
    Export selected MSP fields to CSV.
    """
    if fields is None:
        fields = MSP_RT_UPDATE_FIELDS

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with output_csv_path.open("w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for compound_info in compound_info_list:
            row = {field: compound_info.get(field, "") for field in fields}
            writer.writerow(row)


def program1_prepare_msp_for_rt_update(
    msp_file_path,
    output_folder=None,
    prefix=None,
    output_csv_name="msp_for_rt_update.csv",
):
    """
    Wrapper for Streamlit app.

    Outputs:
    1. msp_for_rt_update.csv
    2. <original_stem>_with_ID.msp
    """
    msp_file_path = Path(msp_file_path)

    if output_folder is None:
        output_folder = msp_file_path.parent / "RT_updated"
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if prefix is None or str(prefix).strip() == "":
        prefix = detect_prefix_from_msp_filename(msp_file_path)
    prefix = str(prefix).strip()

    output_csv_path = output_folder / output_csv_name
    output_msp_with_id_path = output_folder / f"{msp_file_path.stem}_with_ID{msp_file_path.suffix}"

    msp_entries = read_msp_file(msp_file_path)
    compound_info_list = parse_msp_entries(msp_entries, prefix)

    write_msp_file(compound_info_list, output_msp_with_id_path)
    export_csv_for_rt_update(compound_info_list, output_csv_path, MSP_RT_UPDATE_FIELDS)

    return {
        "csv_path": output_csv_path,
        "msp_with_id_path": output_msp_with_id_path,
        "entry_count": len(compound_info_list),
        "prefix": prefix,
        "fields": MSP_RT_UPDATE_FIELDS,
    }
