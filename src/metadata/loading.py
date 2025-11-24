from typing import Union
import pandas as pd
import os

ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))

def load_metadata(csv_path: str) -> pd.DataFrame:
    """
    Load a metadata CSV (e.g., HS.csv, LS.csv, Mix.csv).

    Parameters
    ----------
    csv_path : str

    Returns
    -------
    df : pd.DataFrame
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} entries from {csv_path}")
    return df



def load_datasets():
    """
    Load all heart, lung, and mixed datasets from the project directory.

    Parameters
    ----------
    ROOT : str
        Root directory of the project (usually project base path).

    Returns
    -------
    dict
        Dictionary mapping dataset names to pandas DataFrames.
    """

    paths = {
        "heart_data":       os.path.join(ROOT, "data/heart_sounds/HS.csv"),
        "lung_data":        os.path.join(ROOT, "data/lung_sounds/LS.csv"),
        "mix_heart_data":   os.path.join(ROOT, "data/mixed_sounds/heart_ref/Mix_HS.csv"),
        "mix_lung_data":    os.path.join(ROOT, "data/mixed_sounds/lung_ref/Mix_LS.csv"),
        "mixed_data":       os.path.join(ROOT, "data/mixed_sounds/mixed_ref/Mix.csv"),
    }

    datasets = {}

    for name, path in paths.items():
        if os.path.exists(path):
            df = load_metadata(path)

            # prefix file paths
            for col in df.columns:
                if "file" in col.lower() or "filename" in col.lower():
                    df[col] = "../" + df[col].astype(str)

            datasets[name] = df
        else:
            print(f"WARNING: Missing file: {path}")

    return datasets
