from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from audio.io import load_wav_scipy
from .feature_pipeline import extract_all_features


def build_feature_pairs_from_joined_df(df: pd.DataFrame, label_prefix: str) -> pd.DataFrame:
    """
    Extract features for matched pairs of single_source_filename and mixed_source_filename.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'single_source_filename' and 'mixed_source_filename'.
    label_prefix : str
        Prefix to assign to columns (e.g., 'heart' or 'lung').

    Returns
    -------
    pd.DataFrame
        Columns like:
            '{label_prefix}_single_rms', '{label_prefix}_mixed_rms', ...
    """
    rows: List[Dict[str, float]] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting features ({label_prefix})"):
        # Single-source
        sr_single, audio_single = load_wav_scipy(row["single_source_filename"])
        single_features = extract_all_features(audio_single, sr_single)

        # Mixed reference
        sr_mixed, audio_mixed = load_wav_scipy(row["mixed_source_filename"])
        mixed_features = extract_all_features(audio_mixed, sr_mixed)

        combined_row: Dict[str, float] = {}
        for key, value in single_features.items():
            combined_row[f"{label_prefix}_single_{key}"] = value
        for key, value in mixed_features.items():
            combined_row[f"{label_prefix}_mixed_{key}"] = value

        rows.append(combined_row)

    return pd.DataFrame(rows)


def build_heart_feature_table(hs_df: pd.DataFrame, mix_hs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a tall table of heart features with metadata and a 'source_group' column.

    hs_df     : heart-only recordings (single source).
    mix_hs_df : mixed recordings, using heart_ref_file as reference heart.
    """
    rows = []

    # Single-source heart
    for _, record in tqdm(hs_df.iterrows(), total=len(hs_df), desc="Heart single"):
        sr, audio = load_wav_scipy(record["filename"])
        features = extract_all_features(audio, sr)
        features.update(
            {
                "gender": record["gender"],
                "location": record["location"],
                "sound_type": record["heart_sound_type"],
                "source_group": "single",
            }
        )
        rows.append(features)

    # Heart references from mixed
    for _, record in tqdm(mix_hs_df.iterrows(), total=len(mix_hs_df), desc="Heart mixed"):
        sr, audio = load_wav_scipy(record["heart_ref_file"])
        features = extract_all_features(audio, sr)
        features.update(
            {
                "gender": record["gender"],
                "location": record["location"],
                "sound_type": record["heart_sound_type"],
                "source_group": "mixed",
            }
        )
        rows.append(features)

    return pd.DataFrame(rows)


def build_lung_feature_table(ls_df: pd.DataFrame, mix_ls_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a tall table of lung features with metadata and a 'source_group' column.

    ls_df     : lung-only recordings (single source).
    mix_ls_df : mixed recordings, using lung_ref_file as reference lung.
    """
    rows = []

    # Single-source lung
    for _, record in tqdm(ls_df.iterrows(), total=len(ls_df), desc="Lung single"):
        sr, audio = load_wav_scipy(record["filename"])
        features = extract_all_features(audio, sr)
        features.update(
            {
                "gender": record["gender"],
                "location": record["location"],
                "sound_type": record["lung_sound_type"],
                "source_group": "single",
            }
        )
        rows.append(features)

    # Lung references from mixed
    for _, record in tqdm(mix_ls_df.iterrows(), total=len(mix_ls_df), desc="Lung mixed"):
        sr, audio = load_wav_scipy(record["lung_ref_file"])
        features = extract_all_features(audio, sr)
        features.update(
            {
                "gender": record["gender"],
                "location": record["location"],
                "sound_type": record["lung_sound_type"],
                "source_group": "mixed",
            }
        )
        rows.append(features)

    return pd.DataFrame(rows)
