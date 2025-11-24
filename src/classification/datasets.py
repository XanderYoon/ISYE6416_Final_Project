from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict

from .features import extract_feature_matrix


@dataclass
class DesignMatrix:
    X: np.ndarray
    y: np.ndarray
    filepaths: List[str]
    genders: List[str]
    locations: List[str]


def build_heart_dataframe(hs: pd.DataFrame, mix: pd.DataFrame) -> pd.DataFrame:
    heart_single = hs.copy()
    heart_single = heart_single[["gender", "heart_sound_type", "location", "filename"]]
    heart_single = heart_single.rename(columns={"filename": "filepath"})
    heart_single["source"] = "heart_single"

    heart_ref = mix[["gender", "heart_sound_type", "location", "heart_ref_file"]]
    heart_ref = heart_ref.rename(columns={"heart_ref_file": "filepath"})
    heart_ref["source"] = "heart_ref"

    return pd.concat([heart_single, heart_ref], ignore_index=True)


def build_lung_dataframe(ls: pd.DataFrame, mix: pd.DataFrame) -> pd.DataFrame:
    lung_single = ls.copy()
    lung_single = lung_single[["gender", "lung_sound_type", "location", "filename"]]
    lung_single = lung_single.rename(columns={"filename": "filepath"})
    lung_single["source"] = "lung_single"

    lung_ref = mix[["gender", "lung_sound_type", "location", "lung_ref_file"]]
    lung_ref = lung_ref.rename(columns={"lung_ref_file": "filepath"})
    lung_ref["source"] = "lung_ref"

    return pd.concat([lung_single, lung_ref], ignore_index=True)


def make_heart_design_matrix(heart_df: pd.DataFrame, sr=22050) -> DesignMatrix:
    X, meta = extract_feature_matrix(
        df=heart_df,
        filepath_col="filepath",
        gender_col="gender",
        location_col="location",
        sr=sr,
    )

    return DesignMatrix(
        X=X,
        y=heart_df["heart_sound_type"].values,
        filepaths=meta["filepaths"],
        genders=meta["genders"],
        locations=meta["locations"],
    )


def make_lung_design_matrix(lung_df: pd.DataFrame, sr=22050) -> DesignMatrix:
    X, meta = extract_feature_matrix(
        df=lung_df,
        filepath_col="filepath",
        gender_col="gender",
        location_col="location",
        sr=sr,
    )

    return DesignMatrix(
        X=X,
        y=lung_df["lung_sound_type"].values,
        filepaths=meta["filepaths"],
        genders=meta["genders"],
        locations=meta["locations"],
    )
