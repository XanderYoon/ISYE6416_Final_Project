"""
pipeline_data_builder.py

Loads heart/lung single-source datasets, mixed datasets,
and provides typed dataset classes with utilities for:

- slicing (subset)
- conversion to numpy arrays
- conversion to pandas DataFrames
- string/pretty-printing for inspection
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from audio import load_wav_scipy
from features import extract_all_features
from metadata.loading import load_datasets

class Organ(Enum):
    HEART = "heart"
    LUNG = "lung"


@dataclass
class SingleSourceDataset:
    organ: Organ
    features: List[Dict[str, float]]
    labels: List[Any]

    def subset(self, indices: List[int]) -> "SingleSourceDataset":
        """Return a new SingleSourceDataset containing only selected rows."""
        return SingleSourceDataset(
            organ=self.organ,
            features=[self.features[i] for i in indices],
            labels=[self.labels[i] for i in indices],
        )

    def as_arrays(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Convert list-of-dicts → numeric matrix (X) and labels."""
        X, keys = self._flatten_dicts(self.features)
        y = np.array(self.labels)
        return X, y, keys

    def to_pandas(self) -> pd.DataFrame:
        """Return dataset as a pandas DataFrame."""
        X, keys = self._flatten_dicts(self.features)
        df = pd.DataFrame(X, columns=keys)
        df["label"] = self.labels
        df["organ"] = self.organ.value
        return df

    def to_string(self, max_rows=5) -> str:
        """Pretty-print summary for quick inspection."""
        df = self.to_pandas()
        return df.head(max_rows).to_string(index=False)

    @staticmethod
    def _flatten_dicts(dict_list: List[Dict[str, float]]):
        """Convert feature dicts → numpy matrix and sorted keys."""
        keys = sorted(dict_list[0].keys())
        arr = np.array([[d[k] for k in keys] for d in dict_list], dtype=float)
        return arr, keys


@dataclass
class MixedSourceDataset:
    organ: Organ
    mixture_features: List[Dict[str, float]]
    organ_features: List[Dict[str, float]]
    labels: List[Any]

    def subset(self, indices: List[int]) -> "MixedSourceDataset":
        """Return a new MixedSourceDataset with selected rows."""
        return MixedSourceDataset(
            organ=self.organ,
            mixture_features=[self.mixture_features[i] for i in indices],
            organ_features=[self.organ_features[i] for i in indices],
            labels=[self.labels[i] for i in indices],
        )

    def as_arrays(
        self
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Convert list-of-dicts to numpy matrices:
        Returns:
            X_mix, X_org, y, mix_keys, org_keys
        """
        X_mix, mix_keys = self._flatten_dicts(self.mixture_features)
        X_org, org_keys = self._flatten_dicts(self.organ_features)
        y = np.array(self.labels)
        return X_mix, X_org, y, mix_keys, org_keys

    def to_pandas(self) -> pd.DataFrame:
        """Convert dataset to a pandas DataFrame for inspection or saving."""
        X_mix, mix_keys = self._flatten_dicts(self.mixture_features)
        X_org, org_keys = self._flatten_dicts(self.organ_features)

        df_mix = pd.DataFrame(X_mix, columns=[f"mix_{k}" for k in mix_keys])
        df_org = pd.DataFrame(X_org, columns=[f"{self.organ.value}_{k}" for k in org_keys])

        df = pd.concat([df_mix, df_org], axis=1)
        df["label"] = self.labels
        df["organ"] = self.organ.value

        return df

    def to_string(self, max_rows=5) -> str:
        """Pretty-printed DataFrame head."""
        df = self.to_pandas()
        return df.head(max_rows).to_string(index=False)

    @staticmethod
    def _flatten_dicts(dict_list: List[Dict[str, float]]):
        keys = sorted(dict_list[0].keys())
        arr = np.array([[d[k] for k in keys] for d in dict_list], dtype=float)
        return arr, keys


def extract_from_dataframe(df, filename_col: str, label_col: str) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """
    Generic helper to extract features and labels from a dataframe.
    """
    features = []
    labels = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {label_col}"):
        sr, audio = load_wav_scipy(row[filename_col])
        feat = extract_all_features(audio, sr)
        features.append(feat)
        labels.append(row[label_col])

    return features, labels


def load_single_source_dataset(organ: Organ) -> SingleSourceDataset:
    """
    Loads a single-source dataset (heart OR lung).
    Returns:
        SingleSourceDataset
    """
    datasets = load_datasets()

    if organ == Organ.HEART:
        df = datasets["heart_data"]
        filename_col = "filename"
        label_col = "heart_sound_type"

    elif organ == Organ.LUNG:
        df = datasets["lung_data"]
        filename_col = "filename"
        label_col = "lung_sound_type"

    else:
        raise ValueError(f"Unsupported organ: {organ}")

    features, labels = extract_from_dataframe(df, filename_col, label_col)

    return SingleSourceDataset(
        organ=organ,
        features=features,
        labels=labels,
    )


def load_mixed_dataset(organ: Organ) -> MixedSourceDataset:
    """
    Loads mixed dataset for a specific organ (heart or lung).
    Includes:
        mixture_features
        organ-specific reference features
        labels
    """
    datasets = load_datasets()
    df = datasets["mixed_data"]

    mixture_features = []
    organ_features = []
    labels = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting mixed dataset ({organ.value})"):

        # Always extract mixture features
        sr_mix, audio_mix = load_wav_scipy(row["mixture_file"])
        mixture_features.append(extract_all_features(audio_mix, sr_mix))

        # Organ-specific reference
        if organ == Organ.HEART:
            ref_file = row["heart_ref_file"]
            label_val = row["heart_sound_type"]

        elif organ == Organ.LUNG:
            ref_file = row["lung_ref_file"]
            label_val = row["lung_sound_type"]

        sr_ref, audio_ref = load_wav_scipy(ref_file)
        organ_features.append(extract_all_features(audio_ref, sr_ref))
        labels.append(label_val)

    return MixedSourceDataset(
        organ=organ,
        mixture_features=mixture_features,
        organ_features=organ_features,
        labels=labels,
    )
