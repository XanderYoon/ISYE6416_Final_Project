from __future__ import annotations
import numpy as np
import pandas as pd
import librosa
from typing import Tuple, List, Dict
from sklearn.preprocessing import OneHotEncoder


def extract_audio_feature_vector(y, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = mfcc.mean(axis=1)

    zcr = librosa.feature.zero_crossing_rate(y=y)[0].mean()
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0].mean()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0].mean()

    return np.concatenate([
        mfcc_mean,
        np.array([zcr, centroid, bandwidth, rolloff])
    ])


def extract_feature_matrix(
    df: pd.DataFrame,
    filepath_col="filepath",
    gender_col="gender",
    location_col="location",
    sr=22050,
) -> Tuple[np.ndarray, Dict[str, List[str]]]:

    feats = []
    genders = []
    locations = []
    paths = []

    for _, row in df.iterrows():
        fp = row[filepath_col]
        gender = row[gender_col]
        loc = row[location_col]

        y, sr_loaded = librosa.load(fp, sr=sr)
        fv = extract_audio_feature_vector(y, sr_loaded)

        feats.append(fv)
        genders.append(gender)
        locations.append(loc)
        paths.append(fp)

    feats = np.vstack(feats)

    meta_df = pd.DataFrame({
        "gender": genders,
        "location": locations,
    })

    enc = OneHotEncoder(sparse_output=False)
    meta = enc.fit_transform(meta_df)

    X = np.hstack([feats, meta])

    return X, {
        "filepaths": paths,
        "genders": genders,
        "locations": locations,
        "categories": enc.categories_
    }
