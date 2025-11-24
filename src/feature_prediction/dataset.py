from features import extract_all_features
from audio.io import load_wav_scipy
from tqdm import tqdm
import pandas as pd

def build_dataset(mixed, organ="heart"):
    X_set = []
    y_set = []

    single_file_column = "heart_ref_file" if organ == "heart" else "lung_ref_file"

    for _, record in tqdm(mixed.iterrows(), total=len(mixed), desc=f"Creating (X,y) features for {organ} dataset"):
        mixed_sr, mixed_audio = load_wav_scipy(record["mixture_file"])
        X_features = extract_all_features(mixed_audio, mixed_sr)
        X_set.append(X_features)

        single_sr, single_audio = load_wav_scipy(record[single_file_column])
        y_features = extract_all_features(single_audio, single_sr)
        y_set.append(y_features)

    return pd.DataFrame(X_set), pd.DataFrame(y_set)

