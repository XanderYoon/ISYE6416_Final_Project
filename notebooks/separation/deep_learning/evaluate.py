import torch
import pandas as pd
from audio.io import load_wav_scipy as load_audio
from .metrics import mse, corr, r2


def evaluate_full_files(model, df):
    device = next(model.parameters()).device
    model.eval()

    results = []

    for idx, row in df.iterrows():
        sr, mix = load_audio(row["mixture_file"], normalize=True)
        _, heart_ref = load_audio(row["heart_ref_file"], normalize=True)

        x = torch.tensor(mix, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(x).cpu().numpy()[0, 0]

        L = min(len(pred), len(heart_ref))
        y = pred[:L]
        h = heart_ref[:L]

        results.append({
            "id": row.get("mixed_sound_id", idx),
            "mse": mse(h, y),
            "corr": corr(h, y),
            "r2": r2(h, y),
        })

    return pd.DataFrame(results)
