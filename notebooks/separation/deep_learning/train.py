import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .datasets import HeartRegressionDataset
from .unet_model import HeartUNet


def train_heart_predictor(
    df,
    epochs=20,
    batch_size=4,
    lr=1e-3,
    segment_len=65536,
):
    device = torch.device("cpu")

    dataset = HeartRegressionDataset(df, segment_len=segment_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = HeartUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    losses = []

    for ep in range(epochs):
        total = 0.0
        model.train()

        for mix, heart in loader:
            mix = mix.unsqueeze(1).to(device)
            heart = heart.unsqueeze(1).to(device)

            pred = model(mix)

            # crop if mismatch
            L = min(pred.shape[-1], heart.shape[-1])
            pred = pred[..., :L]
            target = heart[..., :L]

            loss = loss_fn(pred, target)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        ep_loss = total / len(loader)
        losses.append(ep_loss)
        print(f"Epoch {ep+1}/{epochs} - Loss: {ep_loss:.6f}")

    return model, losses
