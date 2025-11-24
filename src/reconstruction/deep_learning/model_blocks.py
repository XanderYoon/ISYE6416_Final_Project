import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=9, s=1, p=4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=k, stride=1, padding=p),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=9, s=2, p=4, out_p=1):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, k, s, p, output_padding=out_p)
        self.conv = ConvBlock(out_ch * 2, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        # crop mismatch
        min_len = min(x.shape[-1], skip.shape[-1])
        x = x[..., :min_len]
        skip = skip[..., :min_len]

        return self.conv(torch.cat([x, skip], dim=1))
