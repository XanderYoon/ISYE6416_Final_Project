import torch.nn as nn
from .model_blocks import ConvBlock, UpBlock


class HeartUNet(nn.Module):

    def __init__(self):
        super().__init__()

        # ----- Encoder -----
        self.enc1 = ConvBlock(1, 32)
        self.down1 = nn.Conv1d(32, 64, 9, stride=2, padding=4)

        self.enc2 = ConvBlock(64, 64)
        self.down2 = nn.Conv1d(64, 128, 9, stride=2, padding=4)

        self.enc3 = ConvBlock(128, 128)
        self.down3 = nn.Conv1d(128, 256, 9, stride=2, padding=4)

        self.enc4 = ConvBlock(256, 256)
        self.down4 = nn.Conv1d(256, 512, 9, stride=2, padding=4)

        # bottleneck
        self.bottleneck = ConvBlock(512, 512)

        # ----- Decoder -----
        self.up4 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up2 = UpBlock(128, 64)
        self.up1 = UpBlock(64, 32)

        # final projection
        self.final = nn.Conv1d(32, 1, kernel_size=1)

    def forward(self, x):
        # encoder
        s1 = self.enc1(x)
        x = self.down1(s1)

        s2 = self.enc2(x)
        x = self.down2(s2)

        s3 = self.enc3(x)
        x = self.down3(s3)

        s4 = self.enc4(x)
        x = self.down4(s4)

        # bottleneck
        x = self.bottleneck(x)

        # decoder
        x = self.up4(x, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)

        return self.final(x)
