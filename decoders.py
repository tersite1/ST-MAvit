import torch
import torch.nn as nn

class SegmentationDecoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.up_layers = nn.Sequential(
            nn.ConvTranspose2d(768, 384, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(384, 192, kernel_size=3, stride=2),
            nn.ReLU()
        )
        self.final_conv = nn.Conv2d(192, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.up_layers(x)
        x = self.final_conv(x)
        return x
