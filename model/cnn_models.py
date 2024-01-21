import torch
import torch.nn as nn
from normalizer import Normalizer


class AlexNet(nn.Module):
    #TODO implement AlexNet w.r.t. normalizer and deconv
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1_000,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


m = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=2),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),
    Normalizer(),
)

B, C, H, W = 4, 3, 224, 224
batch = torch.randn(size=(B, C, H, W), dtype=torch.float32)
output = m(batch)
print(output.shape)
