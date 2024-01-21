import torch
import torch.nn as nn
from normalizer import Normalizer


class AlexNet1(nn.Module):
    """
    This implementation is based on section 4.1:
    https://arxiv.org/pdf/1311.2901.pdf

    By default we use the local response normalization.

    Input: Tensor-Images with the shape B x 3 x 224 x 224 (B x C x H x W).
    Output: Tensor with B x num_classes, where B represents the batch size.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1_000,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=7, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Normalizer(),
            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Normalizer(),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Normalizer(),
        )

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
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def conv_forward(self, x: torch.Tensor) -> torch.tensor:
        return self.features(x)


class DeconvAlexNet1(nn.Module):
    # TODO correct deconv for alexnet1
    def __init__(self, in_channels: int = 256, out_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 384, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 96, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, out_channels, kernel_size=7, stride=2, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.tensor:
        return self.features(x)


class AlexNet2(nn.Module):
    """
    This implementation is based on section 3.5:
    https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

    By default we use the local response normalization.

    Input: Tensor-Images with the shape B x 3 x 227 x 227 (B x C x H x W).
    Output: Tensor with B x num_classes, where B represents the batch size.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1_000,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            Normalizer(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5),
            nn.ReLU(inplace=True),
            Normalizer(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    m = AlexNet1()
    # m = AlexNet2()
    md = DeconvAlexNet1()

    B, C, H, W = 4, 3, 224, 224
    batch = torch.randn(size=(B, C, H, W), dtype=torch.float32)
    print(batch.shape)

    output = m.conv_forward(batch)
    print(output.shape)

    output = md(output)
    print(output.shape)
