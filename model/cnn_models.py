from typing import Any
import torch
import torch.nn as nn
from normalizer import Normalizer


class AlexNet(nn.Module):
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
        self.pooling_indicies = list()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=7, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, return_indices=True),
            Normalizer(),
            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, return_indices=True),
            Normalizer(),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, return_indices=True),
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

    def _forward_conv_layers(self, x: torch.Tensor) -> torch.TensorType:
        feat = x
        for name, sub_module in self.features.named_modules():
            if len(name) == 0:
                continue
            elif isinstance(sub_module, nn.MaxPool2d):
                feat, pool_idx = sub_module(feat)
                self.pooling_indicies.append(pool_idx)
            else:
                feat = sub_module(feat)

        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._forward_conv_layers(x)
        feat = torch.flatten(feat, 1)
        feat = self.classifier(feat)
        return feat


class DeConvBuilder:
    def __init__(self, pooling_indicies: list[torch.tensor]) -> None:
        self.pooling_indicies = pooling_indicies
        self.layers = nn.ModuleList()

    def build(self, model: nn.Module) -> nn.Module:
        
        for name, sub_module in self.features.named_modules():
            if len(name) == 0:
                continue
            elif isinstance(sub_module, nn.MaxPool2d):
                feat, pool_idx = sub_module(feat)
                self.pooling_indicies.append(pool_idx)
            else:
                feat = sub_module(feat)

        return None
    


if __name__ == "__main__":
    m = AlexNet()

    B, C, H, W = 4, 3, 224, 224
    batch = torch.randn(size=(B, C, H, W), dtype=torch.float32)
    print(batch.shape)

    # m = nn.MaxPool2d(3, stride=2, return_indices=True)
    u = nn.MaxUnpool2d(3, stride=2)

    
    for name, sub_module in m.named_children():
        if isinstance(sub_module, nn.Sequential):
            for name, seq_module in sub_module.named_modules():
                print(seq_module)
        # print(name)
    
    # output, indices = m(batch)
    # print(output.shape)

    # output = u(output, indices, output_size=batch.shape)
    # print(output.shape)

    # output, idxs_list = m(batch)
    # print(output.shape)
    # for idx_l in idxs_list:
    #     print(idx_l.shape)
