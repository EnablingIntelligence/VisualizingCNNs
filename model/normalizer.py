import torch
import torch.nn as nn
from enum import Enum


class Norm(Enum):
    CONTRAST = 0
    LOCAL = 1


class ContrastNorm(nn.Module):
    """
    https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.abs(torch.tanh(x))
        return x


class LocalResponseNorm(nn.Module):
    """
    https://pytorch.org/docs/stable/generated/torch.nn.LocalResponseNorm.html#torch.nn.LocalResponseNorm
    """

    def __init__(self, size: int = 2) -> None:
        super().__init__()
        self.local_norm = nn.LocalResponseNorm(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.local_norm(x)


class Normalizer(nn.Module):
    def __init__(self, normalization_method: Norm = Norm.CONTRAST, local_size: int = 2) -> None:
        super().__init__()
        if normalization_method == Norm.CONTRAST:
            self.model = ContrastNorm()
        elif normalization_method == Norm.LOCAL:
            self.model = LocalResponseNorm(local_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)