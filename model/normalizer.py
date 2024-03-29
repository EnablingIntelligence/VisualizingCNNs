from enum import Enum

import torch
from torch import nn
from torch.nn.modules import LocalResponseNorm


class NormMode(Enum):
    CONTRAST = 0
    LOCAL = 1


class Normalizer(nn.Module):
    """
    This module enables two normalization methods.
     - Local response normalization
     - Contrast normalization
    """

    def __init__(self, normalization_method: NormMode, local_size: int):
        super().__init__()

        match normalization_method:
            case NormMode.LOCAL:
                self.model = LocalResponseNorm(local_size)

            case NormMode.CONTRAST:
                # https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
                self.model = lambda x: torch.abs(torch.tanh(x))

            case _:
                raise ValueError("Unknown normalization method")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
