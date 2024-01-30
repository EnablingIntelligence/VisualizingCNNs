from typing import Any
import torch
import torch.nn as nn
from normalizer import Normalizer
from dataclasses import dataclass


@dataclass
class MaxPoolingResults:
    pooling_indicies: torch.Tensor
    feature_map: torch.Tensor


class Conv_Model(nn.Module):
    """
    Interface for the Deconv Model: TODO rewrite doc string
    """

    pooling_indicies: list[MaxPoolingResults]
    features: nn.Sequential
    classifier: nn.Sequential


class AlexNet(Conv_Model):
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

    def __forward_conv_layers(self, x: torch.Tensor) -> torch.TensorType:
        feat = x
        for name, sub_module in self.features.named_modules():
            if len(name) == 0:
                continue
            elif isinstance(sub_module, nn.MaxPool2d):
                feat, pool_idx = sub_module(feat)
                self.pooling_indicies.append(MaxPoolingResults(pool_idx, feat))
            else:
                feat = sub_module(feat)

        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.__forward_conv_layers(x)
        feat = torch.flatten(feat, 1)
        feat = self.classifier(feat)
        return feat


class DeConv:
    """
    This module defines the deconvolution model for a Conv_Model instance.
    The Conv_Model must be trained beforehand to generate the required pooling indices for the unpooling operation.
    See: https://pytorch.org/docs/stable/generated/torch.nn.MaxUnpool2d.html
    """

    def __init__(self, trained_model: Conv_Model = AlexNet()) -> None:
        super().__init__()
        self.DeConvBuilder = DeConvBuilder(
            trained_model.features, trained_model.pooling_indicies
        )
        self.deconv_model: nn.Module = self.DeConvBuilder.build()
        self.pooling_indicies: list[MaxPoolingResults] = trained_model.pooling_indicies

    def pooling_idx_generator(self) -> MaxPoolingResults:
        for pool_idx in self.pooling_indicies:
            yield pool_idx

    def forward_deconv(self, x: torch.Tensor) -> list[torch.Tensor]:
        feat = x
        feature_maps = list()
        for name, module in self.deconv_model.named_modules():
            if isinstance(module, nn.ConvTranspose2d):
                feat = module(x)
                feature_maps.append(feat)
            else:
                feat = module(feat)
        return feature_maps


class DeConvBuilder:
    # TransConv: required_grad = False
    def __init__(
        self, model: Conv_Model, pooling_indicies: list[MaxPoolingResults]
    ) -> None:
        self.model = model
        self.pooling_indicies = pooling_indicies
        self.layers = list()

    def conv2transpose(self, conv_layer: nn.Conv2d) -> nn.ConvTranspose2d:
        return nn.ConvTranspose2d(
            in_channels=conv_layer.out_channels,
            out_channels=conv_layer.in_channels,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
        )

    def cvt2unpooling_layer(self, max_pooling_layer: nn.MaxPool2d) -> nn.MaxUnpool2d:
        return nn.MaxUnpool2d(
            kernel_size=max_pooling_layer.kernel_size, stride=max_pooling_layer.stride
        )

    def disable_gradients(self, model):
        for module in model.children():
            for param in module.parameters():
                param.requires_grad = False

    def build(self) -> nn.Sequential:
        for name, sub_module in self.model[::-1].named_children():
            if isinstance(sub_module, nn.MaxPool2d):
                unpooling_layer = self.cvt2unpooling_layer(sub_module)
                self.layers.append(unpooling_layer)
            elif isinstance(sub_module, Normalizer) or isinstance(sub_module, nn.ReLU):
                activation_function = nn.ReLU(inplace=True)
                self.layers.append(activation_function)
            elif isinstance(sub_module, nn.Conv2d):
                transConv = self.conv2transpose(sub_module)
                self.layers.append(transConv)

        deconv = nn.Sequential(*self.layers)
        self.disable_gradients(deconv)

        return deconv


if __name__ == "__main__":  #
    B, C, H, W = 4, 3, 224, 224
    batch = torch.randn(size=(B, C, H, W), dtype=torch.float32)
    print(f"Batch size: {batch.shape}")

    model = AlexNet()
    output = model(batch)
    print(f"Output shape: {output.shape}")

    deconv_model = DeConv(model)

    # m = AlexNet()

    # indices_m = m.pooling_indicies
    # for idx in indices_m:
    #     print(type(idx), idx.shape)

    # dmb = DeConvBuilder(pooling_indicies=indices_m)

    # output = u(output, indices, output_size=batch.shape)
    # print(output.shape)

    # output, idxs_list = m(batch)
    # print(output.shape)
    # for idx_l in idxs_list:
    #     print(idx_l.shape)
