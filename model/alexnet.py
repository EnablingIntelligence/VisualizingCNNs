from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.modules.module import T

from model.normalizer import Normalizer
from utils import Config, get_device


@dataclass
class MaxPoolingResults:
    pooling_indices: torch.Tensor
    feature_size: torch.Size


class AlexNetConfig:
    # pylint: disable=too-few-public-methods

    def __init__(self, config: Config):
        self.in_channels = config.in_channels
        self.num_classes = config.num_classes
        self.dropout = config.dropout
        self.normalization_method = config.normalization_method
        self.local_size = config.local_size


class AlexNet(nn.Module):
    # pylint: disable=too-many-instance-attributes

    def __init__(self, config: AlexNetConfig):
        super().__init__()
        self._deconv_eval = False
        self.pooling_indicies = []
        self.relu = nn.ReLU(inplace=True)
        self.norm = Normalizer(config.normalization_method, config.local_size)

        self.conv1 = nn.Conv2d(config.in_channels, 96, kernel_size=7, stride=2, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, return_indices=True)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, return_indices=True)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2, return_indices=True)

        self.unpool5 = nn.MaxUnpool2d(kernel_size=3, stride=2)
        self.deconv5 = nn.ConvTranspose2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.deconv4 = nn.ConvTranspose2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(384, 256, kernel_size=3, stride=1, padding=1)

        self.unpool2 = nn.MaxUnpool2d(kernel_size=3, stride=2)
        self.deconv2 = nn.ConvTranspose2d(256, 96, kernel_size=5, stride=2, padding=2)

        self.unpool1 = nn.MaxUnpool2d(kernel_size=3, stride=2)
        self.deconv1 = nn.ConvTranspose2d(96, config.in_channels, kernel_size=7, stride=2, padding=2)

        self.classifier = nn.Sequential(
            nn.Dropout(p=config.dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, config.num_classes),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def __initialize_deconv_layers(self):
        self.__initialize_deconv_layer(self.deconv1, self.conv1)
        self.__initialize_deconv_layer(self.deconv2, self.conv2)
        self.__initialize_deconv_layer(self.deconv3, self.conv3)
        self.__initialize_deconv_layer(self.deconv4, self.conv4)
        self.__initialize_deconv_layer(self.deconv5, self.conv5)

    @staticmethod
    def __initialize_deconv_layer(deconv: nn.Module, conv: nn.Module):
        deconv.weight = conv.weight

    def __clear_forward_cache(self):
        self.pooling_indicies.clear()

    def __add_forward_cache_entry(self, idx, size):
        if self._deconv_eval:
            self.pooling_indicies.append(MaxPoolingResults(idx, size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.__clear_forward_cache()

        feat = self.conv1(x)
        feat = self.relu(feat)
        size1 = feat.size()
        feat, idx1 = self.pool1(feat)
        feat = self.norm(feat)
        self.__add_forward_cache_entry(idx1, size1)

        feat = self.conv2(feat)
        feat = self.relu(feat)
        size2 = feat.size()
        feat, idx2 = self.pool2(feat)
        feat = self.norm(feat)
        self.__add_forward_cache_entry(idx2, size2)

        feat = self.conv3(feat)
        feat = self.relu(feat)

        feat = self.conv4(feat)
        feat = self.relu(feat)

        feat = self.conv5(feat)
        feat = self.relu(feat)
        size5 = feat.size()
        feat, idx5 = self.pool5(feat)
        feat = self.norm(feat)
        self.__add_forward_cache_entry(idx5, size5)

        feat = torch.flatten(feat, 1)
        feat = self.classifier(feat)
        return feat

    @staticmethod
    def log_results(feat: torch.Tensor, deconv_layer: nn.ConvTranspose2d) -> dict:
        feat_maps = feat.clone()
        layer_weights = deconv_layer.weight.clone()
        return {"activation_maps": feat_maps.detach().cpu(), "kernel_weights": layer_weights.detach().cpu()}

    def deconv_forward(self, x: torch.Tensor) -> dict:
        if not self._deconv_eval or len(self.pooling_indicies) == 0:
            raise ValueError("Model not in deconv mode")

        deconv_results = {}
        with torch.no_grad():
            feat = x

            results5 = self.pooling_indicies[2]
            idx5, size5 = results5.pooling_indices, results5.feature_size
            feat = self.unpool5(feat, idx5, output_size=size5)
            feat = self.relu(feat)
            feat = self.deconv5(feat)
            deconv_results["DeConv5"] = self.log_results(feat, self.deconv5)

            feat = self.relu(feat)
            feat = self.deconv4(feat)
            deconv_results["DeConv4"] = self.log_results(feat, self.deconv4)

            feat = self.relu(feat)
            feat = self.deconv3(feat)
            deconv_results["DeConv3"] = self.log_results(feat, self.deconv3)

            results2 = self.pooling_indicies[1]
            idx2, size2 = results2.pooling_indices, results2.feature_size
            feat = self.unpool2(feat, idx2, output_size=size2)
            feat = self.relu(feat)
            feat = self.deconv2(feat)
            deconv_results["DeConv2"] = self.log_results(feat, self.deconv2)

            results1 = self.pooling_indicies[0]
            idx1, size1 = results1.pooling_indices, results1.feature_size
            feat = self.unpool2(feat, idx1, output_size=size1)
            feat = self.relu(feat)
            feat = self.deconv1(feat)
            deconv_results["DeConv1"] = self.log_results(feat, self.deconv1)

        return deconv_results

    def deconv_eval(self: T) -> T:
        super().eval()
        self._deconv_eval = True
        self.__clear_forward_cache()
        self.__initialize_deconv_layers()
        return self

    def train(self: T, mode: bool = True) -> T:
        self._deconv_eval = False
        self.__clear_forward_cache()
        return super().train(mode)

    def eval(self: T) -> T:
        self._deconv_eval = False
        self.__clear_forward_cache()
        return super().eval()

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))

    @staticmethod
    def get_model_from_config(config: Config) -> T:
        alexnet_config = AlexNetConfig(config)
        model = AlexNet(alexnet_config)

        if config.model_file:
            model.load(config.model_file)
            print("model checkpoint loaded successfully")

        model.to(get_device())

        return model
