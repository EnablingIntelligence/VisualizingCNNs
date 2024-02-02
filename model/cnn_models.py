from typing import Any
import torch
import torch.nn as nn
from normalizer import Normalizer
from dataclasses import dataclass


@dataclass
class MaxPoolingResults:
    pooling_indicies: torch.Tensor
    feature_size: torch.Size


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

    def forward_conv_layers(self, x: torch.Tensor) -> torch.TensorType:
        feat = x
        for name, sub_module in self.features.named_modules():
            if len(name) == 0:
                continue
            elif isinstance(sub_module, nn.MaxPool2d):
                # TODO get the next feat map shape for the outputsize of unmaxpooling
                feat, pool_idx = sub_module(feat)
                random_feat = torch.rand(size=(feat.shape))
                
                self.pooling_indicies.append(MaxPoolingResults(pool_idx, feat))
            else:
                feat = sub_module(feat)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_conv_layers(x)
        feat = torch.flatten(feat, 1)
        feat = self.classifier(feat)
        return feat


class DeConv:
    """
    This class defines the deconvolution model for a Conv_Model instance.
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
        self.trained_model = trained_model
        self.pool_idx_gen = self.pooling_idx_generator()
        
        if len(self.pooling_indicies) > 3:
            # always consider the last 3 pooling indicies
            self.pooling_indicies = self.pooling_indicies[3:]

    def pooling_idx_generator(self) -> MaxPoolingResults:
        for pool_idx in self.pooling_indicies[::-1]:
            yield pool_idx

    def __call__(self, x: torch.Tensor) -> list[torch.Tensor]:
        feat = x
        feature_maps = list()
        for name, module in self.deconv_model.named_children():
            if isinstance(module, nn.ConvTranspose2d):
                feat = module(feat)
                feature_maps.append(feat)
            elif isinstance(module, nn.MaxUnpool2d):
                pooling_results = next(self.pool_idx_gen)
                feat = module(feat, pooling_results.pooling_indicies)
            else:
                feat = module(feat)
        return feature_maps


class DeConvBuilder:
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

    def disable_gradients(self, model: nn.Module):
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



class AlexNet_Final(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1_000,
        dropout: float = 0.5,
    ):
        super().__init__()
        self._deconv_eval = False
        self.pooling_indicies = list()
        self.relu = nn.ReLU(inplace=True)
        self.norm = Normalizer()
        
        self.conv1 = nn.Conv2d(in_channels, 96, kernel_size=7, stride=2, padding=2)
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
        self.deconv1 = nn.ConvTranspose2d(96, in_channels, kernel_size=7, stride=2, padding=2)
                
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
        for layer in self.children():
            if isinstance(layer, nn.ConvTranspose2d):
                for param in layer.parameters():
                    param.requires_grad = False


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.pooling_indicies = list()
        
        # B x 96 x 111 x 111
        feat = self.conv1(x) 
        feat = self.relu(feat)
        size1 = feat.size() # store the output size for the corresponding unpooling layer
        # B x 96 x 55 x 55
        feat, idx1 = self.pool1(feat)
        feat = self.norm(feat)
        
        # B x 256 x 28 x 28
        feat = self.conv2(feat)
        feat = self.relu(feat)
        size2 = feat.size() # store the output size for the corresponding unpooling layer
        # B x 256 x 13 x 13
        feat, idx2 = self.pool2(feat)
        feat = self.norm(feat)
        
        # B x 384 x 13 x 13
        feat = self.conv3(feat)
        feat = self.relu(feat)
        
        # B x 384 x 13 x 13
        feat = self.conv4(feat)
        feat = self.relu(feat)
        
        # B x 256 x 13 x 13
        feat = self.conv5(feat)
        feat = self.relu(feat)
        size5 = feat.size() # store the output size for the corresponding unpooling layer 
        # B x 256 x 6 x 6
        feat, idx5 = self.pool5(feat)
        feat = self.norm(feat)
        
        if self._deconv_eval:
            self.pooling_indicies.append(MaxPoolingResults(idx1, size1))
            self.pooling_indicies.append(MaxPoolingResults(idx2, size2))
            self.pooling_indicies.append(MaxPoolingResults(idx5, size5))
        
        feat = torch.flatten(feat, 1)
        feat = self.classifier(feat)
        return feat
    
    def deconv_eval(self):
        self._deconv_eval = True
    
    def deconv_forward(self, x: torch.Tensor) -> dict:
        
        if not self._deconv_eval or len(self.pooling_indicies) == 0 :
            raise ValueError("Model not in deconv mode")
        
        feature_maps = {}
        with torch.no_grad():
            # B x 256 x 6 x 6
            feat = x
            
            results5 = self.pooling_indicies[2]
            idx5, size5 = results5.pooling_indicies, results5.feature_size
            # B x 256 x 13 x 13
            feat = self.unpool5(feat, idx5, output_size=size5)
            feat = self.relu(feat)
             # B x 384 x 13 x 13
            feat = self.deconv5(feat)
            feature_maps["conv5"] = feat
            
            feat = self.relu(feat)
            # B x 384 x 13 x 13
            feat = self.deconv4(feat)
            feature_maps["conv4"] = feat
            
            feat = self.relu(feat)
            # B x 256 x 28 x 28
            feat = self.deconv3(feat)
            feature_maps["conv3"] = feat
            
            results2 = self.pooling_indicies[1]
            idx2, size2 = results2.pooling_indicies, results2.feature_size
            # B x 256 x 28 x 28
            feat = self.unpool2(feat, idx2, output_size=size2)
            feat = self.relu(feat)
            # B x 96 x 55 x 55
            feat = self.deconv2(feat)
            feature_maps["conv2"] = feat
            
            results1 = self.pooling_indicies[0]
            idx1, size1 = results1.pooling_indicies, results1.feature_size
            feat = self.unpool2(feat, idx1, output_size=size1)
            feat = self.relu(feat)
            feat = self.deconv1(feat)
            feature_maps["conv1"] = feat
        
        return feature_maps
        
    

if __name__ == "__main__":  #()
    B, C, H, W = 4, 3, 224, 224
    batch = torch.randn(size=(B, C, H, W), dtype=torch.float32)
    print(f"Batch size: {batch.shape}")

    # model = AlexNet()
    model = AlexNet_Final()
    model.deconv_eval()
    output = model(batch)
    print(f"Output shape: {output.shape}")
    
    feat_output = torch.randn(size=(B, 256, 6, 6), dtype=torch.float32)
    output_deconv = model.deconv_forward(feat_output)    
    
    for feat_map in output_deconv.values():
        print(feat_map.shape)
    