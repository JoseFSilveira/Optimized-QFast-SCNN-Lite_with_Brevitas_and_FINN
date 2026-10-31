###########################################################################
# Created by: Tramac
# Date: 2019-03-25
# Copyright (c) 2017
###########################################################################

'''
From repo https://github.com/Tramac/Fast-SCNN-pytorch with the following modifications to make it compatible with translation to ONNX and then to FINN:
--> The auxiliary output and the corresponding code in the training loop were removed since they are not being used in the training and can cause issues when translating the model to ONNX and then to FINN, which do not support multiple outputs.
--> The number of blocks in the GlobalFeatureExtractor was reduced from [3, 3, 3] to [1, 1, 1] to reduce model depth and FPGA FIFO's.
--> Expand factor 't' reduced from 6 to 2 in the GlobalFeatureExtractor to reduce the number of parameters.
--> Pyramid Pooling module was replaced by a Context Aggregation module that uses a single 3x3 convolution followed by a 1x1 convolution to reduce the number of model FIFOs.
--> Feature Fusion module was replaced by a new Feature Upsample Module that only upsamples the features from the Global Feature Extractor to match the input image size,the long skip connection was removed.
'''

"""Fast Segmentation Convolutional Neural Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['FastSCNN', 'get_fast_scnn']


class FastSCNN(nn.Module):
    def __init__(self, num_classes, aux=False, **kwargs):
        super(FastSCNN, self).__init__()
        #self.aux = aux
        self.learning_to_downsample = LearningToDownsample(32, 48, 64)
        # The number of blocks in the GlobalFeatureExtractor was reduced from [3, 3, 3] to [1, 1, 1] to reduce model depth and FPGA FIFO's.
        self.global_feature_extractor = GlobalFeatureExtractor(in_channels=64, block_channels=[64, 96, 128], out_channels=128, t=2, num_blocks=[1, 1, 1])
        self.feature_upsample = FeatureUpsampleModule(128, 128)
        self.classifier = Classifer(128, num_classes)
        # Removing the tuple output since it can cause issues when translating the model to ONNX and then to FINN, and the auxiliary output is not being used in the training loop
        '''
        if self.aux:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(32, num_classes, 1)
            )
        '''

    def forward(self, x):
        size = x.size()[2:]
        x = self.learning_to_downsample(x)
        x = self.global_feature_extractor(x)
        x = self.feature_upsample(x)
        x = self.classifier(x)
        #outputs = []
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        # Removing the auxiliary output since it can cause issues when translating the model to ONNX and then to FINN, and the auxiliary output is not being used in the training loop
        '''
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(higher_res_features)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)
        '''
        return x


class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _DSConv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _DWConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class LinearBottleneck(nn.Module):
    """LinearBottleneck used in MobileNetV2"""

    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            # pw
            _ConvBNReLU(in_channels, in_channels * t, 1),
            # dw
            _DWConv(in_channels * t, in_channels * t, stride),
            # pw-linear
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


class ContextAggregationModule(nn.Module):
    """Context aggregation module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(ContextAggregationModule, self).__init__()
        self.conv1 = _ConvBNReLU(in_channels, in_channels, kernel_size=3, stride = 1, padding=1, **kwargs)
        self.conv2 = _ConvBNReLU(in_channels, out_channels, kernel_size=1, stride = 1, padding=0, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class LearningToDownsample(nn.Module):
    """Learning to downsample module"""

    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64, **kwargs):
        super(LearningToDownsample, self).__init__()
        self.conv = _ConvBNReLU(3, dw_channels1, 3, 2)
        self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = _DSConv(dw_channels2, out_channels, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class GlobalFeatureExtractor(nn.Module):
    """Global feature extractor module"""

    def __init__(self, in_channels=64, block_channels=(64, 96, 128),
                 out_channels=128, t=6, num_blocks=(3, 3, 3), **kwargs):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 2)
        self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 2)
        self.bottleneck3 = self._make_layer(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 1)
        self.cag = ContextAggregationModule(block_channels[2], out_channels)

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(block(inplanes, planes, t, stride))
        for i in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.cag(x)
        return x


class FeatureUpsampleModule(nn.Module):
    """Feature fusion module"""

    def __init__(self, in_channels, out_channels, scale_factor=4, **kwargs):
        super(FeatureUpsampleModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = _DWConv(in_channels, out_channels, 1)
        self.conv_lower_res = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(True)

    def forward(self, input_feature):
        input_feature = F.interpolate(input_feature, scale_factor=4, mode='nearest')
        input_feature = self.dwconv(input_feature)
        input_feature = self.conv_lower_res(input_feature)

        return self.relu(input_feature)

class Classifer(nn.Module):
    """Classifer"""

    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super(Classifer, self).__init__()
        self.dsconv1 = _DSConv(dw_channels, dw_channels, stride)
        self.dsconv2 = _DSConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(dw_channels, num_classes, 1)
        )

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x