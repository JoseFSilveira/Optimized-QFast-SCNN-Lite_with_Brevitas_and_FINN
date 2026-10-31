'''
Based on the modified version of Fast-SCNN in models/FastSCNN.py, changing the tradicional torch.nn (nn) layers to brevitas.nn (qnn) layers.
The following changes to make it compatible with quantization and translation to ONNX and then to FINN:
--> The input and activations are quantized to 8 bits using per-tensor quantization with a floating-point scale.
--> The weights are quantized to 8 bits using per-tensor quantization with a floating-point scale.
--> Added "qat" mode to the model, which allows for quantization-aware training and export to ONNX and FINN. This mode enables the final upsampling layer to be used.
--> Added "finn" mode to the model, which is used for QONNX export for FINN framework. In this mode, the input preprocessing is done in the model, since FINN expects uint8 inputs.
--> Standard Add operations were replaced by Brevitas qnn.QuantEltwiseAdd to allow for addition of QuantTensors.
--> Dropout layers were removed from the model. QAT does not need dropout layers.
    -> nn.identity() was added in place of dropout layers to not alter the layer indexes, which is important for loading the weights for QAT.
--> The last upsampling layer is removed to avoid a large upsampling factor with 'nearest' mode, which can comprimise severly the accuracy of the model.
    -> obs: The last upsampling layer can be done in external post-processing step, with the output of the model being passed to a CPU or small GPU.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat, Uint8ActPerTensorFloat, Int8ActPerTensorFloat, Int8BiasPerTensorFloatInternalScaling
from brevitas.quant_tensor import QuantTensor
from brevitas.inject.enum import *

from config import ACT_BIT_WIDTH, OUT_BIT_WIDTH, WEIGHT_BIT_WIDTH, IM_SIZE


class IntActPerTensorFloat(Int8ActPerTensorFloat):
    """
    Custom subclass of Int8ActPerTensorFloat to allow for a different bit width to be set at runtime.
    This is necessary since the original Int8ActPerTensorFloat class has a fixed bit width of 8 bits.
    """
    bit_width = ACT_BIT_WIDTH

class UintActPerTensorFloat(Uint8ActPerTensorFloat):
    """
    Custom subclass of Uint8ActPerTensorFloat to allow for a different bit width to be set at runtime.
    This is necessary since the original Uint8ActPerTensorFloat class has a fixed bit width of 8 bits.
    """
    bit_width = ACT_BIT_WIDTH

class IntOutActPerTensorFloat(Int8ActPerTensorFloat):
    """
    Custom subclass of Int8ActPerTensorFloat to allow for a different bit width to be set at runtime.
    This is necessary since the original Int8ActPerTensorFloat class has a fixed bit width of 8 bits.
    """
    bit_width = OUT_BIT_WIDTH

class UintOutActPerTensorFloat(Uint8ActPerTensorFloat):
    """
    Custom subclass of Uint8ActPerTensorFloat to allow for a different bit width to be set at runtime.
    This is necessary since the original Uint8ActPerTensorFloat class has a fixed bit width of 8 bits.
    """
    bit_width = OUT_BIT_WIDTH

class IntWeightPerTensorFloat(Int8WeightPerTensorFloat):
    """
    Custom subclass of Int8WeightPerTensorFloat to allow for a different bit width to be set at runtime.
    This is necessary since the original Int8WeightPerTensorFloat class has a fixed bit width of 8 bits.
    """
    bit_width = WEIGHT_BIT_WIDTH


class QFastSCNN(nn.Module):

    def __init__(self, num_classes, mode="raw", **kwargs):
        super().__init__()

        # This modified model has modes instead of wrappers to facilitate state dict management when saving and loading the model.
        possible_modes = ["raw", "qat", "finn"]
        assert mode in possible_modes, f"Mode {mode} not supported. Choose one of {possible_modes}."
        self.mode = mode

        self.inp_quant = qnn.QuantIdentity(act_quant=IntActPerTensorFloat, return_quant_tensor=True)
        self.learning_to_downsample = LearningToDownsample(32, 48, 64)
        self.global_feature_extractor = GlobalFeatureExtractor(in_channels=64, block_channels=[64, 96, 128], out_channels=128, t=2, num_blocks=[1, 1, 1])
        self.feature_upsample = FeatureUpsampleModule(128, 128)
        self.classifier = Classifer(128, num_classes)

        # Pre calculating the input normalization to reduce the number of operation in the model. This is done fora a more hardware friendly implementation.
        # The original operation for the forward pass is: x = ((x / 225.0) - self.mean) / self.std)
        # The goal is to only perform x = x * finn_A - finn_B, where finn_A and finn_B are pre-calculated constants.
        # Only used in "finn" mode. FINN expects the input to be uint8, so the normalization is done in the model.
        if self.mode == "finn":
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            finn_A = (1.0 / (255.0 * std))
            finn_B = (mean / std)
            self.register_buffer('finn_A', finn_A)
            self.register_buffer('finn_B', finn_B)

    def forward(self, x):
        if self.mode == "qat":
            size = x.size()[2:] # Saves the original size of the input to upsample the output.
        if self.mode == "finn":
            # doing the hardware (finn) friendly version of x = ((x / 225.0) - self.mean) / self.std)
            x = (x * self.finn_A - self.finn_B)
        x = self.inp_quant(x)
        x = self.learning_to_downsample(x)
        x = self.global_feature_extractor(x)
        x = self.feature_upsample(x)
        x = self.classifier(x)
        if self.mode == "qat":
            x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return x


class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            qnn.QuantConv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False,
                            weight_quant=IntWeightPerTensorFloat,
                            return_quant_tensor=True),
            nn.BatchNorm2d(out_channels),
            qnn.QuantReLU(inplace=True, act_quant=UintActPerTensorFloat, return_quant_tensor=True)
        )

    def forward(self, x):
        return self.conv(x)


class _DSConv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            qnn.QuantConv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False,
                            weight_quant=IntWeightPerTensorFloat,                            
                            return_quant_tensor=True),
            nn.BatchNorm2d(dw_channels),
            qnn.QuantReLU(inplace=True, act_quant=UintActPerTensorFloat, return_quant_tensor=True),
            qnn.QuantConv2d(dw_channels, out_channels, 1, bias=False,
                            weight_quant=IntWeightPerTensorFloat,
                            return_quant_tensor=True),
            nn.BatchNorm2d(out_channels),
            qnn.QuantReLU(inplace=True, act_quant=UintActPerTensorFloat, return_quant_tensor=True)
        )

    def forward(self, x):
        return self.conv(x)


class _DWConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            qnn.QuantConv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False,
                            weight_quant=IntWeightPerTensorFloat,                            
                            return_quant_tensor=True),
            nn.BatchNorm2d(out_channels),
            qnn.QuantReLU(inplace=True, act_quant=UintActPerTensorFloat, return_quant_tensor=True)
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
            qnn.QuantConv2d(in_channels * t, out_channels, 1, bias=False,
                            weight_quant=IntWeightPerTensorFloat,
                            return_quant_tensor=True),
            nn.BatchNorm2d(out_channels),
        )

        # quantize the output so it can be used in the skip connection addition
        self.act_quant = qnn.QuantIdentity(act_quant=IntActPerTensorFloat, return_quant_tensor=True)
        
        # Added quantization for the skip connection
        if self.use_shortcut:
            self.add = qnn.QuantEltwiseAdd(return_quant_tensor=True,
                                           input_quant=IntActPerTensorFloat, output_quant=IntActPerTensorFloat)
        
    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = self.add(x, out) # QuantEltwiseAdd already quantizes the output, so no need to quantize it again.
        else:
            out = self.act_quant(out) # If there is no skip connection, we need to quantize the output before returning it.
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
            qnn.QuantConv2d(out_channels, out_channels, 1,
                            weight_quant=IntWeightPerTensorFloat,
                            return_quant_tensor=False), # Relu will be applied after, so the output of this layer does not need to be quantized.
            nn.BatchNorm2d(out_channels)
        )

        self.relu = qnn.QuantReLU(inplace=True, act_quant=UintActPerTensorFloat, return_quant_tensor=True)

    def upsample(self, x, scale_factor):
                return F.interpolate(x, scale_factor=scale_factor, mode='nearest')                                           

    def forward(self, input_feature):
        input_feature = self.upsample(input_feature, scale_factor=self.scale_factor)
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
            nn.Identity(), # in place of dropout to not alter layer indexes.
            qnn.QuantConv2d(dw_channels, num_classes, 1,
                            weight_quant=IntWeightPerTensorFloat,
                            act_quant=IntOutActPerTensorFloat,
                            return_quant_tensor=False)
        )
        self.out_quant = qnn.QuantIdentity(act_quant=IntOutActPerTensorFloat, return_quant_tensor=True)

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return self.out_quant(x)