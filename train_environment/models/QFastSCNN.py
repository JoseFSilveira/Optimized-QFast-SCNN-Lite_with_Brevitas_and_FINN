'''
Based on the modified version of Fast-SCNN in models/FastSCNN.py, changing the tradicional torch.nn (nn) layers to brevitas.nn (qnn) layers.
The following changes to make it compatible with quantization and translation to ONNX and then to FINN:
--> The input and activations are quantized to 8 bits using per-tensor quantization with a floating-point scale.
--> The weights are quantized to 8 bits using per-tensor quantization with a floating-point scale.
--> Added "qat" mode to the model, which allows for quantization-aware training and export to ONNX and FINN. This mode enables the final upsampling layer to be used.
--> Added "finn" mode to the model, which is used for QONNX export for FINN framework. In this mode, the input preprocessing is done in the model, since FINN expects uint8 inputs.
--> PyramidPooling F.interpolate bilinear with final tensor size as argument is replaced by another F.interpolate with nearest neighbor interpolation with pre-calculated scale factors.
--> The adaptive average pooling layers are replaced by depthwise Conv layers ith kernel weights as 1/kernel_size, which is equivalent to average pooling, but allows for quantization and export to ONNX and FINN.
--> torch.cat() was replaced by a modification of Brevitas qnn.QuantCat to allow for concatenation of QuantTensors, since the QuantTensor class does not have a cat() method anymore (Brevitas Bug).
--> The original concat operation was replaced by a two-stage cascade of concat operations to make the operation compatible for hardware implementation.
    -> The original model had a concatenation of tensors with depths [128, 32, 32, 32, 32] to form a new tensor with depth=256, though Hardware concat layer requires tensors with equal shapes.
    -> The first concat operation only concatenates the tensors with depth=32, resulting in a new tensor with depth=128.
    -> The second concat operation concatenates the original 128-depth tensor with the previously generated 128-depth tensor, resulting in the final tensor with depth=256.
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


class AvgPoolFINNFriendly(nn.Module):
    """
    A replacement for the adaptive average pooling layers in the original model, which is compatible with quantization and export to ONNX and FINN.
    This is done by using a depthwise convolution with kernel weights as 1/kernel_size, which is equivalent to average pooling, but allows for quantization and export to ONNX and FINN.
    """

    def __init__(self, channels: int, kernel_size: tuple[int, int] | int, return_quant_tensor=False):
        super().__init__()
        self.return_quant_tensor = return_quant_tensor
        # Create the depthwise convolution layer with frozen weights of 1, which results in a depthwise sum of pixels in the kernel area.
        self.pool = qnn.QuantConv2d(in_channels=channels,
                                out_channels=channels,
                                kernel_size=kernel_size,
                                stride=kernel_size,
                                groups=channels,
                                bias=False,
                                weight_quant=IntWeightPerTensorFloat,
                                
                                return_quant_tensor=False)
        # Freeze the weights of the depthwise convolution layer to 1
        for param in self.pool.parameters():
            param.requires_grad = False
        nn.init.constant_(self.pool.weight.data, 1.0)
        self.pool.weight.data = self.pool.weight.data.to(torch.float32) # make sure the weights are in float32, since something was promoting the weights to double precision, which is not supported by FINN.

        # Define the kernel area for the average pooling operation
        if type(kernel_size) == int:
            kernel_area = kernel_size * kernel_size
        else:
            kernel_area = kernel_size[0] * kernel_size[1]
        avgpool_scale_factor = 1.0 / kernel_area # Defining a scale factor to assert a multiplication the forward method, since FINN does not support division operations.

        # The output of the depthwise convolution layer is divided by the kernel area to get the average pooling result.
        self.div = lambda x: x * avgpool_scale_factor

        # Quantize the output of the average pooling operation using per-tensor quantization with a floating-point scale.
        self.act_quant = qnn.QuantIdentity(act_quant=IntActPerTensorFloat, return_quant_tensor=True)

    def forward(self, x):
        x = self.pool(x)
        x = self.div(x)
        if self.return_quant_tensor:
            x = self.act_quant(x)
        return x


class CustomQuantCat(qnn.QuantCat):
    '''
    A modification of the original qnn.QuantCat replacing QuantTensor.cat() with torch.cat(), since the QuantTensor does not have cat() method anymore (Brevitas Bug).
    '''
    def forward(self,
                tensor_list: list[torch.Tensor] | list[QuantTensor],
                dim: int = 1) -> torch.Tensor | QuantTensor:
        quant_tensor_list = [self.unpack_input(t) for t in tensor_list]
        # shortcut execution through the export impl during export
        if self.export_mode:
            out = self.export_handler([qt.value for qt in quant_tensor_list])
            self._set_global_is_quant_layer(False)
            return out
        quant_tensor_list = [self.input_quant(qt) for qt in quant_tensor_list]
        # trigger an assert if scale factors and bit widths are None or different
        output = torch.cat(quant_tensor_list, dim=dim)
        quant_output = self.output_quant(output)
        return self.pack_output(quant_output)


class QFastSCNN(nn.Module):

    def __init__(self, num_classes, mode="raw", **kwargs):
        super().__init__()

        # This modified model has modes instead of wrappers to facilitate state dict management when saving and loading the model.
        possible_modes = ["raw", "qat", "finn"]
        assert mode in possible_modes, f"Mode {mode} not supported. Choose one of {possible_modes}."
        self.mode = mode

        self.inp_quant = qnn.QuantIdentity(act_quant=IntActPerTensorFloat, return_quant_tensor=True)
        self.learning_to_downsample = LearningToDownsample(32, 48, 64)
        self.global_feature_extractor = GlobalFeatureExtractor(in_channels=64, block_channels=[64, 96, 128], out_channels=128, t=2, num_blocks=[2, 2, 2])
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

class PyramidPooling(nn.Module):
    """
    Pyramid pooling module

    --> The pool sizes were changed from [1x1, 2x2, 3x3, 6x6] to [1x1, 2x2, 4x4, 8x8] for two distint reasons, since for ONNX export, the image size needs to be divisible by the pool size (in this case 32 and 64).
    --> The original model uses adaptive average pooling, which can be replaced by a quantized version of standard Average Pooling with fixes pre defined kernel sizes.
    --> The original model uses F.interpolate, which is being replaced by a QuantConvTranspose2d with frozen weights of torch.ones() to avoid issues when translating the model to ONNX and then to FINN, since onnx runtime outputs an error when a Resize block parameter is empty.
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)

        # [pool_out_H, pool_out_W] = [in_H, in_W] / kernel_size
        # in_H and in_W are equal to the model input image sizes divided by 32.
        in_H = IM_SIZE[0] // 32
        self.pool_out_H = [1, 2, 4, 8]  # Kernel sizes for the pyramid pooling layers
        self.kernel_size = [in_H // pool_out for pool_out in self.pool_out_H]  # Kernel sizes for the pyramid pooling layers during training

        # The original model uses adaptive average pooling, which can be replaced by a standard Average Pooling with pre-defined kernel sizes.
        # Necessary to create new instance for each pool size since the output size is a parameter in the quantized version.

        self.pool1 = AvgPoolFINNFriendly(in_channels, self.kernel_size[0], return_quant_tensor=True)
        self.pool2 = AvgPoolFINNFriendly(in_channels, self.kernel_size[1], return_quant_tensor=True)
        self.pool3 = AvgPoolFINNFriendly(in_channels, self.kernel_size[2], return_quant_tensor=True)
        self.pool4 = AvgPoolFINNFriendly(in_channels, self.kernel_size[3], return_quant_tensor=True)

        # First Concat operation to concatenate the features with depth=32, resulting in a new tensor with depth=128; return_quant_tensor=False and output_quant=None since the next concat operation will quantize the output again.
        self.concat1 = CustomQuantCat(return_quant_tensor=False,
                                      input_quant=IntActPerTensorFloat, output_quant=None)
        # Second Concat operation to concatenate the original tensor with depth=128 with the previously generated tensor with depth=128, resulting in the final tensor with depth=256.
        self.concat2 = CustomQuantCat(return_quant_tensor=True,
                                      input_quant=IntActPerTensorFloat, output_quant=IntActPerTensorFloat)

    def upsample(self, x, scale_factor):
            return F.interpolate(x, scale_factor=scale_factor, mode='nearest', recompute_scale_factor=False)

    def forward(self, x):

        feat1 = self.upsample(self.conv1(self.pool1(x)), scale_factor=self.kernel_size[0])
        feat2 = self.upsample(self.conv2(self.pool2(x)), scale_factor=self.kernel_size[1])
        feat3 = self.upsample(self.conv3(self.pool3(x)), scale_factor=self.kernel_size[2])
        feat4 = self.upsample(self.conv4(self.pool4(x)), scale_factor=self.kernel_size[3])

        feat_concat = self.concat1([feat1, feat2, feat3, feat4])
        x = self.concat2([x, feat_concat])
        return self.out(x)


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
        self.ppm = PyramidPooling(block_channels[2], out_channels)

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
        x = self.ppm(x)
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

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x