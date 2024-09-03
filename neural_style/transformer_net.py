"""
1)  The original paper's architecture used batch normalization. However, the author later
    mentioned that instance normalization led to better results. We also use instance normalization
    but keep an option to use the original batch normalization.

        More details on instance normalization:
            Instance Normalization: The Missing Ingredient for Fast Stylization
            ref = https://arxiv.org/abs/1607.08022

2)  No tanh function is used after the last layer of the model because pixel value scaling
    and normalization is done during training using the VGG16 mean and std instead of in the model.

3)  For the upsampling layers, we upsample using nearest-neighbor up-sampling followed by a convolution
    instead of a tranpose convolution. This helps avoid the checkerboard artifact.

        More details on this method:
            Deconvolution and Checkerboard Artifacts
            ref = https://distill.pub/2016/deconv-checkerboard/

        Because we had never seen this method before, we referenced outside code on how to
        do the up-sampling followed by a convolution:
            ref = https://github.com/pytorch/examples/blob/main/fast_neural_style/neural_style/transformer_net.py
"""

import torch.nn as nn


class TransformerNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Each downsample layer: convolution -> normalization -> Relu activation
        self.down_sample = nn.Sequential(
            DownsampleConvLayer(3, 32, 9, 1),  # 32 × 9 × 9 conv, stride 1
            DownsampleConvLayer(32, 64, 3, 2),  # 64 × 3 × 3 conv, stride 2
            DownsampleConvLayer(64, 128, 3, 2)  # 128 × 3 × 3 conv, stride 2
        )

        # Each residual layer:  convolution -> normalization -> Relu activation
        #                       --> convolution -> normalization
        self.residual = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )

        # Each upsample layer: interpolation -> convolution -> normalization -> Relu activation
        self.up_sample = nn.Sequential(
            UpsampleConvLayer(128, 64, 3, 2),
            UpsampleConvLayer(64, 32, 3, 2),
            nn.Conv2d(32, 3, 9, 1, padding=9 // 2, padding_mode='reflect')
        )

    def forward(self, x):
        out = self.down_sample(x)
        out = self.residual(out)
        out = self.up_sample(out)
        return out


class DownsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, instance_norm=True):
        super(DownsampleConvLayer, self).__init__()

        """
        Output dimensions H_out and W_out (both are equal in our case) of a convolution:
            H_out = ⌊(H_in + 2 * padding − dilation * (kernel_size − 1) − 1 / stride) + 1⌋

        Thus, padding of ⌊kernel size / 2⌋ is necessary to maintain the output sizes 
        described in the paper's architecture. We decided to keep the default padding
        type of "reflect" as mentioned in the supplementary material of the paper.
        """
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                padding=kernel_size // 2, padding_mode='reflect')
        if instance_norm:
            self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.norm = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.norm(self.conv2d(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels, instance_norm=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=3 // 2, padding_mode='reflect')
        if instance_norm:
            self.norm1 = nn.InstanceNorm2d(channels, affine=True)
            self.norm2 = nn.InstanceNorm2d(channels, affine=True)
        else:
            self.norm1 = nn.BatchNorm2d(channels)
            self.norm2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=3 // 2, padding_mode='reflect')
        self.relu = nn.ReLU()

    def forward(self, x):
        orig = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return out + orig


class UpsampleConvLayer(nn.Module):
    """
        Code is influenced from https://github.com/pytorch/examples/blob/main/fast_neural_style/neural_style/utils.py
        Nearest-neighbor up-sampling followed by a convolution
        Appears to give better results than learned up-sampling aka transposed conv (avoids the checkerboard artifact)

        Initially proposed on distill pub: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor):
        super().__init__()
        self.upsampling_factor = upsampling_factor
        self.conv = DownsampleConvLayer(in_channels, out_channels, kernel_size, 1)

    def forward(self, x):
        if self.upsampling_factor > 1:
            x = nn.functional.interpolate(x, scale_factor=self.upsampling_factor, mode='nearest')
        return self.conv(x)
