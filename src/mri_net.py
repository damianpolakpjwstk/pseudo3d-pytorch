"""
Network for MRI classification.

It is inspired by the network from "Multi-scale attention-based pseudo-3D convolution neural network
for Alzheimer’s disease diagnosis using structural MRI" paper (https://doi.org/10.1016/j.patcog.2022.108825).
Instead of using transposed convolutions, it uses dilation to achieve similar effect.
"""

import torch
import torchsummary
from torch import nn

from src.blocks import P3DBlockTypeA, P3DBlockTypeB, P3DBlockTypeC


class MultiScaleStem(nn.Module):
    """
    Multiscale stem block for MRI classification. It consists of three Pseudo-3D ResNet-like blocks with different
    dilation rates (1, 2, 3) and a max pooling layer. The output of the block is concatenated in filter dimenstion
    and passed through a 1x1x1 convolution to reduce the number of channels.
    """

    def __init__(self, num_filters: int = 12, pad_size: int = 4, block_type: str = "A",
                 base_channels: int = 16) -> None:
        """
        Initialize the multiscale stem block.

        :param num_filters: number of output filters in the first convolutional layers of each block.
        :param pad_size: size of the padding in the max pooling layer. It is used to reduce the size of the input.
        :param block_type: type of the Pseudo-3D ResNet-like block. It can be "A", "B" or "C".
        :param base_channels: number of output channels of the 1x1x1 convolution.
        """
        super().__init__()
        assert block_type in ("A", "B", "C"), "Block type must be one of the following: A, B, C"
        block = P3DBlockTypeA if block_type == "A" else P3DBlockTypeB if block_type == "B" else P3DBlockTypeC

        self.small_scale_stem = nn.Sequential(
            block(1, num_filters, num_filters, kernel_size=3, stride=2, dilation=1),
            nn.MaxPool3d(kernel_size=pad_size, stride=2, padding=1)
        )
        self.medium_scale_stem = nn.Sequential(
            block(1, num_filters, num_filters, kernel_size=3, stride=2, dilation=2),
            nn.MaxPool3d(kernel_size=pad_size, stride=2, padding=1)
        )
        self.large_scale_stem = nn.Sequential(
            block(1, num_filters, num_filters, kernel_size=3, stride=2, dilation=3),
            nn.MaxPool3d(kernel_size=pad_size, stride=2, padding=1)
        )
        self.downsample = nn.Conv3d(num_filters * 3, base_channels, kernel_size=1, stride=1, padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the multiscale stem block."""
        x_small = self.small_scale_stem(x)
        x_medium = self.medium_scale_stem(x)
        x_large = self.large_scale_stem(x)
        x = torch.cat((x_small, x_medium, x_large), dim=1)
        x = self.downsample(x)
        return x


class MRINet(nn.Module):
    """
    Network for MRI classification.
    """

    def __init__(self, num_classes=2) -> None:
        super().__init__()
        self.base_channels = 16
        self.stem = MultiScaleStem(num_filters=12, pad_size=4, block_type="A", base_channels=self.base_channels)

        self.block1 = nn.Sequential(
            P3DBlockTypeA(16, 16, 32, kernel_size=3, stride=2, dilation=1),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        )

        self.block2 = nn.Sequential(
            P3DBlockTypeB(32, 32, 64, kernel_size=3, stride=2, dilation=1),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        )

        self.block3 = nn.Sequential(
            P3DBlockTypeC(64, 64, 128, kernel_size=3, stride=2, dilation=1),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
        )

        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 1 * 1 * 1, num_classes),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = MRINet(num_classes=2)
    print(torchsummary.summary(model, (1, 224, 224, 224)))
