import torch
import torch.nn as nn
from src.utils.custom_typing import EncoderOutput


class BaseEncoder(nn.Module):
    def __init__(
        self,
        img_size: int,
        in_channels: int,
        num_filters: int,
        kernel_size: int,
        repr_dim: int,
    ):
        """Encoder to extract the representations

        Args:
            img_size (int): [Image size (must be squared size)]
            in_channels (int): Number of input channels
            num_filters (int): Intermediate number of filters
            kernel_size (int): Convolution kernel size
            repr_dim (int): Dimension of the desired representation
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_filters * 2 ** 0,
            kernel_size=kernel_size,
            stride=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=num_filters * 2 ** 0,
            out_channels=num_filters * 2 ** 1,
            kernel_size=kernel_size,
            stride=2,
        )
        self.bn2 = nn.BatchNorm2d(num_features=num_filters * 2 ** 1)
        self.conv3 = nn.Conv2d(
            in_channels=num_filters * 2 ** 1,
            out_channels=num_filters * 2 ** 2,
            kernel_size=kernel_size,
            stride=2,
        )
        self.bn3 = nn.BatchNorm2d(num_features=num_filters * 2 ** 2)
        self.leaky_relu = nn.LeakyReLU()

        self.flatten = nn.Flatten()

        self.dense = nn.Linear(
            in_features=(4 ** 2) * (num_filters * 2 ** 2),
            out_features=repr_dim,
        )

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        """Forward encoder

        Args:
            x (torch.Tensor): Image from a given domain

        Returns:
            EncoderOutput: Representation and feature map
        """
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        feature = self.leaky_relu(x)
        x = self.leaky_relu(x)
        flatten_x = self.flatten(x)
        representation = self.dense(flatten_x)

        return EncoderOutput(representation=representation, feature=feature)
