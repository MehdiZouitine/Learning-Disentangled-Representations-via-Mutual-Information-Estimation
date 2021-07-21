import torch.nn as nn
from src.utils.custom_typing import EncoderOutput


class BaseEncoder(nn.Module):
    def __init__(self, img_size, in_channels, num_filters, kernel_size, repr_dim):
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

    def forward(self, x):
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


if __name__ == "__main__":
    import torch

    img_size = 28
    x = torch.zeros((64, 3, img_size, img_size))
    enc2 = BaseEncoder(
        img_size=img_size,
        in_channels=3,
        num_filters=64,
        kernel_size=4,
        repr_dim=64,
    )
    print(enc2(x)[1].shape)
