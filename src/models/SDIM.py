import torch
import torch.nn as nn
from src.neural_networks.encoder import BaseEncoder
from src.neural_networks.statistics_network import (
    LocalStatisticsNetwork,
    GlobalStatisticsNetwork,
)
from src.neural_networks.classifier import Classifier
from src.utils.colored_mnist_dataloader import ColoredMNISTDataset


class SDIM(nn.Module):
    def __init__(self, img_size, channels, shared_dim, switched):
        super().__init__()

        self.img_size = img_size
        self.channels = channels
        self.shared_dim = shared_dim
        self.switched = switched

        self.img_feature_size = 4
        self.img_feature_channels = 256
        self.switched = switched

        # Encoders
        self.sh_enc_x = BaseEncoder(
            img_size=img_size,
            in_channels=channels,
            num_filters=64,
            kernel_size=4,
            repr_dim=shared_dim,
        )

        self.sh_enc_y = BaseEncoder(
            img_size=img_size,
            in_channels=channels,
            num_filters=64,
            kernel_size=4,
            repr_dim=shared_dim,
        )
        # Local statistics network
        self.local_stat_x = LocalStatisticsNetwork(
            img_feature_channels=self.img_feature_channels
        )

        self.local_stat_y = LocalStatisticsNetwork(
            img_feature_channels=self.img_feature_channels
        )

        # Global statistics network
        self.global_stat_x = GlobalStatisticsNetwork(
            feature_map_size=self.img_feature_size,
            feature_map_channels=self.img_feature_channels,
            latent_dim=self.shared_dim,
        )

        self.global_stat_y = GlobalStatisticsNetwork(
            feature_map_size=self.img_feature_size,
            feature_map_channels=self.img_feature_channels,
            latent_dim=self.shared_dim,
        )

        # Metric nets
        self.digit_classifier = Classifier(feature_dim=shared_dim, output_dim=10)
        self.color_bg_classifier = Classifier(feature_dim=shared_dim, output_dim=12)
        self.color_fg_classifier = Classifier(feature_dim=shared_dim, output_dim=12)

    def forward(self, x, y):
        shared_x, shared_M_x = self.sh_enc_x(x)
        shared_y, shared_M_y = self.sh_enc_y(x)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    sdim = SDIM(img_size=28, channels=3, shared_dim=10, switched=True)
    d = ColoredMNISTDataset(train=True)
    train_dataloader = DataLoader(d, batch_size=3, shuffle=True)
    for elem in train_dataloader:
        print(sdim(elem.fg, elem.bg))
        a = input()
