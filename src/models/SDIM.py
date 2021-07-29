import torch
import torch.nn as nn
from src.neural_networks.encoder import BaseEncoder
from src.neural_networks.statistics_network import (
    LocalStatisticsNetwork,
    GlobalStatisticsNetwork,
    tile_and_concat,
)
from src.utils.custom_typing import SDIMOutputs
from src.neural_networks.classifier import Classifier
from src.utils.colored_mnist_dataloader import ColoredMNISTDataset


class SDIM(nn.Module):
    def __init__(self, img_size: int, channels: int, shared_dim: int, switched: bool):
        """Shared Deep Info Max model. Extract the shared information from the images

        Args:
            img_size (int): Image size (must be squared size)
            channels (int): Number of inputs channels
            shared_dim (int): Dimension of the desired shared representation
            switched (bool): True to use cross mutual information, see paper equation (5)
        """
        super().__init__()

        self.img_size = img_size
        self.channels = channels
        self.shared_dim = shared_dim
        self.switched = switched

        self.img_feature_size = 4
        self.img_feature_channels = 256

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
            img_feature_channels=self.img_feature_channels + self.shared_dim
        )

        self.local_stat_y = LocalStatisticsNetwork(
            img_feature_channels=self.img_feature_channels + self.shared_dim
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

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> SDIMOutputs:
        """Forward pass of the shared model

        Args:
            x (torch.Tensor): Image from domain X
            y (torch.Tensor): Image from domain Y

        Returns:
            SDIMOutputs: Outputs of the SDIM model
        """

        # Get the shared and exclusive features from x and y
        shared_x, M_x = self.sh_enc_x(x)
        shared_y, M_y = self.sh_enc_y(y)
        # Shuffle M to create M'
        M_x_prime = torch.cat([M_x[1:], M_x[0].unsqueeze(0)], dim=0)
        M_y_prime = torch.cat([M_y[1:], M_y[0].unsqueeze(0)], dim=0)

        # Tile the exclusive representations (R) of each image and get the cross representations
        if self.switched:  # Shared representations are switched
            R_x_y = shared_x
            R_y_x = shared_y
        else:  # Shared representations are not switched
            R_x_y = shared_y
            R_y_x = shared_x

        # Global mutual information estimation
        global_mutual_M_R_x = self.global_stat_x(M_x, R_y_x)
        global_mutual_M_R_x_prime = self.global_stat_x(M_x_prime, R_y_x)

        global_mutual_M_R_y = self.global_stat_y(M_y, R_x_y)
        global_mutual_M_R_y_prime = self.global_stat_y(M_y_prime, R_x_y)

        # Merge the feature map with the shared representation

        concat_M_R_x = tile_and_concat(tensor=M_x, vector=R_y_x)
        concat_M_R_x_prime = tile_and_concat(tensor=M_x_prime, vector=R_y_x)

        concat_M_R_y = tile_and_concat(tensor=M_y, vector=R_x_y)
        concat_M_R_y_prime = tile_and_concat(tensor=M_y_prime, vector=R_x_y)

        # Local mutual information estimation

        local_mutual_M_R_x = self.local_stat_x(concat_M_R_x)
        local_mutual_M_R_x_prime = self.local_stat_x(concat_M_R_x_prime)
        local_mutual_M_R_y = self.local_stat_y(concat_M_R_y)
        local_mutual_M_R_y_prime = self.local_stat_y(concat_M_R_y_prime)

        # Stop the gradient and compute classification task
        digit_logits = self.digit_classifier(shared_x.detach())
        color_bg_logits = self.color_bg_classifier(shared_x.detach())
        color_fg_logits = self.color_fg_classifier(shared_x.detach())

        return SDIMOutputs(
            global_mutual_M_R_x=global_mutual_M_R_x,
            global_mutual_M_R_x_prime=global_mutual_M_R_x_prime,
            global_mutual_M_R_y=global_mutual_M_R_y,
            global_mutual_M_R_y_prime=global_mutual_M_R_y_prime,
            local_mutual_M_R_x=local_mutual_M_R_x,
            local_mutual_M_R_x_prime=local_mutual_M_R_x_prime,
            local_mutual_M_R_y=local_mutual_M_R_y,
            local_mutual_M_R_y_prime=local_mutual_M_R_y_prime,
            digit_logits=digit_logits,
            color_bg_logits=color_bg_logits,
            color_fg_logits=color_fg_logits,
            shared_x=shared_x,
            shared_y=shared_y,
        )
