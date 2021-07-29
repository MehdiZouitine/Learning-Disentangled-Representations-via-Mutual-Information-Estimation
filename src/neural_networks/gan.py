import torch.nn as nn
import torch


class Discriminator(nn.Module):
    def __init__(self, shared_dim: int, exclusive_dim: int):
        """Dense discriminator

        Args:
            shared_dim (int): [Dimension of the shared representation]
            exclusive_dim (int): [Dimension of the exclusive representation]
        """
        super().__init__()
        self.dense1 = nn.Linear(
            in_features=shared_dim + exclusive_dim, out_features=1000
        )
        self.dense2 = nn.Linear(in_features=1000, out_features=200)
        self.dense3 = nn.Linear(in_features=200, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, concat_repr: torch.Tensor) -> torch.Tensor:
        """Forward discriminator using the shared and the exclusive representation

        Args:
            concat_repr (torch.Tensor): Shared & exclusive representation

        Returns:
            torch.Tensor: Probability that the data are fake or real
        """
        x = self.dense1(concat_repr)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        logit = self.dense3(x)

        return logit
