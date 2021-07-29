import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, feature_dim: int, output_dim, units: int = 15) -> None:
        """Simple dense classifier

        Args:
            feature_dim (int): [Number of input feature]
            output_dim ([type]): [Number of classes]
            units (int, optional): [Intermediate layers dimension]. Defaults to 15.
        """
        super().__init__()
        self.dense1 = nn.Linear(in_features=feature_dim, out_features=units)
        self.bn1 = nn.BatchNorm1d(num_features=units)
        self.dense2 = nn.Linear(in_features=units, out_features=output_dim)
        self.bn2 = nn.BatchNorm1d(num_features=output_dim)
        self.dense3 = nn.Linear(in_features=output_dim, out_features=output_dim)
        self.bn3 = nn.BatchNorm1d(num_features=output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dense1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dense3(x)
        logits = self.bn3(x)
        return logits