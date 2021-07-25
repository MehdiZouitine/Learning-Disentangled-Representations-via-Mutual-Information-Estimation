import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, feature_dim, output_dim, units=15) -> None:
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


if __name__ == "__main__":
    import torch

    # from src..encoder import Bas

    # img_size = 128
    # x = torch.zeros((64, 3, img_size, img_size))
    # enc_sh = SharedEncoder(
    #     img_size=img_size, in_channels=3, num_filters=16, kernel_size=1, shared_dim=64
    # )
    # enc_ex = ExclusiveEncoder(
    #     img_size=img_size,
    #     in_channels=3,
    #     num_filters=16,
    #     kernel_size=3,
    #     exclusive_dim=64,
    # )

    # sh_repr, sh_feature = enc_sh(x)
    # ex_repr, ex_feature = enc_ex(x)
    # clf = Classifier(feature_dim=sh_repr.size(1), output_dim=10)
    # print(clf(sh_repr).shape)
