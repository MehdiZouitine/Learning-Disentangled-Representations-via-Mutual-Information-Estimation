import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, shared_dim, exclusive_dim):
        super().__init__()
        self.dense1 = nn.Linear(
            in_features=shared_dim + exclusive_dim, out_features=1000
        )
        self.dense2 = nn.Linear(in_features=1000, out_features=200)
        self.dense3 = nn.Linear(in_features=200, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, concat_repr):
        x = self.dense1(concat_repr)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        logit = self.dense3(x)

        return logit


if __name__ == "__main__":
    import torch
    from src.neural_networks.encoder import BaseEncoder

    img_size = 128
    x = torch.zeros((64, 3, img_size, img_size))
    enc_sh = BaseEncoder(
        img_size=img_size, in_channels=3, num_filters=16, kernel_size=1, repr_dim=64
    )
    enc_ex = BaseEncoder(
        img_size=img_size,
        in_channels=3,
        num_filters=16,
        kernel_size=1,
        repr_dim=64,
    )

    sh_repr, sh_feature = enc_sh(x)
    ex_repr, ex_feature = enc_ex(x)
    merge_repr = torch.cat([sh_repr, ex_repr], dim=1)
    discriminator = Discriminator(
        shared_dim=sh_repr.size(1), exclusive_dim=ex_repr.size(1)
    )
    print(discriminator(concat_repr=merge_repr).shape)
