import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.custom_typing import GanLossOutput


class DJSLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, T, T_prime):
        joint_expectation = (-F.softplus(-T)).mean(dim=0)
        marginal_expectation = F.softplus(T_prime).mean(dim=0)
        mutual_info = joint_expectation - marginal_expectation

        return -mutual_info


class GanLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, real_logits, fake_logits):
        discriminator_real = F.binary_cross_entropy_with_logits(
            input=real_logits, target=torch.ones_like(real_logits)
        )
        discriminator_fake = F.binary_cross_entropy_with_logits(
            input=fake_logits, target=torch.zeros_like(fake_logits)
        )
        discriminator_loss = discriminator_real.mean(dim=0) + discriminator_fake.mean(
            dim=0
        )

        generator_loss = F.binary_cross_entropy_with_logits(
            input=fake_logits, target=torch.ones_like(fake_logits)
        )
        return GanLossOutput(generator=generator_loss, discriminator=discriminator_loss)


if __name__ == "__main__":
    djs = DJSLoss()
    gl = GanLoss()
    x = torch.ones((64, 1))
    y = torch.zeros((64, 1))
    print(gl(x, y))
    print(djs(x, y))
