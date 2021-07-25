import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.custom_typing import GanLossOutput


class ClassifLoss(nn.Module):
    @staticmethod
    def accuracy(y_pred, target):
        return torch.sum(y_pred == target).float().mean()

    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def __call__(self, y_pred, target):
        batch_size = y_pred.size(0)

        classif_error = self.cross_entropy(
            F.softmax(y_pred, dim=1), target.long()
        ).mean()
        accuracy = self.accuracy(y_pred=torch.argmax(y_pred, dim=1), target=target)
        return classif_error, accuracy / batch_size


class DJSLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, T, T_prime):
        joint_expectation = (-F.softplus(-T)).mean()
        marginal_expectation = F.softplus(T_prime).mean()
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
        discriminator_loss = discriminator_real.mean() + discriminator_fake.mean()

        generator_loss = F.binary_cross_entropy_with_logits(
            input=fake_logits, target=torch.ones_like(fake_logits)
        )
        return GanLossOutput(discriminator=discriminator_loss, generator=generator_loss)


if __name__ == "__main__":
    djs = DJSLoss()
    gl = GanLoss()
    clf = ClassifLoss()
    x = torch.zeros((64, 10))
    x[0:50, 0] = 1
    x[50:, 2] = 1
    y = torch.zeros((64))
    print(clf(x, y))
    # print(gl(x, y))
    # print(djs(x, y))
