import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.custom_typing import GanLossOutput, Tuple


class ClassifLoss(nn.Module):
    """Classifier loss"""

    @staticmethod
    def accuracy(y_pred, target):
        return torch.sum(y_pred == target).float().mean()

    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def __call__(
        self, y_pred: torch.tensor, target: torch.tensor
    ) -> Tuple[float, float]:
        """Compute cross entropy loss

        Args:
            y_pred (torch.tensor): Classifier prediction
            target (torch.tensor): Ground truth

        Returns:
            Tuple[float, float]: Error and accuracy over the current batch
        """
        batch_size = y_pred.size(0)

        classif_error = self.cross_entropy(
            F.softmax(y_pred, dim=1), target.long()
        ).mean()
        accuracy = self.accuracy(y_pred=torch.argmax(y_pred, dim=1), target=target)
        return classif_error, accuracy / batch_size


class DJSLoss(nn.Module):
    """Jensen Shannon Divergence loss"""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, T: torch.tensor, T_prime: torch.tensor) -> float:
        """Estimator of the Jensen Shannon Divergence see paper equation (2)

        Args:
            T (torch.tensor): Statistique network estimation from the marginal distribution P(x)P(z)
            T_prime (torch.tensor): Statistique network estimation from the joint distribution P(xz)

        Returns:
            float: DJS estimation value
        """
        joint_expectation = (-F.softplus(-T)).mean()
        marginal_expectation = F.softplus(T_prime).mean()
        mutual_info = joint_expectation - marginal_expectation

        return -mutual_info


class DiscriminatorLoss(nn.Module):
    """Basic discriminator GAN loss """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, real_logits: torch.tensor, fake_logits: torch.tensor) -> float:
        """Discriminator loss gan

        Args:
            real_logits (torch.tensor): Sample from the real distribution here from P(Sx)P(Ex)
            fake_logits (torch.tensor): Sample from the fake (generated) distribution here from P(SxEx)

        Returns:
            float: Discriminator loss value
        """

        # Discriminator should predict real logits as logits from the real distribution
        discriminator_real = F.binary_cross_entropy_with_logits(
            input=real_logits, target=torch.ones_like(real_logits)
        )
        # Discriminator should predict fake logits as logits from the generated distribution
        discriminator_fake = F.binary_cross_entropy_with_logits(
            input=fake_logits, target=torch.zeros_like(fake_logits)
        )
        discriminator_loss = discriminator_real.mean() + discriminator_fake.mean()

        return discriminator_loss


class GeneratorLoss(nn.Module):
    """Basic generator GAN loss """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, fake_logits: torch.tensor) -> float:
        """Generator loss

        Args:
            fake_logits (torch.tensor): Sample from the fake (generated) distribution here from P(SxEx)

        Returns:
            float: Generator loss value
        """
        # Discriminator should generate fake logits that fool the discriminator
        generator_loss = F.binary_cross_entropy_with_logits(
            input=fake_logits, target=torch.ones_like(fake_logits)
        )
        return generator_loss
