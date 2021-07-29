import torch
import torch.optim as optim
import torch.nn as nn
from src.models.EDIM import EDIM
from src.utils.custom_typing import (
    ClassifLosses,
    ClassifierOutputs,
    DiscrLosses,
    GenLosses,
    GeneratorOutputs,
    DiscriminatorOutputs,
    EDIMOutputs,
)
from src.losses.EDIM_loss import EDIMLoss
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset
import mlflow
import mlflow.pytorch as mpy


def freeze_grad_and_eval(model: nn.Module):
    """Freeze a given network, disable batch norm and dropout layers

    Args:
        model (nn.Module): [Pretrained shared encoder]
    """
    for param in model.parameters():
        param.requires_grad = False
    model.eval()


class EDIMTrainer:
    """Exclusive Deep Info Max trainer

    Args:
        model (EDIM): Exclusive model backbone
        loss (EDIMLoss): Exclusive loss
        dataset_train (Dataset): Train dataset
        learning_rate (float): Learning rate
        batch_size (int): Batch size
        device (str): Device among cuda/cpu
    """

    def __init__(
        self,
        model: EDIM,
        loss: EDIMLoss,
        dataset_train: Dataset,
        learning_rate: float,
        batch_size: int,
        device: str,
    ):

        self.train_dataloader = DataLoader(dataset_train, batch_size=batch_size)
        self.model = model.to(device)
        self.loss = loss
        self.device = device

        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Network optimizers

        self.optimizer_encoder_x = optim.Adam(
            model.ex_enc_x.parameters(), lr=learning_rate
        )
        self.optimizer_encoder_y = optim.Adam(
            model.ex_enc_y.parameters(), lr=learning_rate
        )
        self.optimizer_local_stat_x = optim.Adam(
            model.local_stat_x.parameters(), lr=learning_rate
        )
        self.optimizer_local_stat_y = optim.Adam(
            model.local_stat_y.parameters(), lr=learning_rate
        )
        self.optimizer_global_stat_x = optim.Adam(
            model.global_stat_x.parameters(), lr=learning_rate
        )
        self.optimizer_global_stat_y = optim.Adam(
            model.global_stat_y.parameters(), lr=learning_rate
        )

        self.optimizer_discriminator_x = optim.Adam(
            model.discriminator_x.parameters(), lr=learning_rate
        )

        self.optimizer_discriminator_y = optim.Adam(
            model.discriminator_y.parameters(), lr=learning_rate
        )

        self.optimizer_digit_bg_classifier = optim.Adam(
            model.digit_bg_classifier.parameters(), lr=learning_rate
        )
        self.optimizer_digit_fg_classifier = optim.Adam(
            model.digit_fg_classifier.parameters(), lr=learning_rate
        )
        self.optimizer_bg_classifier = optim.Adam(
            model.color_bg_classifier.parameters(), lr=learning_rate
        )
        self.optimizer_fg_classifier = optim.Adam(
            model.color_fg_classifier.parameters(), lr=learning_rate
        )

    def update_generator(self, edim_outputs: EDIMOutputs) -> GenLosses:
        """Compute the generator gradient and make an optimisation step

        Args:
            edim_outputs (EDIMOutputs): Exclusive model outputs

        Returns:

            GenLosses: Generator losses
        """
        self.optimizer_encoder_x.zero_grad()
        self.optimizer_encoder_y.zero_grad()

        self.optimizer_local_stat_x.zero_grad()
        self.optimizer_local_stat_y.zero_grad()

        self.optimizer_global_stat_x.zero_grad()
        self.optimizer_global_stat_y.zero_grad()

        losses = self.loss.compute_generator_loss(
            edim_outputs=edim_outputs,
        )
        losses.encoder_loss.backward()
        self.optimizer_encoder_x.step()
        self.optimizer_encoder_y.step()

        self.optimizer_local_stat_x.step()
        self.optimizer_local_stat_y.step()

        self.optimizer_global_stat_x.step()
        self.optimizer_global_stat_y.step()
        return losses

    def update_discriminator(
        self,
        discr_outputs: DiscriminatorOutputs,
    ) -> DiscrLosses:
        """
        Compute the generator gradient and make an optimisation step.
        (With torch gan training should be in 2 stage )
        Args:
            discr_outputs (DiscriminatorOutputs): Discriminator outputs

        Returns:
            DiscrLosses: Discriminator losses
        """
        self.optimizer_discriminator_x.zero_grad()
        self.optimizer_discriminator_y.zero_grad()
        losses = self.loss.compute_discriminator_loss(discr_outputs=discr_outputs)
        losses.gan_loss_d.backward()
        self.optimizer_discriminator_x.step()
        self.optimizer_discriminator_y.step()

        return losses

    def update_classifier(
        self,
        classif_outputs: ClassifierOutputs,
        digit_labels: torch.Tensor,
        color_bg_labels: torch.Tensor,
        color_fg_labels: torch.Tensor,
    ) -> ClassifLosses:
        """Update classifier on generated exclusive reprensentation

        Args:
            classif_outputs (ClassifierOutputs): Classifier ouputs probabilities
            digit_labels (torch.Tensor): Label of the digit
            color_bg_labels (torch.Tensor): Background color label
            color_fg_labels (torch.Tensor): Foreground color label

        Returns:
            ClassifLosses: Classifiers losses
        """
        self.optimizer_digit_bg_classifier.zero_grad()
        self.optimizer_digit_fg_classifier.zero_grad()
        self.optimizer_bg_classifier.zero_grad()
        self.optimizer_fg_classifier.zero_grad()
        losses = self.loss.compute_classif_loss(
            classif_outputs=classif_outputs,
            digit_labels=digit_labels,
            color_bg_labels=color_bg_labels,
            color_fg_labels=color_fg_labels,
        )
        losses.classif_loss.backward()
        self.optimizer_digit_bg_classifier.step()
        self.optimizer_digit_fg_classifier.step()
        self.optimizer_bg_classifier.step()
        self.optimizer_fg_classifier.step()
        return losses

    def train(self, epochs: int, xp_name: str = "test"):
        """Trained excluvise model and log losses and accuracy on Mlflow.

        Args:
            epochs (int): Number of epochs
            xp_name (str, optional): Name of the Mlfow experiment. Defaults to "test".
        """
        mlflow.set_experiment(experiment_name=xp_name)
        with mlflow.start_run() as run:
            mlflow.log_param("Batch size", self.batch_size)
            mlflow.log_param("Learning rate", self.learning_rate)
            mlflow.log_param("Local mutual weight", self.loss.local_mutual_loss_coeff)
            mlflow.log_param("Global mutual weight", self.loss.global_mutual_loss_coeff)
            mlflow.log_param("Discriminator weight", self.loss.disentangling_loss_coeff)
            log_step = 0
            for epoch in tqdm(range(epochs)):
                for idx, train_batch in enumerate(self.train_dataloader):
                    sample = train_batch
                    edim_outputs = self.model.forward_generator(
                        x=sample.bg.to(self.device), y=sample.fg.to(self.device)
                    )
                    gen_losses = self.update_generator(edim_outputs=edim_outputs)

                    discr_outputs = self.model.forward_discriminator(
                        edim_outputs=edim_outputs
                    )
                    discr_losses = self.update_discriminator(
                        discr_outputs=discr_outputs
                    )

                    classif_outputs = self.model.forward_classifier(
                        edim_outputs=edim_outputs
                    )
                    classif_losses = self.update_classifier(
                        classif_outputs=classif_outputs,
                        digit_labels=sample.digit_label.to(self.device),
                        color_bg_labels=sample.bg_label.to(self.device),
                        color_fg_labels=sample.fg_label.to(self.device),
                    )

                    dict_gen_losses = gen_losses._asdict()
                    mlflow.log_metrics(
                        {k: v.item() for k, v in dict_gen_losses.items()}, step=log_step
                    )
                    dict_discr_losses = discr_losses._asdict()
                    mlflow.log_metrics(
                        {k: v.item() for k, v in dict_discr_losses.items()},
                        step=log_step,
                    )
                    dict_classif_losses = classif_losses._asdict()
                    mlflow.log_metrics(
                        {k: v.item() for k, v in dict_classif_losses.items()},
                        step=log_step,
                    )
                    log_step += 1

            encoder_x_path, encoder_y_path = "ex_encoder", "ex_encoder_y"
            mpy.log_state_dict(self.model.ex_enc_x.state_dict(), encoder_x_path)
            mpy.log_state_dict(self.model.ex_enc_y.state_dict(), encoder_y_path)
