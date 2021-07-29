import torch.optim as optim
import torch
from losses.EDIM_loss import EDIMLoss
from src.models.SDIM import SDIM
from src.losses.SDIM_loss import SDIMLoss
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import mlflow
import mlflow.pytorch as mpy

from utils.custom_typing import SDIMOutputs, SDIMLosses


class SDIMTrainer:
    def __init__(
        self,
        model: SDIM,
        loss: SDIMLoss,
        dataset_train: Dataset,
        learning_rate: float,
        batch_size: int,
        device: str,
    ):
        """Shared Deep Info Max trainer

        Args:
            model (SDIM): Shared model backbone
            loss (SDIMLoss): Shared loss
            dataset_train (Dataset): Train dataset
            learning_rate (float): Learning rate
            batch_size (int): Batch size
            device (str): Device among cuda/cpu
        """
        self.train_dataloader = DataLoader(dataset_train, batch_size=batch_size)
        self.model = model.to(device)
        self.loss = loss
        self.device = device

        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Network optimizers
        self.optimizer_encoder_x = optim.Adam(
            model.sh_enc_x.parameters(), lr=learning_rate
        )
        self.optimizer_encoder_y = optim.Adam(
            model.sh_enc_y.parameters(), lr=learning_rate
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

        self.optimizer_digit_classifier = optim.Adam(
            model.digit_classifier.parameters(), lr=learning_rate
        )
        self.optimizer_bg_classifier = optim.Adam(
            model.color_bg_classifier.parameters(), lr=learning_rate
        )
        self.optimizer_fg_classifier = optim.Adam(
            model.color_fg_classifier.parameters(), lr=learning_rate
        )

    def gradient_zero(self):
        """Set all the networks gradient to zero"""
        self.optimizer_encoder_x.zero_grad()
        self.optimizer_encoder_y.zero_grad()

        self.optimizer_local_stat_x.zero_grad()
        self.optimizer_local_stat_y.zero_grad()

        self.optimizer_global_stat_x.zero_grad()
        self.optimizer_global_stat_y.zero_grad()

        self.optimizer_digit_classifier.zero_grad()
        self.optimizer_bg_classifier.zero_grad()
        self.optimizer_fg_classifier.zero_grad()

    def compute_gradient(
        self,
        sdim_output: SDIMOutputs,
        digit_labels: torch.Tensor,
        color_bg_labels: torch.Tensor,
        color_fg_labels: torch.Tensor,
    ) -> SDIMLosses:
        """Compute the SDIM gradient

        Args:
            sdim_output (SDIMOutputs): Shared model outputs
            digit_labels (torch.Tensor): [description]
            color_bg_labels (torch.Tensor): [description]
            color_fg_labels (torch.Tensor): [description]

        Returns:
            SDIMLosses: [Shared model losses value]
        """
        losses = self.loss(
            sdim_outputs=sdim_output,
            digit_labels=digit_labels,
            color_bg_labels=color_bg_labels,
            color_fg_labels=color_fg_labels,
        )
        losses.total_loss.backward()
        return losses

    def gradient_step(self):
        """Make an optimisation step for all the networks"""

        self.optimizer_encoder_x.step()
        self.optimizer_encoder_y.step()

        self.optimizer_local_stat_x.step()
        self.optimizer_local_stat_y.step()

        self.optimizer_global_stat_x.step()
        self.optimizer_global_stat_y.step()

        self.optimizer_digit_classifier.step()
        self.optimizer_bg_classifier.step()
        self.optimizer_fg_classifier.step()

    def train(self, epochs, xp_name="test"):
        """Trained shared model and log losses and accuracy on Mlflow.

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
            mlflow.log_param("L1 weight", self.loss.shared_loss_coeff)
            log_step = 0
            for epoch in tqdm(range(epochs)):
                for idx, train_batch in enumerate(self.train_dataloader):
                    sample = train_batch
                    self.gradient_zero()
                    sdim_outputs = self.model(
                        x=sample.bg.to(self.device), y=sample.fg.to(self.device)
                    )
                    losses = self.compute_gradient(
                        sdim_output=sdim_outputs,
                        digit_labels=sample.digit_label.to(self.device),
                        color_bg_labels=sample.bg_label.to(self.device),
                        color_fg_labels=sample.fg_label.to(self.device),
                    )
                    dict_losses = losses._asdict()
                    mlflow.log_metrics(
                        {k: v.item() for k, v in dict_losses.items()}, step=log_step
                    )
                    log_step += 1
                    self.gradient_step()

            encoder_x_path, encoder_y_path = "sh_encoder_x", "sh_encoder_y"
            mpy.log_state_dict(self.model.sh_enc_x.state_dict(), encoder_x_path)
            mpy.log_state_dict(self.model.sh_enc_y.state_dict(), encoder_y_path)
