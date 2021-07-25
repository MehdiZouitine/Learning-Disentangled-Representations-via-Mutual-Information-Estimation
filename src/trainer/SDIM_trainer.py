import torch.optim as optim
from src.models.SDIM import SDIM
from src.losses.SDIM_loss import SDIMLoss
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import mlflow
import mlflow.pytorch as mpy


class SDIMTrainer:
    def __init__(
        self,
        model: SDIM,
        loss: SDIMLoss,
        dataset_train,
        dataset_test,
        learning_rate,
        batch_size,
        device,
    ):
        self.train_dataloader = DataLoader(dataset_train, batch_size=batch_size)
        self.test_dataloader = DataLoader(dataset_test, batch_size=batch_size)
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
        self, sdim_output, digit_labels, color_bg_labels, color_fg_labels
    ):
        losses = self.loss(
            sdim_outputs=sdim_output,
            digit_labels=digit_labels,
            color_bg_labels=color_bg_labels,
            color_fg_labels=color_fg_labels,
        )
        losses.total_loss.backward()
        return losses

    def gradient_step(self):

        self.optimizer_encoder_x.step()
        self.optimizer_encoder_y.step()

        self.optimizer_local_stat_x.step()
        self.optimizer_local_stat_y.step()

        self.optimizer_global_stat_x.step()
        self.optimizer_global_stat_y.step()

        self.optimizer_digit_classifier.step()
        self.optimizer_bg_classifier.step()
        self.optimizer_fg_classifier.step()

    def train(self, epochs, experiment_name="test"):
        mlflow.set_experiment(experiment_name=experiment_name)
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

            encoder_x_path, encoder_y_path = "sh_encoder_x.pth", "sh_encoder_y.pth"
            mpy.log_state_dict(self.model.sh_enc_x.state_dict(), encoder_x_path)
            mpy.log_state_dict(self.model.sh_enc_y.state_dict(), encoder_y_path)


if __name__ == "__main__":
    # 0.5 1 01
    from src.utils.colored_mnist_dataloader import ColoredMNISTDataset

    sdim = SDIM(img_size=28, channels=3, shared_dim=64, switched=True)
    loss = SDIMLoss(
        local_mutual_loss_coeff=1, global_mutual_loss_coeff=0.5, shared_loss_coeff=0.1
    )

    train_dataset = ColoredMNISTDataset(train=True)
    test_dataset = ColoredMNISTDataset(train=False)
    trainer = SDIMTrainer(
        dataset_train=train_dataset,
        dataset_test=train_dataset,
        model=sdim,
        loss=loss,
        learning_rate=1e-4,
        batch_size=64,
        device="cuda",
    )
    trainer.train(epochs=16, experiment_name="Get_shared_repr")
