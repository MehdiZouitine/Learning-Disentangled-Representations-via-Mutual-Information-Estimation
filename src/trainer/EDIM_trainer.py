import torch.optim as optim
from src.models.EDIM import EDIM
from src.utils.custom_typing import GeneratorOutputs, DiscriminatorOutputs, EDIMOutputs
from src.losses.EDIM_loss import EDIMLoss
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import mlflow
import mlflow.pytorch as mpy


def freeze_grad_and_eval(model):
    for param in model.parameters():
        param.requires_grad = False
    model.eval()


class EDIMTrainer:
    def __init__(
        self,
        model: EDIM,
        loss: EDIMLoss,
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

    def update_generator(self, edim_outputs: EDIMOutputs):
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
    ):
        self.optimizer_discriminator_x.zero_grad()
        self.optimizer_discriminator_y.zero_grad()
        losses = self.loss.compute_discriminator_loss(discr_outputs=discr_outputs)
        losses.gan_loss_d.backward()
        self.optimizer_discriminator_x.step()
        self.optimizer_discriminator_y.step()

        return losses

    def update_classifier(
        self,
        classif_outputs,
        digit_labels,
        color_bg_labels,
        color_fg_labels,
    ):
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

    def train(self, epochs, experiment_name="test"):
        mlflow.set_experiment(experiment_name=experiment_name)
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


if __name__ == "__main__":
    # 0.5 1 01
    from src.utils.colored_mnist_dataloader import ColoredMNISTDataset
    from src.neural_networks.encoder import BaseEncoder
    import torch

    trained_enc_x = BaseEncoder(
        img_size=28, in_channels=3, num_filters=64, kernel_size=4, repr_dim=64
    )
    trained_enc_y = BaseEncoder(
        img_size=28, in_channels=3, num_filters=64, kernel_size=4, repr_dim=64
    )
    trained_enc_x.load_state_dict(
        torch.load(
            "/home/mehdi.zouitine/spaghetti/mlruns/1/7713ce12821b4d41af9d68719c4bceb9/artifacts/sh_encoder_x.pth/state_dict.pth"
        )
    )
    trained_enc_y.load_state_dict(
        torch.load(
            "/home/mehdi.zouitine/spaghetti/mlruns/1/7713ce12821b4d41af9d68719c4bceb9/artifacts/sh_encoder_y.pth/state_dict.pth"
        )
    )

    freeze_grad_and_eval(trained_enc_x)
    freeze_grad_and_eval(trained_enc_y)
    edim = EDIM(
        img_size=28,
        channels=3,
        shared_dim=64,
        exclusive_dim=8,
        trained_encoder_x=trained_enc_x,
        trained_encoder_y=trained_enc_y,
    )
    loss = EDIMLoss(
        local_mutual_loss_coeff=1,
        global_mutual_loss_coeff=0.5,
        disentangling_loss_coeff=0.1,
    )
    train_dataset = ColoredMNISTDataset(train=True)
    test_dataset = ColoredMNISTDataset(train=False)
    trainer = EDIMTrainer(
        dataset_train=train_dataset,
        dataset_test=train_dataset,
        model=edim,
        loss=loss,
        learning_rate=1e-4,
        batch_size=64,
        device="cuda",
    )
    trainer.train(epochs=11, experiment_name="_Get_exclusive_repr")