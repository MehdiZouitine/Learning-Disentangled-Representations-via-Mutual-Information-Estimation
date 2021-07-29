import argparse
import numpy as np
import os
import random
import torch
import torch.nn as nn
import glob
import ruamel.yaml as yaml
from src.losses.EDIM_loss import EDIMLoss
from src.models.EDIM import EDIM
from src.utils.colored_mnist_dataloader import ColoredMNISTDataset
from src.trainer.EDIM_trainer import EDIMTrainer, freeze_grad_and_eval
from src.neural_networks.encoder import BaseEncoder


def run(
    xp_name: str,
    conf_path: str,
    data_base_folder: str,
    trained_enc_x_path: str,
    trained_enc_y_path: str,
    seed: int = None,
):
    with open(conf_path, "r") as f:
        conf = yaml.safe_load(f)
    if seed is not None:
        seed = seed
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)

    TRAINING_PARAM = conf["training_param"]
    MODEL_PARAM = conf["model_param"]
    LOSS_PARAM = conf["loss_param"]
    SHARED_PARAM = conf["shared_param"]

    trained_enc_x = BaseEncoder(
        img_size=SHARED_PARAM["img_size"],
        in_channels=MODEL_PARAM["channels"],
        num_filters=64,
        kernel_size=4,
        repr_dim=SHARED_PARAM["shared_dim"],
    )
    trained_enc_y = BaseEncoder(
        img_size=SHARED_PARAM["img_size"],
        in_channels=MODEL_PARAM["channels"],
        num_filters=64,
        kernel_size=4,
        repr_dim=SHARED_PARAM["shared_dim"],
    )
    trained_enc_x.load_state_dict(torch.load(trained_enc_x_path))
    trained_enc_y.load_state_dict(torch.load(trained_enc_y_path))
    freeze_grad_and_eval(trained_enc_x)
    freeze_grad_and_eval(trained_enc_y)

    edim = EDIM(
        img_size=MODEL_PARAM["img_size"],
        channels=MODEL_PARAM["channels"],
        shared_dim=SHARED_PARAM["shared_dim"],
        exclusive_dim=MODEL_PARAM["exclusive_dim"],
        trained_encoder_x=trained_enc_x,
        trained_encoder_y=trained_enc_y,
    )
    loss = EDIMLoss(
        local_mutual_loss_coeff=LOSS_PARAM["local_mutual_loss_coeff"],
        global_mutual_loss_coeff=LOSS_PARAM["global_mutual_loss_coeff"],
        disentangling_loss_coeff=LOSS_PARAM["disentangling_loss_coeff"],
    )

    train_dataset = ColoredMNISTDataset(train=True, data_folder=data_base_folder)

    device = TRAINING_PARAM["device"]
    learning_rate = TRAINING_PARAM["learning_rate"]
    batch_size = TRAINING_PARAM["batch_size"]
    epochs = TRAINING_PARAM["epochs"]
    trainer = EDIMTrainer(
        dataset_train=train_dataset,
        model=edim,
        loss=loss,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=device,
    )
    trainer.train(epochs=epochs, xp_name=xp_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Learning Disentangled Representations via Mutual Information Estimation"
    )
    parser.add_argument(
        "--xp_name",
        nargs="?",
        type=str,
        default="Shared_training",
        help="Mlflow experiment name",
    )
    parser.add_argument(
        "--conf_path", nargs="?", type=str, default=None, help="Configuration file"
    )
    parser.add_argument(
        "--data_base_folder", nargs="?", type=str, default=None, help="Data folder"
    )

    parser.add_argument("--seed", nargs="?", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--trained_enc_x_path",
        nargs="?",
        type=str,
        default=None,
        help="Pretrained shared encoder x",
    )
    parser.add_argument(
        "--trained_enc_y_path",
        nargs="?",
        type=str,
        default=None,
        help="Pretrained shared encoder y",
    )

    args = parser.parse_args()
    xp_name = args.xp_name
    conf_path = args.conf_path
    data_base_folder = args.data_base_folder
    seed = args.seed
    trained_enc_x_path = args.trained_enc_x_path
    trained_enc_y_path = args.trained_enc_y_path
    run(
        xp_name=xp_name,
        conf_path=conf_path,
        data_base_folder=data_base_folder,
        trained_enc_x_path=args.trained_enc_x_path,
        trained_enc_y_path=args.trained_enc_y_path,
        seed=seed,
    )
