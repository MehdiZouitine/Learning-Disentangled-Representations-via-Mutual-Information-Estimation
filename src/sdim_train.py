import argparse
import numpy as np
import os
import random
import torch
import torch.nn as nn
import glob
import ruamel.yaml as yaml
from src.losses.SDIM_loss import SDIMLoss
from src.models.SDIM import SDIM
from src.utils.colored_mnist_dataloader import ColoredMNISTDataset
from src.trainer.SDIM_trainer import SDIMTrainer


def run(
    xp_name: str,
    conf_path: str,
    data_base_folder: str,
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

    sdim = SDIM(
        img_size=MODEL_PARAM["img_size"],
        channels=MODEL_PARAM["channels"],
        shared_dim=MODEL_PARAM["shared_dim"],
        switched=MODEL_PARAM["switched"],
    )
    loss = SDIMLoss(
        local_mutual_loss_coeff=LOSS_PARAM["local_mutual_loss_coeff"],
        global_mutual_loss_coeff=LOSS_PARAM["global_mutual_loss_coeff"],
        shared_loss_coeff=LOSS_PARAM["shared_loss_coeff"],
    )

    train_dataset = ColoredMNISTDataset(train=True, data_folder=data_base_folder)

    device = TRAINING_PARAM["device"]
    learning_rate = TRAINING_PARAM["learning_rate"]
    batch_size = TRAINING_PARAM["batch_size"]
    epochs = TRAINING_PARAM["epochs"]
    trainer = SDIMTrainer(
        dataset_train=train_dataset,
        model=sdim,
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

    args = parser.parse_args()
    xp_name = args.xp_name
    conf_path = args.conf_path
    data_base_folder = args.data_base_folder
    seed = args.seed

    run(
        xp_name=xp_name,
        conf_path=conf_path,
        data_base_folder=data_base_folder,
        seed=seed,
    )
