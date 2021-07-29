from typing import NamedTuple, Tuple
import torch


class GanLossOutput(NamedTuple):
    discriminator: torch.Tensor
    generator: torch.Tensor


class EncoderOutput(NamedTuple):
    representation: torch.Tensor
    feature: torch.Tensor


class ColoredMNISTData(NamedTuple):
    fg: torch.tensor
    bg: torch.tensor
    fg_label: torch.tensor
    bg_label: torch.tensor
    digit_label: torch.tensor


class SDIMOutputs(NamedTuple):
    global_mutual_M_R_x: torch.tensor
    global_mutual_M_R_x_prime: torch.tensor
    global_mutual_M_R_y: torch.tensor
    global_mutual_M_R_y_prime: torch.tensor
    local_mutual_M_R_x: torch.tensor
    local_mutual_M_R_x_prime: torch.tensor
    local_mutual_M_R_y: torch.tensor
    local_mutual_M_R_y_prime: torch.tensor
    digit_logits: torch.tensor
    color_bg_logits: torch.tensor
    color_fg_logits: torch.tensor
    shared_x: torch.tensor
    shared_y: torch.tensor


class EDIMOutputs(NamedTuple):
    global_mutual_M_R_x: torch.tensor
    global_mutual_M_R_x_prime: torch.tensor
    global_mutual_M_R_y: torch.tensor
    global_mutual_M_R_y_prime: torch.tensor
    local_mutual_M_R_x: torch.tensor
    local_mutual_M_R_x_prime: torch.tensor
    local_mutual_M_R_y: torch.tensor
    local_mutual_M_R_y_prime: torch.tensor
    shared_x: torch.tensor
    shared_y: torch.tensor
    fake_x: torch.tensor
    fake_y: torch.tensor
    R_y_x: torch.tensor
    R_x_y: torch.tensor
    shuffle_x: torch.tensor
    shuffle_y: torch.tensor
    exclusive_x: torch.tensor
    exclusive_y: torch.tensor


class SDIMLosses(NamedTuple):
    total_loss: torch.tensor
    encoder_loss: torch.tensor
    local_mutual_loss: torch.tensor
    global_mutual_loss: torch.tensor
    shared_loss: torch.tensor
    digit_classif_loss: torch.tensor
    color_bg_classif_loss: torch.tensor
    color_fg_classif_loss: torch.tensor
    digit_accuracy: torch.tensor
    color_bg_accuracy: torch.tensor
    color_fg_accuracy: torch.tensor


class GenLosses(NamedTuple):
    encoder_loss: torch.tensor
    local_mutual_loss: torch.tensor
    global_mutual_loss: torch.tensor
    gan_loss_g: torch.tensor


class ClassifLosses(NamedTuple):
    classif_loss: torch.tensor
    digit_bg_classif_loss: torch.tensor
    digit_fg_classif_loss: torch.tensor
    color_bg_classif_loss: torch.tensor
    color_fg_classif_loss: torch.tensor
    digit_bg_accuracy: torch.tensor
    digit_fg_accuracy: torch.tensor
    color_bg_accuracy: torch.tensor
    color_fg_accuracy: torch.tensor


class DiscrLosses(NamedTuple):
    gan_loss_d: torch.tensor


class GeneratorOutputs(NamedTuple):
    real_x: torch.tensor
    fake_x: torch.tensor
    real_y: torch.tensor
    fake_y: torch.tensor
    exclusive_x: torch.tensor
    exclusive_y: torch.tensor


class DiscriminatorOutputs(NamedTuple):
    disentangling_information_x: torch.tensor
    disentangling_information_x_prime: torch.tensor
    disentangling_information_y: torch.tensor
    disentangling_information_y_prime: torch.tensor


class ClassifierOutputs(NamedTuple):
    digit_bg_logits: torch.tensor
    digit_fg_logits: torch.tensor
    color_bg_logits: torch.tensor
    color_fg_logits: torch.tensor
