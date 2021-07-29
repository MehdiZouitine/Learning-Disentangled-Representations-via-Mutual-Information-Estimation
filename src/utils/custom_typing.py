from typing import NamedTuple, Tuple
import torch


class GanLossOutput(NamedTuple):
    discriminator: torch.Tensor
    generator: torch.Tensor


class EncoderOutput(NamedTuple):
    representation: torch.Tensor
    feature: torch.Tensor


class ColoredMNISTData(NamedTuple):
    fg: torch.Tensor
    bg: torch.Tensor
    fg_label: torch.Tensor
    bg_label: torch.Tensor
    digit_label: torch.Tensor


class SDIMOutputs(NamedTuple):
    global_mutual_M_R_x: torch.Tensor
    global_mutual_M_R_x_prime: torch.Tensor
    global_mutual_M_R_y: torch.Tensor
    global_mutual_M_R_y_prime: torch.Tensor
    local_mutual_M_R_x: torch.Tensor
    local_mutual_M_R_x_prime: torch.Tensor
    local_mutual_M_R_y: torch.Tensor
    local_mutual_M_R_y_prime: torch.Tensor
    digit_logits: torch.Tensor
    color_bg_logits: torch.Tensor
    color_fg_logits: torch.Tensor
    shared_x: torch.Tensor
    shared_y: torch.Tensor


class EDIMOutputs(NamedTuple):
    global_mutual_M_R_x: torch.Tensor
    global_mutual_M_R_x_prime: torch.Tensor
    global_mutual_M_R_y: torch.Tensor
    global_mutual_M_R_y_prime: torch.Tensor
    local_mutual_M_R_x: torch.Tensor
    local_mutual_M_R_x_prime: torch.Tensor
    local_mutual_M_R_y: torch.Tensor
    local_mutual_M_R_y_prime: torch.Tensor
    shared_x: torch.Tensor
    shared_y: torch.Tensor
    fake_x: torch.Tensor
    fake_y: torch.Tensor
    R_y_x: torch.Tensor
    R_x_y: torch.Tensor
    shuffle_x: torch.Tensor
    shuffle_y: torch.Tensor
    exclusive_x: torch.Tensor
    exclusive_y: torch.Tensor


class SDIMLosses(NamedTuple):
    total_loss: torch.Tensor
    encoder_loss: torch.Tensor
    local_mutual_loss: torch.Tensor
    global_mutual_loss: torch.Tensor
    shared_loss: torch.Tensor
    digit_classif_loss: torch.Tensor
    color_bg_classif_loss: torch.Tensor
    color_fg_classif_loss: torch.Tensor
    digit_accuracy: torch.Tensor
    color_bg_accuracy: torch.Tensor
    color_fg_accuracy: torch.Tensor


class GenLosses(NamedTuple):
    encoder_loss: torch.Tensor
    local_mutual_loss: torch.Tensor
    global_mutual_loss: torch.Tensor
    gan_loss_g: torch.Tensor


class ClassifLosses(NamedTuple):
    classif_loss: torch.Tensor
    digit_bg_classif_loss: torch.Tensor
    digit_fg_classif_loss: torch.Tensor
    color_bg_classif_loss: torch.Tensor
    color_fg_classif_loss: torch.Tensor
    digit_bg_accuracy: torch.Tensor
    digit_fg_accuracy: torch.Tensor
    color_bg_accuracy: torch.Tensor
    color_fg_accuracy: torch.Tensor


class DiscrLosses(NamedTuple):
    gan_loss_d: torch.Tensor


class GeneratorOutputs(NamedTuple):
    real_x: torch.Tensor
    fake_x: torch.Tensor
    real_y: torch.Tensor
    fake_y: torch.Tensor
    exclusive_x: torch.Tensor
    exclusive_y: torch.Tensor


class DiscriminatorOutputs(NamedTuple):
    disentangling_information_x: torch.Tensor
    disentangling_information_x_prime: torch.Tensor
    disentangling_information_y: torch.Tensor
    disentangling_information_y_prime: torch.Tensor


class ClassifierOutputs(NamedTuple):
    digit_bg_logits: torch.Tensor
    digit_fg_logits: torch.Tensor
    color_bg_logits: torch.Tensor
    color_fg_logits: torch.Tensor
