from typing import NamedTuple
import torch


class GanLossOutput(NamedTuple):
    generator: torch.Tensor
    discriminator: torch.Tensor


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
    disentangling_information_x: torch.tensor
    disentangling_information_x_prime: torch.tensor
    disentangling_information_y: torch.tensor
    disentangling_information_y_prime: torch.tensor
    digit_bg_logits: torch.tensor
    digit_fg_logits: torch.tensor
    color_bg_logits: torch.tensor
    color_fg_logits: torch.tensor
    shared_x: torch.tensor
    shared_y: torch.tensor
