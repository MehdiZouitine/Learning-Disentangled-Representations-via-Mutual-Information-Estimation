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
