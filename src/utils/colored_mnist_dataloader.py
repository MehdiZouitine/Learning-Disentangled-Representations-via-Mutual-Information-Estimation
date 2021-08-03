import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import Dataset
import random
import numpy as np
import torch.nn.functional as F
from src.utils.custom_typing import ColoredMNISTData
from torch.utils.data import DataLoader
import os


class ColoredMNISTDataset(Dataset):
    @staticmethod
    def get_random_colors():
        rgb_code_list = [
            (255.0, 0.0, 0.0),
            (255.0, 128.0, 0.0),
            (255.0, 255.0, 0.0),
            (128.0, 255.0, 0.0),
            (0.0, 255.0, 0.0),
            (0.0, 255.0, 128.0),
            (0.0, 255.0, 255.0),
            (0.0, 128.0, 255.0),
            (0.0, 0.0, 255.0),
            (128.0, 0.0, 255.0),
            (255.0, 0.0, 255.0),
            (255.0, 0.0, 128.0),
        ]

        lenght = len(rgb_code_list)
        bg_index = random.randint(0, lenght - 1)
        fg_index = random.randint(0, lenght - 1)
        color_bg = rgb_code_list[bg_index]
        color_fg = rgb_code_list[fg_index]

        return color_bg, color_fg, bg_index, fg_index

    @staticmethod
    def create_colored_pairs(image, rgb_color_bg, rgb_color_fg):
        """
        Get an MNIST image an generate two nex images by changing the background and foreground of the image

        :param image: Array whose values are in the range of [0.0, 1.0]
        """
        index_background = (image < 0.5).long()
        index_foreground = (image >= 0.5).long()

        keep_background = index_background * image
        keep_foreground = index_foreground * image

        index_background = index_background - keep_background
        index_foreground = keep_foreground

        colored_background = torch.stack(
            [
                rgb_color_bg[0] * index_background + keep_foreground * 255.0,
                rgb_color_bg[1] * index_background + keep_foreground * 255.0,
                rgb_color_bg[2] * index_background + keep_foreground * 255.0,
            ],
            axis=2,
        )

        colored_foreground = torch.stack(
            [
                rgb_color_fg[0] * index_foreground + keep_background * 255.0,
                rgb_color_fg[1] * index_foreground + keep_background * 255.0,
                rgb_color_fg[2] * index_foreground + keep_background * 255.0,
            ],
            axis=2,
        )

        return colored_background.permute(2, 0, 1), colored_foreground.permute(2, 0, 1)

    def __init__(self, train=True, data_folder="data") -> None:
        super().__init__()
        if not os.path.isdir(data_folder):
            os.makedirs(data_folder)
        self.data = datasets.MNIST(
            root=data_folder, train=train, download=True, transform=ToTensor()
        )

    def __getitem__(self, index):
        image, digit_label = self.data[index]
        # image /= 255
        rgb_color_bg, rgb_color_fg, bg_label, fg_label = self.get_random_colors()
        bg_digit, fg_digit = self.create_colored_pairs(
            image=image.squeeze(0), rgb_color_bg=rgb_color_bg, rgb_color_fg=rgb_color_fg
        )
        fg_digit /= 255
        bg_digit /= 255
        fg_label = torch.tensor(fg_label, dtype=torch.float32)
        bg_label = torch.tensor(bg_label, dtype=torch.float32)
        digit_label = torch.tensor(digit_label, dtype=torch.float32)
        return ColoredMNISTData(
            bg=bg_digit,
            fg=fg_digit,
            fg_label=fg_label,
            bg_label=bg_label,
            digit_label=digit_label,
        )

    def __len__(self):
        return len(self.data)
