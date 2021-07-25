import torch.nn as nn
from src.losses.loss_functions import DJSLoss, ClassifLoss
from src.utils.custom_typing import SDIMLosses


class SDIMLoss(nn.Module):
    def __init__(
        self, local_mutual_loss_coeff, global_mutual_loss_coeff, shared_loss_coeff
    ):
        super().__init__()
        self.local_mutual_loss_coeff = local_mutual_loss_coeff
        self.global_mutual_loss_coeff = global_mutual_loss_coeff
        self.shared_loss_coeff = shared_loss_coeff

        self.djs_loss = DJSLoss()
        self.classif_loss = ClassifLoss()
        self.l1_loss = nn.L1Loss()

    def __call__(
        self,
        sdim_outputs,
        digit_labels,
        color_bg_labels,
        color_fg_labels,
    ):

        # Compute Global mutual loss
        global_mutual_loss_x = self.djs_loss(
            T=sdim_outputs.global_mutual_M_R_x,
            T_prime=sdim_outputs.global_mutual_M_R_x_prime,
        )
        global_mutual_loss_y = self.djs_loss(
            T=sdim_outputs.global_mutual_M_R_y,
            T_prime=sdim_outputs.global_mutual_M_R_y_prime,
        )
        global_mutual_loss = (
            global_mutual_loss_x + global_mutual_loss_y
        ) * self.global_mutual_loss_coeff

        # Compute Local mutual loss

        local_mutual_loss_x = self.djs_loss(
            T=sdim_outputs.local_mutual_M_R_x,
            T_prime=sdim_outputs.local_mutual_M_R_x_prime,
        )
        local_mutual_loss_y = self.djs_loss(
            T=sdim_outputs.local_mutual_M_R_y,
            T_prime=sdim_outputs.local_mutual_M_R_y_prime,
        )
        local_mutual_loss = (
            local_mutual_loss_x + local_mutual_loss_y
        ) * self.local_mutual_loss_coeff

        # Compute L1 on shared features
        shared_loss = self.l1_loss(sdim_outputs.shared_x, sdim_outputs.shared_y)
        shared_loss = shared_loss * self.shared_loss_coeff

        # Get classification error
        digit_classif_loss, digit_accuracy = self.classif_loss(
            y_pred=sdim_outputs.digit_logits, target=digit_labels
        )
        color_bg_classif_loss, color_bg_accuracy = self.classif_loss(
            y_pred=sdim_outputs.color_bg_logits, target=color_bg_labels
        )
        color_fg_classif_loss, color_fg_accuracy = self.classif_loss(
            y_pred=sdim_outputs.color_fg_logits, target=color_fg_labels
        )

        encoder_loss = global_mutual_loss + local_mutual_loss + shared_loss

        total_loss = (
            global_mutual_loss
            + local_mutual_loss
            + shared_loss
            + digit_classif_loss
            + color_bg_classif_loss
            + color_fg_classif_loss
        )

        return SDIMLosses(
            total_loss=total_loss,
            encoder_loss=encoder_loss,
            local_mutual_loss=local_mutual_loss,
            global_mutual_loss=global_mutual_loss,
            shared_loss=shared_loss,
            digit_classif_loss=digit_classif_loss,
            color_bg_classif_loss=color_bg_classif_loss,
            color_fg_classif_loss=color_fg_classif_loss,
            digit_accuracy=digit_accuracy,
            color_bg_accuracy=color_bg_accuracy,
            color_fg_accuracy=color_fg_accuracy,
        )
