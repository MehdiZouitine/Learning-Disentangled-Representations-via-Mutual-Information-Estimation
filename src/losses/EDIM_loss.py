import torch.nn as nn
from src.losses.loss_functions import (
    DJSLoss,
    ClassifLoss,
    DiscriminatorLoss,
    GeneratorLoss,
)
from src.utils.custom_typing import (
    DiscriminatorOutputs,
    ClassifierOutputs,
    EDIMOutputs,
    GenLosses,
    DiscrLosses,
    ClassifLosses,
)


class EDIMLoss(nn.Module):
    def __init__(
        self,
        local_mutual_loss_coeff,
        global_mutual_loss_coeff,
        disentangling_loss_coeff,
    ):
        super().__init__()
        self.local_mutual_loss_coeff = local_mutual_loss_coeff
        self.global_mutual_loss_coeff = global_mutual_loss_coeff
        self.disentangling_loss_coeff = disentangling_loss_coeff

        self.djs_loss = DJSLoss()
        self.classif_loss = ClassifLoss()
        self.discriminator_loss = DiscriminatorLoss()
        self.generator_loss = GeneratorLoss()

    def compute_generator_loss(self, edim_outputs: EDIMOutputs):

        # Compute Global mutual loss
        global_mutual_loss_x = self.djs_loss(
            T=edim_outputs.global_mutual_M_R_x,
            T_prime=edim_outputs.global_mutual_M_R_x_prime,
        )
        global_mutual_loss_y = self.djs_loss(
            T=edim_outputs.global_mutual_M_R_y,
            T_prime=edim_outputs.global_mutual_M_R_y_prime,
        )
        global_mutual_loss = (
            global_mutual_loss_x + global_mutual_loss_y
        ) * self.global_mutual_loss_coeff

        # Compute Local mutual loss

        local_mutual_loss_x = self.djs_loss(
            T=edim_outputs.local_mutual_M_R_x,
            T_prime=edim_outputs.local_mutual_M_R_x_prime,
        )
        local_mutual_loss_y = self.djs_loss(
            T=edim_outputs.local_mutual_M_R_y,
            T_prime=edim_outputs.local_mutual_M_R_y_prime,
        )
        local_mutual_loss = (
            local_mutual_loss_x + local_mutual_loss_y
        ) * self.local_mutual_loss_coeff

        gan_loss_x_g = self.generator_loss(fake_logits=edim_outputs.fake_x)
        gan_loss_y_g = self.generator_loss(fake_logits=edim_outputs.fake_y)

        gan_loss_g = (gan_loss_x_g + gan_loss_y_g) * self.disentangling_loss_coeff

        # Get classification error

        # For each network, we assign a loss objective
        encoder_loss = global_mutual_loss + local_mutual_loss + gan_loss_g

        return GenLosses(
            encoder_loss=encoder_loss,
            local_mutual_loss=local_mutual_loss,
            global_mutual_loss=global_mutual_loss,
            gan_loss_g=gan_loss_g,
        )

    def compute_discriminator_loss(self, discr_outputs: DiscriminatorOutputs):
        gan_loss_x_d = self.discriminator_loss(
            real_logits=discr_outputs.disentangling_information_x_prime,
            fake_logits=discr_outputs.disentangling_information_x,
        )
        gan_loss_y_d = self.discriminator_loss(
            real_logits=discr_outputs.disentangling_information_y_prime,
            fake_logits=discr_outputs.disentangling_information_y,
        )

        gan_loss_d = (gan_loss_x_d + gan_loss_y_d) * self.disentangling_loss_coeff

        return DiscrLosses(gan_loss_d=gan_loss_d)

    def compute_classif_loss(
        self,
        classif_outputs: ClassifierOutputs,
        digit_labels,
        color_bg_labels,
        color_fg_labels,
    ):

        digit_bg_classif_loss, digit_bg_accuracy = self.classif_loss(
            y_pred=classif_outputs.digit_bg_logits,
            target=digit_labels,
        )
        digit_fg_classif_loss, digit_fg_accuracy = self.classif_loss(
            y_pred=classif_outputs.digit_fg_logits, target=digit_labels
        )
        color_bg_classif_loss, color_bg_accuracy = self.classif_loss(
            y_pred=classif_outputs.color_bg_logits,
            target=color_bg_labels,
        )
        color_fg_classif_loss, color_fg_accuracy = self.classif_loss(
            y_pred=classif_outputs.color_fg_logits, target=color_fg_labels
        )
        classif_loss = (
            digit_bg_classif_loss
            + digit_fg_classif_loss
            + color_bg_classif_loss
            + color_fg_classif_loss
        )

        return ClassifLosses(
            classif_loss=classif_loss,
            digit_bg_classif_loss=digit_bg_classif_loss,
            digit_fg_classif_loss=digit_fg_classif_loss,
            color_bg_classif_loss=color_bg_classif_loss,
            color_fg_classif_loss=color_fg_classif_loss,
            digit_bg_accuracy=digit_bg_accuracy,
            digit_fg_accuracy=digit_fg_accuracy,
            color_bg_accuracy=color_bg_accuracy,
            color_fg_accuracy=color_fg_accuracy,
        )
