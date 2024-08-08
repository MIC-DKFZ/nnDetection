# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from batchgeneratorsv2.transforms.intensity.brightness import (
    MultiplicativeBrightnessTransform,
)
from batchgeneratorsv2.transforms.intensity.contrast import (
    BGContrast,
    ContrastTransform,
)
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import (
    SimulateLowResolutionTransform,
)
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import (
    Convert2DTo3DTransform,
    Convert3DTo2DTransform,
)
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform

from nndet.io.augmentation import AUGMENTATION_REGISTRY
from nndet.io.augmentation.base import ComposeBG2
from nndet.io.augmentation.pipeline.noaug import NoAugBG2


@AUGMENTATION_REGISTRY.register
class AugModularBG2(NoAugBG2):
    """
    Augmentations using BatchGeneratorsV2
    """

    def get_training_transforms(self):
        """
        - UtilTransforms
        - BGv2SpatialTransform
        - BGv2GaussianNoiseTransform
        - BGv2GaussianBlurTransform
        - BGv2MultiplicativeBrightnessTransform
        - BGv2ContrastTransform
        - [optional] BGv2SimulateLowResolutionTransform
        - [optional] BGv2GammaTransform (inverted)
        - [optional] BGv2GammaTransform
        - [optional] BGv2MirrorTransform
        - UtilTransforms
        """
        if self.use_box_io:
            raise NotImplementedError("Box Augs are not implemented for this augmentation pipeline")

        tr_transforms = []
        # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
        if self.params.get("dummy_2D", False):
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
        else:
            ignore_axes = None

        tr_transforms.append(
            SpatialTransform(
                patch_size=self._spatial_transform_patch_size,
                patch_center_dist_from_border=self.params["spatial"]["patch_center_dist_from_border"],
                random_crop=self.params["spatial"]["random_crop"],
                p_elastic_deform=self.params["spatial"]["p_elastic_deform"],
                elastic_deform_scale=self.params["spatial"]["elastic_deform_scale"],
                elastic_deform_magnitude=self.params["spatial"]["elastic_deform_magnitude"],
                p_synchronize_def_scale_across_axes=self.params["spatial"]["p_synchronize_def_scale_across_axes"],
                p_rotation=self.params["spatial"]["p_rotation"],
                rotation=self.params["spatial"]["rotation"],
                p_scaling=self.params["spatial"]["p_scaling"],
                scaling=self.params["spatial"]["scaling"],
                p_synchronize_scaling_across_axes=self.params["spatial"]["p_synchronize_scaling_across_axes"],
                bg_style_seg_sampling=self.params["spatial"]["bg_style_seg_sampling"],
                mode_seg=self.params["spatial"]["mode_seg"],
            )
        )

        if self.params.get("dummy_2D"):
            tr_transforms.append(Convert2DTo3DTransform())

        tr_transforms.append(
            RandomTransform(
                GaussianNoiseTransform(
                    noise_variance=self.params["gaussian_noise"]["noise_variance"],
                    p_per_channel=self.params["gaussian_noise"]["p_per_channel"],
                    synchronize_channels=self.params["gaussian_noise"]["synchronize_channels"],
                ),
                apply_probability=self.params["gaussian_noise"]["randomness"],
            )
        )

        tr_transforms.append(
            RandomTransform(
                GaussianBlurTransform(
                    blur_sigma=self.params["gaussian_blur"]["blur_sigma"],
                    synchronize_channels=self.params["gaussian_blur"]["synchronize_channels"],
                    synchronize_axes=self.params["gaussian_blur"]["synchronize_axes"],
                    p_per_channel=self.params["gaussian_blur"]["p_per_channel"],
                    benchmark=self.params["gaussian_blur"]["benchmark"],
                ),
                apply_probability=self.params["gaussian_blur"]["randomness"],
            )
        )

        tr_transforms.append(
            RandomTransform(
                MultiplicativeBrightnessTransform(
                    multiplier_range=BGContrast(self.params["brightness"]["multiplier_range"]),
                    synchronize_channels=self.params["brightness"]["synchronize_channels"],
                    p_per_channel=self.params["brightness"]["p_per_channel"],
                ),
                apply_probability=self.params["brightness"]["randomness"],
            )
        )

        tr_transforms.append(
            RandomTransform(
                ContrastTransform(
                    contrast_range=BGContrast(self.params["contrast"]["contrast_range"]),
                    preserve_range=self.params["contrast"]["preserve_range"],
                    synchronize_channels=self.params["contrast"]["synchronize_channels"],
                    p_per_channel=self.params["contrast"]["p_per_channel"],
                ),
                apply_probability=self.params["contrast"]["randomness"],
            )
        )

        if self.params.get("do_sim_low_res"):
            tr_transforms.append(
                RandomTransform(
                    SimulateLowResolutionTransform(
                        scale=self.params["sim_low_res"]["scale"],
                        synchronize_channels=self.params["sim_low_res"]["synchronize_channels"],
                        synchronize_axes=self.params["sim_low_res"]["synchronize_axes"],
                        ignore_axes=ignore_axes,
                        allowed_channels=self.params["sim_low_res"]["allowed_channels"],
                        p_per_channel=self.params["sim_low_res"]["p_per_channel"],
                    ),
                    apply_probability=self.params["sim_low_res"]["randomness"],
                )
            )

        if self.params.get("do_gamma_inverted"):
            tr_transforms.append(
                RandomTransform(
                    GammaTransform(
                        gamma=self.params["gamma_inverted"]["gamma"],
                        p_invert_image=self.params["gamma_inverted"]["p_invert_image"],
                        synchronize_channels=self.params["gamma_inverted"]["synchronize_channels"],
                        p_per_channel=self.params["gamma_inverted"]["p_per_channel"],
                        p_retain_stats=self.params["gamma_inverted"]["p_retain_stats"],
                    ),
                    apply_probability=self.params["gamma_inverted"]["randomness"],
                )
            )  # inverted gamma

        if self.params.get("do_gamma"):
            tr_transforms.append(
                RandomTransform(
                    GammaTransform(
                        gamma=self.params["gamma"]["gamma"],
                        p_invert_image=self.params["gamma"]["p_invert_image"],
                        synchronize_channels=self.params["gamma"]["synchronize_channels"],
                        p_per_channel=self.params["gamma"]["p_per_channel"],
                        p_retain_stats=self.params["gamma"]["p_retain_stats"],
                    ),
                    apply_probability=self.params["gamma"]["randomness"],
                )
            )

        if self.params.get("do_mirror"):
            tr_transforms.append(MirrorTransform(self.params["mirror"]["mirror_axes"]))

        if self.params.get("use_mask_for_norm"):
            use_mask_for_norm = self.params["use_mask_for_norm"]
            tr_transforms.append(
                MaskImageTransform(
                    apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                    channel_idx_in_seg=0,
                    set_outside_to=0,
                )
            )

        tr_transforms.append(RemoveLabelTansform(-1, 0))
        return ComposeBG2(tr_transforms)
