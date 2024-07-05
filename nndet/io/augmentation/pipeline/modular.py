# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from batchgenerators.transforms.channel_selection_transforms import (
    DataChannelSelectionTransform,
    SegChannelSelectionTransform,
)
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform,
    BrightnessTransform,
    ContrastAugmentationTransform,
    GammaTransform,
)
from batchgenerators.transforms.local_transforms import (
    BrightnessGradientAdditiveTransform,
    LocalGammaTransform,
)
from batchgenerators.transforms.noise_transforms import (
    BlankRectangleTransform,
    GaussianBlurTransform,
    GaussianNoiseTransform,
    MedianFilterTransform,
    SharpeningTransform,
)
from batchgenerators.transforms.resample_transforms import (
    SimulateLowResolutionTransform,
)
from batchgenerators.transforms.spatial_transforms import (
    MirrorTransform,
    Rot90Transform,
    SpatialTransform,
    TransposeAxesTransform,
)
from batchgenerators.transforms.utility_transforms import (
    NumpyToTensor,
    OneOfTransform,
    RemoveLabelTransform,
    RenameTransform,
)
from batchgeneratorsv2.helpers.scalar_type import RandomScalar as BGv2RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import (
    BasicTransform as BGv2BasicTransform,
)
from batchgeneratorsv2.transforms.intensity.brightness import (
    MultiplicativeBrightnessTransform as BGv2MultiplicativeBrightnessTransform,
)
from batchgeneratorsv2.transforms.intensity.contrast import BGContrast as BGv2BGContrast
from batchgeneratorsv2.transforms.intensity.contrast import (
    ContrastTransform as BGv2ContrastTransform,
)
from batchgeneratorsv2.transforms.intensity.gamma import (
    GammaTransform as BGv2GammaTransform,
)
from batchgeneratorsv2.transforms.intensity.gaussian_noise import (
    GaussianNoiseTransform as BGv2GaussianNoiseTransform,
)
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import (
    ApplyRandomBinaryOperatorTransform as BGv2ApplyRandomBinaryOperatorTransform,
)
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import (
    RemoveRandomConnectedComponentFromOneHotEncodingTransform as BGv2RemoveRandomConnectedComponentFromOneHotEncodingTransform,
)
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import (
    MoveSegAsOneHotToDataTransform as BGv2MoveSegAsOneHotToDataTransform,
)
from batchgeneratorsv2.transforms.noise.gaussian_blur import (
    GaussianBlurTransform as BGv2GaussianBlurTransform,
)
from batchgeneratorsv2.transforms.spatial.low_resolution import (
    SimulateLowResolutionTransform as BGv2SimulateLowResolutionTransform,
)
from batchgeneratorsv2.transforms.spatial.mirroring import (
    MirrorTransform as BGv2MirrorTransform,
)
from batchgeneratorsv2.transforms.spatial.spatial import (
    SpatialTransform as BGv2SpatialTransform,
)
from batchgeneratorsv2.transforms.utils.compose import (
    ComposeTransforms as BGv2ComposeTransforms,
)
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import (
    DownsampleSegForDSTransform as BGv2DownsampleSegForDSTransform,
)
from batchgeneratorsv2.transforms.utils.nnunet_masking import (
    MaskImageTransform as BGv2MaskImageTransform,
)
from batchgeneratorsv2.transforms.utils.pseudo2d import (
    Convert2DTo3DTransform as BGv2Convert2DTo3DTransform,
)
from batchgeneratorsv2.transforms.utils.pseudo2d import (
    Convert3DTo2DTransform as BGv2Convert3DTo2DTransform,
)
from batchgeneratorsv2.transforms.utils.random import (
    RandomTransform as BGv2RandomTransform,
)
from batchgeneratorsv2.transforms.utils.remove_label import (
    RemoveLabelTansform as BGv2RemoveLabelTansform,
)
from batchgeneratorsv2.transforms.utils.seg_to_regions import (
    ConvertSegmentationToRegionsTransform as BGv2ConvertSegmentationToRegionsTransform,
)

import nndet.io.transforms.detection as nndet_transforms
from nndet.io.augmentation import AUGMENTATION_REGISTRY
from nndet.io.augmentation.base import ComposePretty
from nndet.io.augmentation.nnunet import (
    Convert2DTo3DTransform,
    Convert3DTo2DTransform,
    MaskTransform,
)
from nndet.io.augmentation.pipeline.noaug import NoAug
from nndet.io.transforms.format import (
    Boxes2ObjectPointsTransform,
    ObjectPoints2BoxesTransform,
)

import torch
def call(self, **data_dict) -> dict:
    image = []
    segmentation = []
    for i in range(len(data_dict['data'])):
        data_dict['image'] = torch.tensor(data_dict['data'][i], dtype=torch.float32)
        data_dict['segmentation'] = torch.tensor(data_dict['seg'][i], dtype=torch.float32)
        params = self.get_parameters(**data_dict)
        data_dict = self.apply(data_dict, **params)
        image.append(data_dict['image'])
        segmentation.append(data_dict['segmentation'])
    data_dict['data'] = torch.stack(image)
    data_dict['seg'] = torch.stack(segmentation)
    return data_dict

BGv2BasicTransform.__call__ = call


@AUGMENTATION_REGISTRY.register
class AugModularBG2(NoAug):
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
        if self.params.get("selected_data_channels"):
            tr_transforms.append(DataChannelSelectionTransform(self.params.get("selected_data_channels")))
        if self.params.get("selected_seg_channels"):
            tr_transforms.append(SegChannelSelectionTransform(self.params.get("selected_seg_channels")))

        # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
        if self.params.get("dummy_2D", False):
            ignore_axes = (0,)
            tr_transforms.append(BGv2Convert3DTo2DTransform())
        else:
            ignore_axes = None

        tr_transforms.append(
            BGv2SpatialTransform(
                patch_size=self._spatial_transform_patch_size,
                patch_center_dist_from_border=self.params["spatial"].get("patch_center_dist_from_border"),
                random_crop=self.params["spatial"].get("random_crop"),
                p_elastic_deform=self.params["spatial"].get("p_elastic_deform"),
                elastic_deform_scale=self.params["spatial"].get("elastic_deform_scale"),
                elastic_deform_magnitude=self.params["spatial"].get("elastic_deform_magnitude"),
                p_synchronize_def_scale_across_axes=self.params["spatial"].get("p_synchronize_def_scale_across_axes"),
                p_rotation=self.params["spatial"].get("p_rotation"),
                rotation=np.array(self.params["spatial"].get("rotation")) * np.pi / 180,
                p_scaling=self.params["spatial"].get("p_scaling"),
                scaling=self.params["spatial"].get("scaling"),
                p_synchronize_scaling_across_axes=self.params["spatial"].get("p_synchronize_scaling_across_axes"),
                bg_style_seg_sampling=self.params["spatial"].get("bg_style_seg_sampling"),
                mode_seg=self.params["spatial"].get("mode_seg"),
            )
        )

        if self.params.get("dummy_2D"):
            tr_transforms.append(BGv2Convert2DTo3DTransform())

        tr_transforms.append(
            BGv2RandomTransform(
                BGv2GaussianNoiseTransform(
                    noise_variance=self.params["gaussian_noise"].get("noise_variance"),
                    p_per_channel=self.params["gaussian_noise"].get("p_per_channel"),
                    synchronize_channels=self.params["gaussian_noise"].get("synchronize_channels"),
                ),
                apply_probability=self.params["gaussian_noise"].get("randomness"),
            )
        )

        tr_transforms.append(
            BGv2RandomTransform(
                BGv2GaussianBlurTransform(
                    blur_sigma=self.params["gaussian_blur"].get("blur_sigma"),
                    synchronize_channels=self.params["gaussian_blur"].get("synchronize_channels"),
                    synchronize_axes=self.params["gaussian_blur"].get("synchronize_axes"),
                    p_per_channel=self.params["gaussian_blur"].get("p_per_channel"),
                    benchmark=self.params["gaussian_blur"].get("benchmark"),
                ),
                apply_probability=self.params["gaussian_blur"].get("randomness"),
            )
        )

        tr_transforms.append(
            BGv2RandomTransform(
                BGv2MultiplicativeBrightnessTransform(
                    multiplier_range=BGv2BGContrast(
                        self.params["brightness"].get("multiplier_range")
                    ),
                    synchronize_channels=self.params["brightness"].get("synchronize_channels"),
                    p_per_channel=self.params["brightness"].get("p_per_channel"),
                ),
                apply_probability=self.params["brightness"].get("randomness"),
            )
        )

        tr_transforms.append(
            BGv2RandomTransform(
                BGv2ContrastTransform(
                    contrast_range=BGv2BGContrast(self.params["contrast"].get("contrast_range")),
                    preserve_range=self.params["contrast"].get("preserve_range"),
                    synchronize_channels=self.params["contrast"].get("synchronize_channels"),
                    p_per_channel=self.params["contrast"].get("p_per_channel"),
                ),
                apply_probability=self.params["contrast"].get("randomness"),
            )
        )

        if self.params.get("do_sim_low_res"):
            tr_transforms.append(
                BGv2RandomTransform(
                    BGv2SimulateLowResolutionTransform(
                        scale=self.params["sim_low_res"].get("scale"),
                        synchronize_channels=self.params["sim_low_res"].get("synchronize_channels"),
                        synchronize_axes=self.params["sim_low_res"].get("synchronize_axes"),
                        ignore_axes=ignore_axes,
                        allowed_channels=self.params["sim_low_res"].get("allowed_channels"),
                        p_per_channel=self.params["sim_low_res"].get("p_per_channel"),
                    ),
                    apply_probability=self.params["sim_low_res"].get("randomness"),
                )
            )

        if self.params.get("do_gamma_inverted"):
            tr_transforms.append(
                BGv2RandomTransform(
                    BGv2GammaTransform(
                        gamma=self.params["gamma_inverted"].get("gamma"),
                        p_invert_image=self.params["gamma_inverted"].get("p_invert_image"),
                        synchronize_channels=self.params["gamma_inverted"].get("synchronize_channels"),
                        p_per_channel=self.params["gamma_inverted"].get("p_per_channel"),
                        p_retain_stats=self.params["gamma_inverted"].get("p_retain_stats"),
                    ),
                    apply_probability=self.params["gamma_inverted"].get("randomness"),
                )
            )  # inverted gamma

        if self.params.get("do_gamma"):
            tr_transforms.append(
                BGv2RandomTransform(
                    BGv2GammaTransform(
                        gamma=self.params["gamma"].get("gamma"),
                        p_invert_image=self.params["gamma"].get("p_invert_image"),
                        synchronize_channels=self.params["gamma"].get("synchronize_channels"),
                        p_per_channel=self.params["gamma"].get("p_per_channel"),
                        p_retain_stats=self.params["gamma"].get("p_retain_stats"),
                    ),
                    apply_probability=self.params["gamma_inverted"].get("randomness"),
                )
            )

        if self.params.get("do_mirror"):
            tr_transforms.append(BGv2MirrorTransform(self.params["mirror"].get("allowed_axes")))

        if self.params.get("use_mask_for_norm"):
            use_mask_for_norm = self.params.get("use_mask_for_norm")
            tr_transforms.append(
                BGv2MaskImageTransform(
                    apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                    channel_idx_in_seg=0,
                    set_outside_to=0,
                )
            )

        tr_transforms.append(RemoveLabelTransform(-1, 0))
        tr_transforms.append(RenameTransform("seg", "target", False))
        tr_transforms.append(NumpyToTensor(["data", "target"], "float"))
        # transforms = ComposePretty(tr_transforms)
        # logger.info(f"Training Transforms: \n{transforms}")
        
        return BGv2ComposeTransforms(tr_transforms)


@AUGMENTATION_REGISTRY.register
class AugModular(NoAug):
    """
    Started out as a direct copy of `BaseInsaneAug` but can be configured
    in various ways to increase / decrease the augmentation strength
    """

    def get_training_transforms(self):
        """
        - UtilTransforms
        - SpatialTransform
        - GaussianNoiseTransform
        - GaussianBlurTransform
        - BrightnessMultiplicativeTransform
        - [optional] BrightnessTransform
        - ContrastAugmentationTransform
        - [optional] SimulateLowResolutionTransform
        - [optional] GammaTransform (inverted)
        - [optional] GammaTransform
        - [optional] MirrorTransform
        - UtilTransforms
        """
        if self.use_box_io:
            raise NotImplementedError("Box Augs are not implemented for this augmentation pipeline")
        assert self.params.get("mirror") is None, "old version of params, use new keyword do_mirror"

        tr_transforms = []
        if self.params.get("selected_data_channels"):
            tr_transforms.append(DataChannelSelectionTransform(self.params.get("selected_data_channels")))
        if self.params.get("selected_seg_channels"):
            tr_transforms.append(SegChannelSelectionTransform(self.params.get("selected_seg_channels")))

        # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
        if self.params.get("dummy_2D", False):
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
        else:
            ignore_axes = None

        tr_transforms.append(
            SpatialTransform(
                self._spatial_transform_patch_size,
                patch_center_dist_from_border=None,
                do_elastic_deform=self.params.get("do_elastic"),
                alpha=self.params.get("elastic_deform_alpha"),
                sigma=self.params.get("elastic_deform_sigma"),
                do_rotation=self.params.get("do_rotation"),
                angle_x=self.params.get("rotation_x"),
                angle_y=self.params.get("rotation_y"),
                angle_z=self.params.get("rotation_z"),
                do_scale=self.params.get("do_scaling"),
                scale=self.params.get("scale_range"),
                order_data=self.params.get("order_data"),
                border_mode_data=self.params.get("border_mode_data"),
                border_cval_data=self.params.get("border_cval_data"),
                order_seg=self.params.get("order_seg"),
                border_mode_seg=self.params.get("border_mode_seg"),
                border_cval_seg=self.params.get("border_cval_seg"),
                random_crop=self.params.get("random_crop"),
                p_el_per_sample=self.params.get("p_eldef"),
                p_scale_per_sample=self.params.get("p_scale"),
                p_rot_per_sample=self.params.get("p_rot"),
                independent_scale_for_each_axis=self.params.get("independent_scale_factor_for_each_axis"),
            )
        )

        if self.params.get("dummy_2D"):
            tr_transforms.append(Convert2DTo3DTransform())

        # Additional spatial transformations
        if self.params.get("do_rot90", False):
            if self.any_matching_axes():
                tr_transforms.append(
                    Rot90Transform(
                        num_rot=(0, 1, 2, 3),
                        axes=self.same_axes(),
                        p_per_sample=self.params["rot90"]["p_per_sample"],
                    ),
                )

        if self.params.get("do_transpose_axes", False):
            if self.any_matching_axes():
                tr_transforms.append(
                    TransposeAxesTransform(
                        transpose_any_of_these=self.same_axes(),
                        p_per_sample=self.params["transpose_axes"]["p_per_sample"],
                    )
                )

        # we need to put the color augmentations after the dummy 2d part (if applicable). Otherwise the overloaded color
        # channel gets in the way

        tr_transforms.append(
            GaussianNoiseTransform(
                p_per_sample=self.params.get("p_per_sample_gaussian_noise"),
            ),
        )

        tr_transforms.append(
            GaussianBlurTransform(
                blur_sigma=self.params.get("gaussian_blur_sigma"),
                different_sigma_per_channel=self.params.get("gaussian_blur_sigma_per_channel"),
                p_per_sample=self.params.get("p_per_sample_gaussian_blur"),
                p_per_channel=self.params.get("p_per_channel_gaussian_blur"),
            ),
        )

        tr_transforms.append(
            BrightnessMultiplicativeTransform(
                p_per_sample=self.params.get("p_per_sample_brightness_mul"),
                multiplier_range=self.params.get("brightness_mul_multiplier_range"),
            ),
        )

        if self.params.get("do_additive_brightness"):
            tr_transforms.append(
                BrightnessTransform(
                    mu=self.params.get("additive_brightness_mu"),
                    sigma=self.params.get("additive_brightness_sigma"),
                    per_channel=self.params.get("additive_brightness_per_channel"),
                    p_per_sample=self.params.get("additive_brightness_p_per_sample"),
                    p_per_channel=self.params.get("additive_brightness_p_per_channel"),
                ),
            )

        tr_transforms.append(
            ContrastAugmentationTransform(
                contrast_range=self.params.get("contrast_range"),
                p_per_sample=self.params.get("p_per_sample_contrast"),
            ),
        )

        if self.params.get("do_sim_low_res"):
            tr_transforms.append(
                SimulateLowResolutionTransform(
                    p_per_sample=self.params.get("p_per_sample_sim_low_res"),
                    p_per_channel=self.params.get("p_per_channel_sim_low_res"),
                    zoom_range=self.params.get("sim_low_res_zoom_range"),
                    per_channel=self.params.get("sim_low_res_per_channel"),
                    order_downsample=self.params.get("sim_low_res_order_downsample"),
                    order_upsample=self.params.get("sim_low_res_order_upsample"),
                    ignore_axes=ignore_axes,
                ),
            )

        if self.params.get("do_gamma_inverted"):
            tr_transforms.append(
                GammaTransform(
                    gamma_range=self.params.get("gamma_range"),
                    invert_image=True,
                    per_channel=True,
                    retain_stats=self.params.get("gamma_retain_stats"),
                    p_per_sample=self.params["p_gamma_inverted"],
                ),
            )  # inverted gamma

        if self.params.get("do_gamma"):
            tr_transforms.append(
                GammaTransform(
                    gamma_range=self.params.get("gamma_range"),
                    invert_image=False,
                    per_channel=True,
                    retain_stats=self.params.get("gamma_retain_stats"),
                    p_per_sample=self.params["p_gamma"],
                ),
            )

        if self.params.get("do_mirror") or self.params.get("mirror"):
            tr_transforms.append(MirrorTransform(self.params.get("mirror_axes")))
        if self.params.get("use_mask_for_norm"):
            use_mask_for_norm = self.params.get("use_mask_for_norm")
            tr_transforms.append(MaskTransform(use_mask_for_norm, mask_idx_in_seg=0, set_outside_to=0))

        tr_transforms.append(RemoveLabelTransform(-1, 0))
        tr_transforms.append(RenameTransform("seg", "target", True))
        tr_transforms.append(NumpyToTensor(["data", "target"], "float"))
        transforms = ComposePretty(tr_transforms)
        # logger.info(f"Training Transforms: \n{transforms}")
        return transforms


@AUGMENTATION_REGISTRY.register
class AugModularPlus(NoAug):
    def get_training_transforms(self):
        """
        - UtilTransforms
        - SpatialTransform
        - [optional] Rot90
        - [optional] TransposeAxes
        - [optional] GaussianNoiseTransform
        - [optional] GaussianBlurTransform
        - [optional] BrightnessMultiplicativeTransform
        - [optional] BrightnessTransform
        - [optional] ContrastAugmentationTransform
        - [optional] SimulateLowResolutionTransform
        - [optional] OneOf
            - GammaTransform (inverted)
            - GammaTransform
        - [optional] MirrorTransform
        - UtilTransforms
        """
        if self.use_box_io:
            raise NotImplementedError("Box Augs are not implemented for this augmentation pipeline")
        assert self.params.get("mirror") is None, "old version of params, use new keyword do_mirror"

        tr_transforms = []
        if self.params.get("selected_data_channels"):
            tr_transforms.append(DataChannelSelectionTransform(self.params.get("selected_data_channels")))
        if self.params.get("selected_seg_channels"):
            tr_transforms.append(SegChannelSelectionTransform(self.params.get("selected_seg_channels")))

        # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
        if self.params.get("dummy_2D", False):
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
        else:
            ignore_axes = None

        tr_transforms.append(
            SpatialTransform(
                self._spatial_transform_patch_size,
                patch_center_dist_from_border=None,
                do_elastic_deform=self.params.get("do_elastic"),
                alpha=self.params.get("elastic_deform_alpha"),
                sigma=self.params.get("elastic_deform_sigma"),
                do_rotation=self.params.get("do_rotation"),
                angle_x=self.params.get("rotation_x"),
                angle_y=self.params.get("rotation_y"),
                angle_z=self.params.get("rotation_z"),
                do_scale=self.params.get("do_scaling"),
                scale=self.params.get("scale_range"),
                order_data=self.params.get("order_data"),
                border_mode_data=self.params.get("border_mode_data"),
                border_cval_data=self.params.get("border_cval_data"),
                order_seg=self.params.get("order_seg"),
                border_mode_seg=self.params.get("border_mode_seg"),
                border_cval_seg=self.params.get("border_cval_seg"),
                random_crop=self.params.get("random_crop"),
                p_el_per_sample=self.params.get("p_eldef"),
                p_scale_per_sample=self.params.get("p_scale"),
                p_rot_per_sample=self.params.get("p_rot"),
                independent_scale_for_each_axis=self.params.get("independent_scale_factor_for_each_axis"),
            )
        )

        if self.params.get("dummy_2D"):
            tr_transforms.append(Convert2DTo3DTransform())

        # we need to put the color augmentations after the dummy 2d part
        # (if applicable). Otherwise the overloaded color
        # channel gets in the way

        # Additional spatial transformations
        if self.params["do_rot90"]:
            if self.any_matching_axes():
                tr_transforms.append(
                    Rot90Transform(
                        num_rot=(0, 1, 2, 3),
                        axes=self.same_axes(),
                        p_per_sample=self.params["rot90"]["p_per_sample"],
                    ),
                )

        if self.params["do_transpose_axes"]:
            if self.any_matching_axes():
                tr_transforms.append(
                    TransposeAxesTransform(
                        transpose_any_of_these=self.same_axes(),
                        p_per_sample=self.params["transpose_axes"]["p_per_sample"],
                    )
                )

        if self.params.get("do_mirror") or self.params.get("mirror"):
            tr_transforms.append(MirrorTransform(self.params.get("mirror_axes")))

        if self.params["do_noise"]:
            tr_transforms.append(
                GaussianNoiseTransform(
                    p_per_sample=self.params.get("p_per_sample_gaussian_noise"),
                ),
            )

        if self.params["do_blur"]:
            tr_transforms.append(
                GaussianBlurTransform(
                    blur_sigma=self.params.get("gaussian_blur_sigma"),
                    different_sigma_per_channel=self.params.get("gaussian_blur_sigma_per_channel"),
                    p_per_sample=self.params.get("p_per_sample_gaussian_blur"),
                    p_per_channel=self.params.get("p_per_channel_gaussian_blur"),
                ),
            )

        if self.params["do_multiplicative_brightness"]:
            tr_transforms.append(
                BrightnessMultiplicativeTransform(
                    p_per_sample=self.params.get("p_per_sample_brightness_mul"),
                    multiplier_range=self.params.get("brightness_mul_multiplier_range"),
                ),
            )

        if self.params.get("do_additive_brightness"):
            tr_transforms.append(
                BrightnessTransform(
                    mu=self.params.get("additive_brightness_mu"),
                    sigma=self.params.get("additive_brightness_sigma"),
                    per_channel=self.params.get("additive_brightness_per_channel"),
                    p_per_sample=self.params.get("additive_brightness_p_per_sample"),
                    p_per_channel=self.params.get("additive_brightness_p_per_channel"),
                ),
            )

        if self.params["do_contrast"]:
            tr_transforms.append(
                ContrastAugmentationTransform(
                    contrast_range=self.params.get("contrast_range"),
                    p_per_sample=self.params.get("p_per_sample_contrast"),
                ),
            )

        if self.params.get("do_gamma"):
            one_of_gamma = []
            one_of_gamma.append(
                GammaTransform(
                    gamma_range=self.params.get("gamma_range"),
                    invert_image=True,
                    per_channel=True,
                    retain_stats=self.params.get("gamma_retain_stats"),
                    p_per_sample=self.params["p_gamma_inverted"],
                ),
            )  # inverted gamma
            one_of_gamma.append(
                GammaTransform(
                    gamma_range=self.params.get("gamma_range"),
                    invert_image=False,
                    per_channel=True,
                    retain_stats=self.params.get("gamma_retain_stats"),
                    p_per_sample=self.params["p_gamma"],
                ),
            )
            tr_transforms.append(OneOfTransform(one_of_gamma))

        if self.params.get("do_sim_low_res"):
            tr_transforms.append(
                SimulateLowResolutionTransform(
                    p_per_sample=self.params.get("p_per_sample_sim_low_res"),
                    p_per_channel=self.params.get("p_per_channel_sim_low_res"),
                    zoom_range=self.params.get("sim_low_res_zoom_range"),
                    per_channel=self.params.get("sim_low_res_per_channel"),
                    order_downsample=self.params.get("sim_low_res_order_downsample"),
                    order_upsample=self.params.get("sim_low_res_order_upsample"),
                    ignore_axes=ignore_axes,
                ),
            )

        if self.params.get("use_mask_for_norm"):
            use_mask_for_norm = self.params.get("use_mask_for_norm")
            tr_transforms.append(MaskTransform(use_mask_for_norm, mask_idx_in_seg=0, set_outside_to=0))

        tr_transforms.append(RemoveLabelTransform(-1, 0))
        tr_transforms.append(RenameTransform("seg", "target", True))
        tr_transforms.append(NumpyToTensor(["data", "target"], "float"))
        transforms = ComposePretty(tr_transforms)
        # logger.info(f"Training Transforms: \n{transforms}")
        return transforms


# Helpers AugV2


class HelperBrightnessScaleV2:
    def __init__(self, scale_param) -> None:
        self.scale_param = scale_param

    def __call__(self, x, y):
        return np.exp(
            np.random.uniform(
                np.log(x[y] // self.scale_param[0]),
                np.log(x[y] / self.scale_param[1]),
            )
        )


class HelperBrightnessStrengthV2:
    def __init__(self, strength_param) -> None:
        self.strength_param = strength_param

    def __call__(self, x, y):
        if np.random.uniform() < 0.5:
            return np.random.uniform(-self.strength_param[1], -self.strength_param[0])
        else:
            return np.random.uniform(self.strength_param[0], self.strength_param[1])


# def _brightness_strength(x, y):
#     if np.random.uniform() < 0.5:
#         return np.random.uniform(-_strength_param[1], -_strength_param[0])
#     else:
#         return np.random.uniform(_strength_param[0], _strength_param[1])


class HelperGammaScaleV2:
    def __init__(self, scale_param) -> None:
        self.scale_param = scale_param

    def __call__(self, x, y):
        return np.exp(
            np.random.uniform(
                np.log(x[y] // self.scale_param[0]),
                np.log(x[y] // self.scale_param[1]),
            )
        )


class HelperGammaStrengthV2:
    def __init__(self, strength_low, strength_high) -> None:
        self.strength_low = strength_low
        self.strength_high = strength_high

    def __call__(self):
        if np.random.uniform() < 0.5:
            return np.random.uniform(self.strength_low[0], self.strength_low[1])
        else:
            return np.random.uniform(self.strength_high[0], self.strength_high[1])


# def _gamma_scale(x, y):
#     return np.exp(
#         np.random.uniform(
#             np.log(x[y] // _scale_param[0]),
#             np.log(x[y] // _scale_param[1]),
#         )
#     )

# def _gamma_strength():
#     if np.random.uniform() < 0.5:
#         return np.random.uniform(_strength_low[0], _strength_low[1])
#     else:
#         return np.random.uniform(_strength_high[0], _strength_high[1])


@AUGMENTATION_REGISTRY.register
class AugModularV2(NoAug):
    """
    Adapted nnU-Net DA5 Augmentation Pipeline
    """

    def get_training_transforms(self):
        """
        - UtilTransforms
        - SpatialTransform
        - [Maybe] Rot90
        - [Maybe] TransposeAxes
        - Mirror
        - Noise
            - GaussianNoise
        - Blur (OneOf)
            - GaussianBlur
            - MedianFilter
        - Brightness (Additive)
        - Contrast (OneOf)
            - Constrast
            - Constrast
        - SimluateLowRes
        - Gamma (inverted)
        - Gamma
        - BlankRectangle
        - BrightnessGradient
        - LocalGamma
        - Sharpening
        - UtilTransforms
        """
        if self.use_box_io:
            raise NotImplementedError("Box Augs are not implemented for this augmentation pipeline")
        tr_transforms = []
        if self.params["selected_data_channels"]:
            tr_transforms.append(
                DataChannelSelectionTransform(
                    self.params["selected_data_channels"],
                )
            )
        if self.params["selected_seg_channels"]:
            tr_transforms.append(
                SegChannelSelectionTransform(
                    self.params["selected_seg_channels"],
                )
            )

        # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
        if self.params.get("dummy_2D", False):
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
        else:
            ignore_axes = None

        tr_transforms.append(
            SpatialTransform(
                self._spatial_transform_patch_size,
                patch_center_dist_from_border=None,
                do_elastic_deform=self.params["do_elastic"],
                alpha=self.params["elastic_deform_alpha"],
                sigma=self.params["elastic_deform_sigma"],
                do_rotation=self.params["do_rotation"],
                angle_x=self.params["rotation_x"],
                angle_y=self.params["rotation_y"],
                angle_z=self.params["rotation_z"],
                p_rot_per_axis=self.params["p_rot_per_axis"],
                do_scale=self.params["do_scaling"],
                scale=self.params["scale_range"],
                order_data=self.params["order_data"],
                border_mode_data=self.params["border_mode_data"],
                border_cval_data=self.params["border_cval_data"],
                order_seg=self.params["order_seg"],
                border_mode_seg=self.params["border_mode_seg"],
                border_cval_seg=self.params["border_cval_seg"],
                random_crop=self.params["random_crop"],
                p_el_per_sample=self.params["p_eldef"],
                p_scale_per_sample=self.params["p_scale"],
                p_rot_per_sample=self.params["p_rot"],
                independent_scale_for_each_axis=self.params["independent_scale_factor_for_each_axis"],
            )
        )

        if self.params.get("dummy_2D"):
            tr_transforms.append(Convert2DTo3DTransform())

        # we need to put the color augmentations after the dummy 2d part
        # (if applicable). Otherwise the overloaded color
        # channel gets in the way

        # Additional spatial transformations
        if self.params["do_rot90"]:
            if self.any_matching_axes():
                tr_transforms.append(
                    Rot90Transform(
                        num_rot=(0, 1, 2, 3),
                        axes=self.same_axes(),
                        p_per_sample=self.params["rot90"]["p_per_sample"],
                    ),
                )

        if self.params["do_transpose_axes"]:
            if self.any_matching_axes():
                tr_transforms.append(
                    TransposeAxesTransform(
                        transpose_any_of_these=self.same_axes(),
                        p_per_sample=self.params["transpose_axes"]["p_per_sample"],
                    )
                )

        if self.params["do_mirror"]:
            tr_transforms.append(MirrorTransform(axes=self.params["mirror_axes"]))

        if self.params["do_noise"]:
            tr_transforms.append(
                GaussianNoiseTransform(
                    noise_variance=tuple(self.params["gaussian_noise"]["noise_variance"]),
                    p_per_sample=self.params["gaussian_noise"]["p_per_sample"],
                ),
            )

        if self.params["do_blur"]:
            one_of_blur = []
            one_of_blur.append(
                GaussianBlurTransform(
                    blur_sigma=self.params["gaussian_blur"]["blur_sigma"],
                    different_sigma_per_channel=self.params["gaussian_blur"]["different_sigma_per_channel"],
                    p_per_sample=self.params["gaussian_blur"]["p_per_sample"],
                    p_per_channel=self.params["gaussian_blur"]["p_per_channel"],
                ),
            )
            one_of_blur.append(
                MedianFilterTransform(
                    filter_size=tuple(self.params["median_filter"]["filter_size"]),
                    same_for_each_channel=self.params["median_filter"]["same_for_each_channel"],
                    p_per_sample=self.params["median_filter"]["p_per_sample"],
                    p_per_channel=self.params["median_filter"]["p_per_channel"],
                ),
            )
            tr_transforms.append(OneOfTransform(one_of_blur))

        if self.params["do_brightness"]:
            tr_transforms.append(
                BrightnessTransform(
                    mu=self.params["brightness"]["mu"],
                    sigma=self.params["brightness"]["sigma"],
                    per_channel=self.params["brightness"]["per_channel"],
                    p_per_sample=self.params["brightness"]["p_per_sample"],
                    p_per_channel=self.params["brightness"]["p_per_channel"],
                ),
            )

        if self.params["do_contrast"]:
            one_of_contrast = []
            one_of_contrast.append(
                ContrastAugmentationTransform(
                    contrast_range=self.params["contrast"]["contrast_range"],
                    preserve_range=True,
                    per_channel=self.params["contrast"]["per_channel"],
                    p_per_sample=self.params["contrast"]["p_per_sample"],
                    p_per_channel=self.params["contrast"]["p_per_channel"],
                ),
            )
            one_of_contrast.append(
                ContrastAugmentationTransform(
                    contrast_range=self.params["contrast"]["contrast_range"],
                    preserve_range=False,
                    per_channel=self.params["contrast"]["per_channel"],
                    p_per_sample=self.params["contrast"]["p_per_sample"],
                    p_per_channel=self.params["contrast"]["p_per_channel"],
                ),
            )
            tr_transforms.append(OneOfTransform(one_of_contrast))

        if self.params["do_sim_low_res"]:
            tr_transforms.append(
                SimulateLowResolutionTransform(
                    p_per_sample=self.params["sim_low_res"]["p_per_sample"],
                    p_per_channel=self.params["sim_low_res"]["p_per_channel"],
                    zoom_range=self.params["sim_low_res"]["zoom_range"],
                    per_channel=self.params["sim_low_res"]["per_channel"],
                    order_downsample=self.params["sim_low_res"]["order_downsample"],
                    order_upsample=self.params["sim_low_res"]["order_upsample"],
                    ignore_axes=ignore_axes,
                ),
            )

        if self.params["do_gamma_inverted"]:
            tr_transforms.append(
                GammaTransform(
                    gamma_range=self.params["gamma"]["gamma_range"],
                    invert_image=True,
                    per_channel=self.params["gamma"]["per_channel"],
                    retain_stats=self.params["gamma"]["retain_stats"],
                    p_per_sample=self.params["p_gamma_inverted"],
                ),
            )  # inverted gamma

        if self.params["do_gamma"]:
            tr_transforms.append(
                GammaTransform(
                    gamma_range=self.params["gamma"]["gamma_range"],
                    invert_image=False,
                    per_channel=self.params["gamma"]["per_channel"],
                    retain_stats=self.params["gamma"]["retain_stats"],
                    p_per_sample=self.params["p_gamma"],
                ),
            )

        if self.params["do_blanks"]:
            rectangle_size = [
                [
                    max(1, p // self.params["blank_rectangle"]["scale"][0]),
                    max(1, p // self.params["blank_rectangle"]["scale"][1]),
                ]
                for p in self.patch_size
            ]
            tr_transforms.append(
                BlankRectangleTransform(
                    rectangle_size=rectangle_size,
                    rectangle_value=np.mean,
                    num_rectangles=self.params["blank_rectangle"]["num_rectangles"],
                    force_square=self.params["blank_rectangle"]["force_square"],
                    p_per_sample=self.params["blank_rectangle"]["p_per_sample"],
                    p_per_channel=self.params["blank_rectangle"]["p_per_channel"],
                )
            )

        if self.params["do_brightness_gradient"]:
            _scale_param = self.params["brightness_gradient"]["scale"]
            _strength_param = self.params["brightness_gradient"]["strength"]

            tr_transforms.append(
                BrightnessGradientAdditiveTransform(
                    scale=HelperBrightnessScaleV2(_scale_param),
                    max_strength=HelperBrightnessStrengthV2(_strength_param),
                    loc=self.params["brightness_gradient"]["loc"],
                    mean_centered=self.params["brightness_gradient"]["mean_centered"],
                    same_for_all_channels=self.params["brightness_gradient"]["same_for_all_channels"],
                    p_per_sample=self.params["brightness_gradient"]["p_per_sample"],
                    p_per_channel=self.params["brightness_gradient"]["p_per_channel"],
                )
            )

        if self.params["do_local_gamma"]:
            _scale_param = self.params["local_gamma"]["scale"]
            _strength_low = self.params["local_gamma"]["strength_low"]
            _strength_high = self.params["local_gamma"]["strength_high"]

            tr_transforms.append(
                LocalGammaTransform(
                    scale=HelperGammaScaleV2(_scale_param),
                    gamma=HelperGammaStrengthV2(_strength_low, _strength_high),
                    loc=self.params["local_gamma"]["loc"],
                    same_for_all_channels=self.params["local_gamma"]["same_for_all_channels"],
                    p_per_sample=self.params["local_gamma"]["p_per_sample"],
                    p_per_channel=self.params["local_gamma"]["p_per_channel"],
                )
            )

        if self.params["do_sharpening"]:
            tr_transforms.append(
                SharpeningTransform(
                    strength=self.params["sharpening"]["strength"],
                    same_for_each_channel=self.params["sharpening"]["same_for_each_channel"],
                    p_per_sample=self.params["sharpening"]["p_per_sample"],
                    p_per_channel=self.params["sharpening"]["p_per_channel"],
                )
            )

        if any(list(self.params["use_mask_for_norm"].values())):
            tr_transforms.append(
                MaskTransform(
                    self.params["use_mask_for_norm"],
                    mask_idx_in_seg=0,
                    set_outside_to=0,
                )
            )

        tr_transforms.append(RemoveLabelTransform(-1, 0))
        tr_transforms.append(RenameTransform("seg", "target", True))
        tr_transforms.append(NumpyToTensor(["data", "target"], "float"))
        transforms = ComposePretty(tr_transforms)
        # logger.info(f"Training Transforms: \n{transforms}")
        return transforms


@AUGMENTATION_REGISTRY.register
class AugModularWBoxes(NoAug):
    """
    Aug Modular with Spatial Aug Transforms from nnDetection
    """

    def get_keys(self):
        if self.use_box_io:
            label_key = None
            point_key = "target_points"
        else:
            label_key = "seg"
            point_key = None
        return "data", label_key, point_key

    def get_training_transforms(self):
        """
        - UtilTransforms
        - SpatialTransform
        - GaussianNoiseTransform
        - GaussianBlurTransform
        - BrightnessMultiplicativeTransform
        - [optional] BrightnessTransform
        - ContrastAugmentationTransform
        - [optional] SimulateLowResolutionTransform
        - [optional] GammaTransform (inverted)
        - [optional] GammaTransform
        - [optional] MirrorTransform
        - UtilTransforms
        """
        assert self.params.get("mirror") is None, "old version of params, use new keyword do_mirror"
        data_key, label_key, point_key = self.get_keys()

        tr_transforms = []
        if self.params.get("selected_data_channels"):
            tr_transforms.append(DataChannelSelectionTransform(self.params.get("selected_data_channels")))
        if self.params.get("selected_seg_channels") and label_key is not None:
            tr_transforms.append(SegChannelSelectionTransform(self.params.get("selected_seg_channels")))

        tr_transforms.append(
            Boxes2ObjectPointsTransform(
                data_key=data_key,
                box_coord_key="target_boxes",
                point_key=point_key,
            )
        )

        # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
        dummy_2D_keys = ["data"] if label_key is None else ["data", "seg"]
        if self.params.get("dummy_2D", False):
            ignore_axes = (0,)
            tr_transforms.append(nndet_transforms.Convert3DTo2DTransform(array_keys=dummy_2D_keys, point_key=point_key))
        else:
            ignore_axes = None

        tr_transforms.append(
            nndet_transforms.SpatialTransform(
                data_key=data_key,
                label_key=label_key,
                point_key=point_key,
                patch_size=self._spatial_transform_patch_size,
                patch_center_dist_from_border=None,
                do_elastic_deform=self.params.get("do_elastic"),
                alpha=self.params.get("elastic_deform_alpha"),
                sigma=self.params.get("elastic_deform_sigma"),
                do_rotation=self.params.get("do_rotation"),
                angle_x=self.params.get("rotation_x"),
                angle_y=self.params.get("rotation_y"),
                angle_z=self.params.get("rotation_z"),
                do_scale=self.params.get("do_scaling"),
                scale=self.params.get("scale_range"),
                order_data=self.params.get("order_data"),
                border_mode_data=self.params.get("border_mode_data"),
                border_cval_data=self.params.get("border_cval_data"),
                order_seg=self.params.get("order_seg"),
                border_mode_seg=self.params.get("border_mode_seg"),
                border_cval_seg=self.params.get("border_cval_seg"),
                random_crop=self.params.get("random_crop"),
                p_el_per_sample=self.params.get("p_eldef"),
                p_scale_per_sample=self.params.get("p_scale"),
                p_rot_per_sample=self.params.get("p_rot"),
                independent_scale_for_each_axis=self.params.get("independent_scale_factor_for_each_axis"),
            )
        )

        if self.params.get("dummy_2D"):
            tr_transforms.append(nndet_transforms.Convert2DTo3DTransform(array_keys=dummy_2D_keys, point_key=point_key))

        # we need to put the color augmentations after the dummy 2d part (if applicable). Otherwise the overloaded color
        # channel gets in the way

        tr_transforms.append(
            GaussianNoiseTransform(
                p_per_sample=self.params.get("p_per_sample_gaussian_noise"),
            ),
        )

        tr_transforms.append(
            GaussianBlurTransform(
                blur_sigma=self.params.get("gaussian_blur_sigma"),
                different_sigma_per_channel=self.params.get("gaussian_blur_sigma_per_channel"),
                p_per_sample=self.params.get("p_per_sample_gaussian_blur"),
                p_per_channel=self.params.get("p_per_channel_gaussian_blur"),
            ),
        )

        tr_transforms.append(
            BrightnessMultiplicativeTransform(
                p_per_sample=self.params.get("p_per_sample_brightness_mul"),
                multiplier_range=self.params.get("brightness_mul_multiplier_range"),
            ),
        )

        if self.params.get("do_additive_brightness"):
            tr_transforms.append(
                BrightnessTransform(
                    mu=self.params.get("additive_brightness_mu"),
                    sigma=self.params.get("additive_brightness_sigma"),
                    per_channel=self.params.get("additive_brightness_per_channel"),
                    p_per_sample=self.params.get("additive_brightness_p_per_sample"),
                    p_per_channel=self.params.get("additive_brightness_p_per_channel"),
                ),
            )

        tr_transforms.append(
            ContrastAugmentationTransform(
                contrast_range=self.params.get("contrast_range"),
                p_per_sample=self.params.get("p_per_sample_contrast"),
            ),
        )

        if self.params.get("do_sim_low_res"):
            tr_transforms.append(
                SimulateLowResolutionTransform(
                    p_per_sample=self.params.get("p_per_sample_sim_low_res"),
                    p_per_channel=self.params.get("p_per_channel_sim_low_res"),
                    zoom_range=self.params.get("sim_low_res_zoom_range"),
                    per_channel=self.params.get("sim_low_res_per_channel"),
                    order_downsample=self.params.get("sim_low_res_order_downsample"),
                    order_upsample=self.params.get("sim_low_res_order_upsample"),
                    ignore_axes=ignore_axes,
                ),
            )

        if self.params.get("do_gamma_inverted"):
            tr_transforms.append(
                GammaTransform(
                    gamma_range=self.params.get("gamma_range"),
                    invert_image=True,
                    per_channel=True,
                    retain_stats=self.params.get("gamma_retain_stats"),
                    p_per_sample=self.params["p_gamma_inverted"],
                ),
            )  # inverted gamma

        if self.params.get("do_gamma"):
            tr_transforms.append(
                GammaTransform(
                    gamma_range=self.params.get("gamma_range"),
                    invert_image=False,
                    per_channel=True,
                    retain_stats=self.params.get("gamma_retain_stats"),
                    p_per_sample=self.params["p_gamma"],
                ),
            )

        if self.params.get("do_mirror") or self.params.get("mirror"):
            tr_transforms.append(
                nndet_transforms.MirrorTransform(
                    data_key=data_key,
                    label_key=label_key,
                    point_key=point_key,
                    axes=self.params.get("mirror_axes"),
                )
            )
        if self.params.get("use_mask_for_norm") and label_key is not None:
            use_mask_for_norm = self.params.get("use_mask_for_norm")
            tr_transforms.append(MaskTransform(use_mask_for_norm, mask_idx_in_seg=0, set_outside_to=0))

        tr_transforms.append(
            ObjectPoints2BoxesTransform(
                data_key="data",
                box_coord_key="target_boxes",
                box_label_key="target_classes",
                point_key="target_points",
            )
        )

        _keys = ["data"]
        if label_key is not None:
            tr_transforms.append(RemoveLabelTransform(-1, 0))
            tr_transforms.append(RenameTransform("seg", "target", True))
            _keys.append("target")
        if point_key is not None:
            _keys.extend(["target_boxes", "target_classes"])
        tr_transforms.append(NumpyToTensor(_keys, "float"))
        transforms = ComposePretty(tr_transforms)
        # logger.info(f"Training Transforms: \n{transforms}")
        return transforms

    def get_validation_transforms(self):
        data_key, label_key, point_key = self.get_keys()
        val_transforms = []

        val_transforms.append(
            Boxes2ObjectPointsTransform(
                data_key=data_key,
                box_coord_key="target_boxes",
                point_key=point_key,
            )
        )

        if self.params.get("selected_data_channels"):
            val_transforms.append(DataChannelSelectionTransform(self.params.get("selected_data_channels")))
        if self.params.get("selected_seg_channels") and label_key is not None:
            val_transforms.append(SegChannelSelectionTransform(self.params.get("selected_seg_channels")))

        val_transforms.append(
            nndet_transforms.CenterCropTransform(
                crop_size=self.patch_size,
                data_key=data_key,
                label_key=label_key,
                point_key=point_key,
            )
        )

        val_transforms.append(
            ObjectPoints2BoxesTransform(
                data_key=data_key,
                box_coord_key="target_boxes",
                box_label_key="target_classes",
                point_key=point_key,
            )
        )

        _keys = ["data"]
        if label_key is not None:
            val_transforms.append(RemoveLabelTransform(-1, 0))
            val_transforms.append(RenameTransform("seg", "target", True))
            _keys.append("target")
        if point_key is not None:
            _keys.extend(["target_boxes", "target_classes"])
        val_transforms.append(NumpyToTensor(_keys, "float"))

        return ComposePretty(val_transforms)
