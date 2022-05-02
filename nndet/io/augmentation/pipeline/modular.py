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
from loguru import logger

from nndet.io.augmentation.base import ComposePretty
from nndet.io.augmentation.pipeline.noaug import NoAug
from nndet.utils.info import SuppressPrint

with SuppressPrint():
    from nnunet.training.data_augmentation.custom_transforms import (
        Convert3DTo2DTransform,
        Convert2DTo3DTransform,
        MaskTransform,
    )

from nndet.io.augmentation import AUGMENTATION_REGISTRY


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
        assert (
            self.params.get("mirror") is None
        ), "old version of params, use new keyword do_mirror"

        tr_transforms = []
        if self.params.get("selected_data_channels"):
            tr_transforms.append(
                DataChannelSelectionTransform(self.params.get("selected_data_channels"))
            )
        if self.params.get("selected_seg_channels"):
            tr_transforms.append(
                SegChannelSelectionTransform(self.params.get("selected_seg_channels"))
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
                independent_scale_for_each_axis=self.params.get(
                    "independent_scale_factor_for_each_axis"
                ),
            )
        )

        if self.params.get("dummy_2D"):
            tr_transforms.append(Convert2DTo3DTransform())

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
                different_sigma_per_channel=self.params.get(
                    "gaussian_blur_sigma_per_channel"
                ),
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
            tr_transforms.append(
                MaskTransform(use_mask_for_norm, mask_idx_in_seg=0, set_outside_to=0)
            )

        tr_transforms.append(RemoveLabelTransform(-1, 0))
        tr_transforms.append(RenameTransform("seg", "target", True))
        tr_transforms.append(NumpyToTensor(["data", "target"], "float"))
        transforms = ComposePretty(tr_transforms)
        logger.info(f"Training Transforms: \n{transforms}")
        return transforms


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
        - Contast (OneOf)
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
                independent_scale_for_each_axis=self.params[
                    "independent_scale_factor_for_each_axis"
                ],
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
                    noise_variance=tuple(
                        self.params["gaussian_noise"]["noise_variance"]
                    ),
                    p_per_sample=self.params["gaussian_noise"]["p_per_sample"],
                ),
            )

        if self.params["do_blur"]:
            one_of_blur = []
            one_of_blur.append(
                GaussianBlurTransform(
                    blur_sigma=self.params["gaussian_blur"]["blur_sigma"],
                    different_sigma_per_channel=self.params["gaussian_blur"][
                        "different_sigma_per_channel"
                    ],
                    p_per_sample=self.params["gaussian_blur"]["p_per_sample"],
                    p_per_channel=self.params["gaussian_blur"]["p_per_channel"],
                ),
            )
            one_of_blur.append(
                MedianFilterTransform(
                    filter_size=tuple(self.params["median_filter"]["filter_size"]),
                    same_for_each_channel=self.params["median_filter"][
                        "same_for_each_channel"
                    ],
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

            def _brightness_scale(x, y):
                return np.exp(
                    np.random.uniform(
                        np.log(x[y] // _scale_param[0]),
                        np.log(x[y] / _scale_param[1]),
                    )
                )

            _strength_param = self.params["brightness_gradient"]["strength"]

            def _brightness_strength(x, y):
                if np.random.uniform() < 0.5:
                    return np.random.uniform(-_strength_param[1], -_strength_param[0])
                else:
                    return np.random.uniform(_strength_param[0], _strength_param[1])

            tr_transforms.append(
                BrightnessGradientAdditiveTransform(
                    scale=_brightness_scale,
                    max_strength=_brightness_strength,
                    loc=self.params["brightness_gradient"]["loc"],
                    mean_centered=self.params["brightness_gradient"]["mean_centered"],
                    same_for_all_channels=self.params["brightness_gradient"][
                        "same_for_all_channels"
                    ],
                    p_per_sample=self.params["brightness_gradient"]["p_per_sample"],
                    p_per_channel=self.params["brightness_gradient"]["p_per_channel"],
                )
            )

        if self.params["do_local_gamma"]:
            _scale_param = self.params["local_gamma"]["scale"]
            _strength_low = self.params["local_gamma"]["strength_low"]
            _strength_high = self.params["local_gamma"]["strength_high"]

            def _gamma_scale(x, y):
                return np.exp(
                    np.random.uniform(
                        np.log(x[y] // _scale_param[0]),
                        np.log(x[y] // _scale_param[1]),
                    )
                )

            def _gamma_strength():
                if np.random.uniform() < 0.5:
                    return np.random.uniform(_strength_low[0], _strength_low[1])
                else:
                    return np.random.uniform(_strength_high[0], _strength_high[1])

            tr_transforms.append(
                LocalGammaTransform(
                    scale=_gamma_scale,
                    gamma=_gamma_strength,
                    loc=self.params["local_gamma"]["loc"],
                    same_for_all_channels=self.params["local_gamma"][
                        "same_for_all_channels"
                    ],
                    p_per_sample=self.params["local_gamma"]["p_per_sample"],
                    p_per_channel=self.params["local_gamma"]["p_per_channel"],
                )
            )

        if self.params["do_sharpening"]:
            tr_transforms.append(
                SharpeningTransform(
                    strength=self.params["sharpening"]["strength"],
                    same_for_each_channel=self.params["sharpening"][
                        "same_for_each_channel"
                    ],
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
        logger.info(f"Training Transforms: \n{transforms}")
        return transforms
