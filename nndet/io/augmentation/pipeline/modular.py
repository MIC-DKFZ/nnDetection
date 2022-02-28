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
from batchgenerators.transforms.noise_transforms import (
    GaussianBlurTransform,
    GaussianNoiseTransform,
)
from batchgenerators.transforms.resample_transforms import (
    SimulateLowResolutionTransform,
)
from batchgenerators.transforms.spatial_transforms import (
    MirrorTransform,
    SpatialTransform,
)
from batchgenerators.transforms.utility_transforms import (
    NumpyToTensor,
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

        # TODO: do transform param
        tr_transforms.append(
            GaussianNoiseTransform(
                p_per_sample=self.params.get("p_per_sample_gaussian_noise"),
            ),  # TODO: make noise_variance a config key
        )

        # TODO: do transform param
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

        # TODO: do transform param
        tr_transforms.append(
            BrightnessMultiplicativeTransform(
                p_per_sample=self.params.get("p_per_sample_brightness_mul"),
                multiplier_range=self.params.get("brightness_mul_multiplier_range"),
            ),  # TODO: make per_channel a config key
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

        # TODO: do transform param
        tr_transforms.append(
            ContrastAugmentationTransform(
                contrast_range=self.params.get("contrast_range"),
                p_per_sample=self.params.get("p_per_sample_contrast"),
            ),  # TODO: make per_channel a config key
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
                    per_channel=True,  # TODO: make per_channel a config key
                    retain_stats=self.params.get("gamma_retain_stats"),
                    p_per_sample=self.params["p_gamma_inverted"],
                ),
            )  # inverted gamma

        if self.params.get("do_gamma"):
            tr_transforms.append(
                GammaTransform(
                    gamma_range=self.params.get("gamma_range"),
                    invert_image=False,
                    per_channel=True,  # TODO: make per_channel a config key
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
