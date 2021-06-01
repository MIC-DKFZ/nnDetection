import os
import copy
from typing import Callable, Sequence, List

import torch
import numpy as np
from loguru import logger

from nndet.planning.estimator import MemoryEstimator, MemoryEstimatorDetection
from nndet.planning.architecture.boxes.base import BoxC001
from nndet.planning.architecture.boxes.utils import (
    proxy_num_boxes_in_patch,
    scale_with_abs_strides,
    )
from nndet.core.boxes import (
    get_anchor_generator,
    expand_to_boxes,
    box_center,
    box_size_np,
    permute_boxes,
    )


class BoxC002(BoxC001):
    def __init__(self,
                 preprocessed_output_dir: os.PathLike,
                 save_dir: os.PathLike,
                 network_cls: Callable,
                 estimator: MemoryEstimator = MemoryEstimatorDetection(),
                 model_cfg: dict = None,
                 **kwargs,
                 ):
        super().__init__(
            preprocessed_output_dir=preprocessed_output_dir,
            save_dir=save_dir,
            network_cls=network_cls,
            estimator=estimator,
            model_cfg=model_cfg,
            **kwargs
            )

    def create_default_settings(self):
        """
        Generate default settings for the architecture
        """
        super().create_default_settings()
        self.architecture_kwargs["start_channels"] = 48 if self.dim == 2 else 32
        self.architecture_kwargs["fpn_channels"] = \
            self.architecture_kwargs["start_channels"] * 4
        self.architecture_kwargs["head_channels"] = \
            self.architecture_kwargs["fpn_channels"]
        self.batch_size = 16 if self.dim == 2 else 4
        self.min_feature_map_size = 8 if self.dim == 2 else 4
        self.num_decoder_level = 5 if self.dim == 2 else 4

    def get_anchor_init(self, boxes: torch.Tensor) -> Sequence[Sequence[int]]:
        """
        Initialize anchors sizes for optimization

        Args:
            boxes: scales and transposed boxes

        Returns:
            Sequence[Sequence[int]]: anchor initialization
        """
        box_dim = int(boxes.shape[1]) // 2
        return [(4, 8, 16), ] * box_dim

    def process_properties(self, **kwargs):
        """
        Load dataset properties and extract information
        """
        logger.info("Processing dataset properties")
        self.all_boxes = [case["boxes"] for case_id, case
                          in self.dataset_properties["instance_props_per_patient"].items()]
        self.all_spacings = [case["original_spacing"] for case_id, case
                             in self.dataset_properties["instance_props_per_patient"].items()]
        self.num_instances_per_case = {case_id: sum(case["num_instances"].values())
                                for case_id, case in self.dataset_properties["instance_props_per_patient"].items()}

        self.all_ious = self.dataset_properties["all_ious"]
        self.class_ious = self.dataset_properties["class_ious"]
        self.num_instances = self.dataset_properties["num_instances"]
        self.dim = self.dataset_properties["dim"]

        self.architecture_kwargs["classifier_classes"] = \
            len(self.dataset_properties["class_dct"])
        self.architecture_kwargs["seg_classes"] = \
            self.architecture_kwargs["classifier_classes"]
        self.architecture_kwargs["in_channels"] = \
            len(self.dataset_properties["modalities"])
        self.architecture_kwargs["dim"] = \
            self.dataset_properties["dim"]

    def plan(self,
             target_spacing_transposed: Sequence[float],
             median_shape_transposed: Sequence[float],
             transpose_forward: Sequence[int],
             mode: str = '3d',
             ) -> dict:
        """
        Plan network architecture, anchors, patch size and batch size

        Args:
            target_spacing_transposed: spacing after data is transposed and resampled
            median_shape_transposed: median shape after data is
                transposed and resampled
            transpose_forward: new ordering of axes for forward pass
            mode: mode to use for planning ('3d' | '2d')

        Returns:
            dict: training and architecture information

        See Also:
            :method:`_plan_architecture`, :method:`_plan_anchors`
        """
        if mode == "2d":
            logger.info("Running 2d mode")
            self.process_properties()
            kwargs_2d = self.activate_2d_mode(
                transpose_forward=transpose_forward,
                target_spacing_transposed=target_spacing_transposed,
                median_shape_transposed=median_shape_transposed,
            )
            res = super().plan(**kwargs_2d)
        else:
            res = super().plan(
                transpose_forward=transpose_forward,
                target_spacing_transposed=target_spacing_transposed,
                median_shape_transposed=median_shape_transposed,
            )
        return res

    def activate_2d_mode(self,
                         target_spacing_transposed: Sequence[float],
                         median_shape_transposed: Sequence[float],
                         transpose_forward: Sequence[int],
                         ) -> dict:
        target_spacing_transposed = target_spacing_transposed[1:]
        median_shape_transposed = median_shape_transposed[1:]
        keep = copy.copy(transpose_forward[1:])
        transpose_forward = [t - 1 for t in keep]

        keep_box = [0, 0, 0, 0]
        for idx, k in enumerate(keep):
            if k < 2:
                keep_box[idx] = k
                keep_box[idx + 2] = k + 2
            else:
                keep_box[idx] = 2 * k
                keep_box[idx + 2] = 2 * k + 1

        self.all_boxes = [b[:, keep_box] if (not isinstance(b, list) and b.shape[1] == 6) else b
                          for b in self.all_boxes]
        self.all_spacings = [c[keep] if len(c) == 3 else c for c in self.all_spacings]

        self.dim = 2
        self.architecture_kwargs["dim"] = self.dim
        return {
            "target_spacing_transposed": target_spacing_transposed,
            "median_shape_transposed": median_shape_transposed,
            "transpose_forward": transpose_forward,
        }

    def _plan_architecture(self,
                           target_spacing_transposed: Sequence[float],
                           target_median_shape_transposed: Sequence[float],
                           transpose_forward: Sequence[int],
                           **kwargs,
                           ) -> Sequence[int]:
        """
        Plan patch size and main aspects of the architecture
        Fills entries in :param:`self.architecture_kwargs`:
            `conv_kernels`
            `strides`
            `decoder_levels`

        Args:
            target_spacing_transposed: spacing after data is transposed and resampled
            target_median_shape_transposed: median shape after data is 
                transposed and resampled
        
        Returns:
            Sequence[int]: patch size to use for training
        """
        self.estimator.batch_size = self.batch_size
        patch_size = np.asarray(self._get_initial_patch_size(
            target_spacing_transposed, target_median_shape_transposed))
        first_run = True
        while True:
            if first_run:
                pass
            else:
                patch_size = self._decrease_patch_size(
                    patch_size, target_median_shape_transposed, pooling, must_be_divisible_by)
            num_pool_per_axis, pooling, convs, patch_size, must_be_divisible_by = \
                self.plan_pool_and_conv_pool_late(patch_size, target_spacing_transposed)
            self.architecture_kwargs["conv_kernels"] = convs
            self.architecture_kwargs["strides"] = pooling
            num_resolutions = len(self.architecture_kwargs["conv_kernels"])

            decoder_levels_start = min(max(1, num_resolutions - self.num_decoder_level), self.min_decoder_level)
            self.architecture_kwargs["decoder_levels"] = \
                tuple([i for i in range(decoder_levels_start, num_resolutions)])
            _, fits_in_mem = self.estimator.estimate(
                min_shape=must_be_divisible_by,
                target_shape=patch_size,
                in_channels=self.architecture_kwargs["in_channels"],
                network=self.network_cls.from_config_plan(
                    model_cfg=self.model_cfg,
                    plan_arch=self.architecture_kwargs,
                    plan_anchors=self.get_anchors_for_estimation()),
                optimizer_cls=torch.optim.Adam,
                num_instances=self._estimte_num_instances_per_patch(
                    patch_size=patch_size,
                    target_spacing_transposed=target_spacing_transposed,
                    transpose_forward=transpose_forward,
                    ),
                )
            if fits_in_mem:
                break
            first_run = False
        logger.info(f"decoder levels: {self.architecture_kwargs['decoder_levels']}; \n"
                    f"pooling strides: {self.architecture_kwargs['strides']}; \n"
                    f"kernel sizes: {self.architecture_kwargs['conv_kernels']}; \n"
                    f"patch size: {patch_size}; \n")
        return patch_size

    def _estimte_num_instances_per_patch(self,
                                         patch_size,
                                         target_spacing_transposed,
                                         transpose_forward,
                                         ) -> int:
        max_instances_per_image = []
        for boxes in self._get_scaled_boxes(
            target_spacing_transposed=target_spacing_transposed,
            transpose_forward=transpose_forward,
            cat=False,
            ):
            max_instances_per_image.append(
                max(proxy_num_boxes_in_patch(torch.from_numpy(boxes), patch_size)).item())
        return max(max_instances_per_image)

    def _plan_anchors(self,
                      target_spacing_transposed: Sequence[float],
                      transpose_forward: Sequence[int],
                      **kwargs,
                      ) -> dict:
        """
        Optimize anchors
        """
        boxes_np_full = self._get_scaled_boxes(
            target_spacing_transposed=target_spacing_transposed,
            transpose_forward=transpose_forward,
        )

        boxes_np = self.filter_boxes(boxes_np_full)
        logger.info(f"Filtered {boxes_np_full.shape[0] - boxes_np.shape[0]} "
                    f"boxes, {boxes_np.shape[0]} boxes remaining for anchor "
                    "planning.")
        boxes_torch = torch.from_numpy(boxes_np).float()
        boxes_torch = boxes_torch - expand_to_boxes(box_center(boxes_torch))
        anchor_generator = get_anchor_generator(self.dim, s_param=True)

        rel_strides = self.architecture_kwargs["strides"]
        filt_rel_strides = [[1] * self.dim, *rel_strides]
        filt_rel_strides = [filt_rel_strides[i] for i in self.architecture_kwargs["decoder_levels"]]
        strides = np.cumprod(filt_rel_strides, axis=0) / np.asarray(rel_strides[0])

        params = self.find_anchors(boxes_torch, strides.astype(np.int32), anchor_generator)
        scaled_params = {key: scale_with_abs_strides(item, strides, dim_idx) for dim_idx, (key, item) in enumerate(params.items())}
        logger.info(f"Determined Anchors: {params}; Results in params: {scaled_params}")
        self.anchors = scaled_params
        self.anchors["stride"] = 1
        return self.anchors
    
    def _get_scaled_boxes(self,
                          target_spacing_transposed: Sequence[float],
                          transpose_forward: Sequence[int],
                          cat: bool = True,
                          ) -> np.ndarray:
        """
        training is conducted in preprocessed image space and thus
        we need to scale the extracted boxes to compensate for resampling
        """
        boxes_np_list = []
        for spacing, boxes in zip(self.all_spacings, self.all_boxes):
            if not isinstance(boxes, list) and boxes.size > 0:
                spacing_transposed = np.asarray(spacing)[transpose_forward]
                scaling_transposed = spacing_transposed / np.asarray(target_spacing_transposed)
                boxes_transposed = permute_boxes(np.asarray(boxes), dims=transpose_forward)
                boxes_np_list.append(boxes_transposed * expand_to_boxes(scaling_transposed))
        if cat:
            return np.concatenate(boxes_np_list).astype(np.float32)
        else:
            return boxes_np_list

    @staticmethod
    def _get_initial_patch_size(target_spacing_transposed: np.ndarray,
                                target_median_shape_transposed: Sequence[int],
                                ) -> List[int]:
        """
        Generate initial patch which relies on the spacing of underlying images.
        This is based on the fact that most acquisition protocols are optimized
        to focus on the most importatnt aspects.
        
        Returns:
            List[int]: initial patch size
        """
        voxels_per_mm = 1 / np.array(target_spacing_transposed)

        # normalize voxels per mm
        input_patch_size = voxels_per_mm / voxels_per_mm.mean()

        # create an isotropic patch of size 512x512x512mm
        input_patch_size *= 1 / min(input_patch_size) * 512  # to get a starting value
        input_patch_size = np.round(input_patch_size).astype(np.int32)

        # clip it to the median shape of the dataset because patches larger
        # then that make not much sense and account for recangular patches
        if len(target_spacing_transposed) > 2:
            lowres_axis = np.argmax(target_spacing_transposed)
            isotropic_axes = list(range(len(target_median_shape_transposed)))
            isotropic_axes.pop(lowres_axis)
            min_isotropic_axes_shape = min([target_median_shape_transposed[t] for t in isotropic_axes])
            lowres_shape = target_median_shape_transposed[lowres_axis]
        else:
            lowres_axis = -1
            lowres_shape = None
            min_isotropic_axes_shape = min(target_median_shape_transposed)

        initial_patch_size = []
        for i in range(len(target_median_shape_transposed)):
            if i == lowres_axis:
                assert lowres_shape is not None
                initial_patch_size.append(min(input_patch_size[i], lowres_shape))
            else:
                initial_patch_size.append(min(input_patch_size[i], min_isotropic_axes_shape))
        initial_patch_size = np.round(initial_patch_size).astype(np.int32)
        logger.info(f"Using initial patch size: {initial_patch_size}")
        return initial_patch_size

    def plot_box_distribution(self, 
                              target_spacing_transposed: Sequence[float],
                              transpose_forward: Sequence[int],
                              **kwargs):
        """
        Plot histogram with ground truth bounding box distribution for
        all axis
        """
        super().plot_box_distribution()
        try:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("Failed to import matplotlib continue anyway.")
            plt = None

        if plt is not None:
            if isinstance(self.all_boxes, list):
                _boxes = np.concatenate(
                    [b for b in self.all_boxes if not isinstance(b, list) and b.size > 0], axis=0)
                dists = box_size_np(_boxes)
            else:
                dists = box_size_np(self.all_boxes)

            if dists.shape[1] == 3:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(dists[:, 0], dists[:, 1], dists[:, 2])
                ax.set_title(f"Transpose forward {transpose_forward}")
                plt.savefig(self.save_dir / f'bbox_sizes_3d_orig.png')
                plt.close()

                dists = box_size_np(self._get_scaled_boxes(
                    target_spacing_transposed, transpose_forward))
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(dists[:, 0], dists[:, 1], dists[:, 2])
                plt.savefig(self.save_dir / f'bbox_sizes_3d.png')
                plt.close()
            else:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(dists[:, 0], dists[:, 1])
                ax.grid(True)
                ax.set_title(f"Transpose forward {transpose_forward}")
                plt.savefig(self.save_dir / f'bbox_sizes_2d_orig.png')
                plt.close()

                dists = box_size_np(self._get_scaled_boxes(
                    target_spacing_transposed, transpose_forward))
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(dists[:, 0], dists[:, 1])
                ax.grid(True)
                plt.savefig(self.save_dir / f'bbox_sizes_2d.png')
                plt.close()
