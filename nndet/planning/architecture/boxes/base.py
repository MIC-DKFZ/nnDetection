import os
from pathlib import Path
from abc import abstractmethod
from typing import Type, Dict, Sequence, List, Callable, Tuple

import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from torchvision.models.detection.rpn import AnchorGenerator

from nndet.utils.info import SuppressPrint

with SuppressPrint():
    from nnunet.experiment_planning.common_utils import get_pool_and_conv_props

from nndet.io.load import load_pickle
from nndet.arch.abstract import AbstractModel
from nndet.planning.estimator import MemoryEstimator, MemoryEstimatorDetection
from nndet.planning.architecture.abstract import ArchitecturePlanner
from nndet.core.boxes import (
    get_anchor_generator,
    expand_to_boxes,
    box_center,
    box_size,
    compute_anchors_for_strides,
    box_iou,
    box_size_np,
    box_area_np,
    permute_boxes,
    )
from nndet.planning.architecture.boxes.utils import (
    fixed_anchor_init,
    scale_with_abs_strides,
    )


class BaseBoxesPlanner(ArchitecturePlanner):
    def __init__(self,
                 preprocessed_output_dir: os.PathLike,
                 save_dir: os.PathLike,
                 network_cls: Type[AbstractModel] = None,
                 estimator: MemoryEstimator = None,
                 **kwargs,
                 ):
        """
        Plan the architecture for training

        Args:
            min_feature_map_length (int): minimal size of feature map in bottleneck
        """
        super().__init__(**kwargs)

        self.preprocessed_output_dir = Path(preprocessed_output_dir)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)        

        self.network_cls = network_cls
        self.estimator = estimator

        self.dataset_properties = load_pickle(
            self.preprocessed_output_dir / "properties" / 'dataset_properties.pkl')

        # parameters initialized from process properties
        self.all_boxes: np.ndarray = None
        self.all_ious: np.ndarray = None
        self.class_ious: Dict[str, np.ndarray] = None
        self.num_instances: Dict[int, int] = None
        self.dim: int = None
        self.architecture_kwargs: dict = {}
        self.transpose_forward = None

    def process_properties(self, **kwargs):
        """
        Load dataset properties and extract information
        """
        assert self.transpose_forward is not None
        boxes = [case["boxes"] for case_id, case
                 in self.dataset_properties["instance_props_per_patient"].items()]
        self.all_boxes = np.concatenate([b for b in boxes if not isinstance(b, list) and b.size > 0], axis=0)
        self.all_boxes = permute_boxes(self.all_boxes, dims=self.transpose_forward)
        self.all_ious = self.dataset_properties["all_ious"]
        self.class_ious = self.dataset_properties["class_ious"]
        self.num_instances = self.dataset_properties["num_instances"]
        self.num_instances_per_case = {case_id: sum(case["num_instances"].values())
                                       for case_id, case in self.dataset_properties["instance_props_per_patient"].items()}
        self.dim = self.dataset_properties["dim"]

        self.architecture_kwargs["classifier_classes"] = \
            len(self.dataset_properties["class_dct"])
        self.architecture_kwargs["seg_classes"] = \
            self.architecture_kwargs["classifier_classes"]
        self.architecture_kwargs["in_channels"] = \
            len(self.dataset_properties["modalities"])
        self.architecture_kwargs["dim"] = \
            self.dataset_properties["dim"]

    def plot_box_distribution(self, **kwargs):
        """
        Plot histogram with ground truth bounding box distribution for
        all axis
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            plt = None
            logger.error("Failed to import matplotlib continue anyway.")
        if plt is not None:
            if isinstance(self.all_boxes, list):
                _boxes = np.concatenate(
                    [b for b in self.all_boxes if not isinstance(b, list) and b.size > 0], axis=0)
                dists = box_size_np(_boxes)
            else:
                dists = box_size_np(self.all_boxes)
            for axis in range(dists.shape[1]):
                dist = dists[:, axis]
                plt.hist(dist, bins=100)
                plt.savefig(
                    self.save_dir / f'bbox_sizes_axis_{axis}.png')
                plt.xscale('log')
                plt.savefig(
                    self.save_dir / f'bbox_sizes_axis_{axis}_xlog.png')
                plt.yscale('log')
                plt.savefig(
                    self.save_dir / f'bbox_sizes_axis_{axis}_xylog.png')
                plt.close()

    def plot_box_area_distribution(self, **kwargs):
        """
        Plot histogram of areas of all ground truth boxes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            plt = None
            logger.error("Failed to import matplotlib continue anyway.")
        if plt is not None:
            if isinstance(self.all_boxes, list):
                _boxes = np.concatenate(
                    [b for b in self.all_boxes if not isinstance(b, list) and b.size > 0], axis=0)
                area = box_area_np(_boxes)
            else:
                area = box_area_np(self.all_boxes)
            plt.hist(area, bins=100)
            plt.savefig(self.save_dir / f'box_areas.png')
            plt.xscale('log')
            plt.savefig(self.save_dir / f'box_areas_xlog.png')
            plt.yscale('log')
            plt.savefig(self.save_dir / f'box_areas_xylog.png')
            plt.close()

    def plot_class_distribution(self, **kwargs):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            plt = None
            logger.error("Failed to import matplotlib continue anyway.")
        if plt is not None:
            num_instances_dict = self.dataset_properties["num_instances"]
            num_instances = []
            classes = []
            for key, item in num_instances_dict.items():
                num_instances.append(item)
                classes.append(str(key))
            ind = np.arange(len(num_instances))
            
            plt.bar(ind, num_instances)
            plt.xlabel("Classes")
            plt.ylabel("Num Instances")
            plt.xticks(ind, classes)
            plt.savefig(self.save_dir / f'num_classes.png')
            plt.yscale('log')
            plt.savefig(self.save_dir / f'num_classes_ylog.png')
            plt.close()

    def plot_instance_distribution(self, **kwargs):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            plt = None
            logger.error("Failed to import matplotlib continue anyway.")
        if plt is not None:
            num_instances_per_case = list(self.num_instances_per_case.values())
            plt.hist(num_instances_per_case, bins=100, range=(0, 100))
            plt.savefig(self.save_dir / f'instances_per_case.png')
            plt.close()
            
            plt.hist(num_instances_per_case, bins=30, range=(0, 30))
            plt.savefig(self.save_dir / f'instances_per_case_0_30.png')
            plt.close()

            plt.hist(num_instances_per_case, bins=11, range=(0, 11))
            plt.savefig(self.save_dir / f'instances_per_case_0_10.png')
            plt.close()

    @abstractmethod
    def _plan_anchors(self) -> dict:
        """
        Plan anchors hyperparameters
        """
        raise NotImplementedError

    @abstractmethod
    def _plan_architecture(self) -> Sequence[int]:
        """
        Plan architecture
        """
        raise NotImplementedError

    def plan(self, **kwargs) -> dict:
        """
        Plan architecture and training params
        """
        for key, item in kwargs.items():
            setattr(self, key, item)
        self.create_default_settings()
        if self.all_boxes is None:
            self.process_properties(**kwargs)
        self.plot_box_area_distribution(**kwargs)
        self.plot_box_distribution(**kwargs)
        self.plot_class_distribution(**kwargs)
        self.plot_instance_distribution(**kwargs)
        return {}

    def create_default_settings(self):
        pass

    def compute_class_weights(self) -> List[float]:
        """
        Compute classification weighting for inbalanced datasets
        (background samples get weight 1 / (num_classes + 1) and forground
        classes are weighted with (1 - 1 / (num_classes + 1))*(1 - ni / nall))
        where ni is the number of sampler for class i and n all
        is the number of all ground truth samples

        Returns:
            List[float]: weights
        """
        num_instances_dict = self.dataset_properties["num_instances"]
        num_classes = len(num_instances_dict)
        num_instances = [0] * num_classes
        for key, item in num_instances_dict.items():
            num_instances[int(key)] = int(item)

        bg_weight = 1 / (num_classes + 1)
        remaining_weight = 1 - bg_weight
        weights = [remaining_weight * (1 - ni / sum(num_instances)) for ni in num_instances]
        return [bg_weight] + weights

    def get_planner_id(self) -> str:
        """
        Create identifier for this planner. If available append
        :attr:`plan_tag` to the base name
        
        Returns:
            str: identifier
        """
        base =  super().get_planner_id()
        if hasattr(self, "plan_tag"):
            base = base + getattr(self, "plan_tag")
        return base


class BoxC001(BaseBoxesPlanner):
    def __init__(self,
                 preprocessed_output_dir: os.PathLike,
                 save_dir: os.PathLike,
                 network_cls: Callable,
                 estimator: MemoryEstimator = MemoryEstimatorDetection(),
                 model_cfg: dict = None,
                 **kwargs,
                 ):
        """
        Plan training architecture with heuristics

        Args:
            preprocessed_output_dir: base preprocessed directory to
                access properties and save analysis files
            save_dir: directory to save analysis plots
            network_cls: constructor of network to plan
            estimator: estimate GPU memory requirements for specific GPU
                architectures. Defaults to MemoryEstimatorDetection().
        """
        super().__init__(
            preprocessed_output_dir=preprocessed_output_dir,
            save_dir=save_dir,
            network_cls=network_cls,
            estimator=estimator,
            **kwargs,
        )
        self.additional_params = {}
        if model_cfg is None:
            model_cfg = {}
        self.model_cfg = model_cfg
        self.plan_anchor_for_estimation = fixed_anchor_init(self.dim)

    def create_default_settings(self):
        """
        Generate some default settings for the architecture
        """
        # MAX_NUM_FILTERS_2D, MAX_NUM_FILTERS_3D from nnUNet
        self.architecture_kwargs["max_channels"] = 480 if self.dim == 2 else 320
        # BASE_NUM_FEATURES_3D from nnUNet
        self.architecture_kwargs["start_channels"] = 32
        # DEFAULT_BATCH_SIZE_3D from nnUNet
        self.batch_size = 32 if self.dim == 2 else 2

        self.max_num_pool = 999
        self.min_feature_map_size = 4
        self.min_decoder_level = 2
        self.num_decoder_level = 4

        self.architecture_kwargs["fpn_channels"] = \
            self.architecture_kwargs["start_channels"] * 2
        self.architecture_kwargs["head_channels"] = \
            self.architecture_kwargs["fpn_channels"]

    def plan(self,
             target_spacing_transposed: Sequence[float],
             median_shape_transposed: Sequence[float],
             transpose_forward: Sequence[int],
             mode: str = "3d",
             ) -> dict:
        """
        Plan network architecture, anchors, patch size and batch size

        Args:
            target_spacing_transposed: spacing after data is transposed and resampled
            median_shape_transposed: median shape after data is
                transposed and resampled
            transpose_forward: new ordering of axes for forward pass
            mode: mode to use for planning (this planner only supports 3d!)

        Returns:
            dict: training and architecture information
        
        See Also:
            :method:`_plan_architecture`, :method:`_plan_anchors`
        """
        super().plan(
            transpose_forward=transpose_forward,
            target_spacing_transposed=target_spacing_transposed,
            median_shape_transposed=median_shape_transposed,
            )
        self.architecture_kwargs["class_weight"] = self.compute_class_weights()
        patch_size = self._plan_architecture(
            transpose_forward=transpose_forward,
            target_spacing_transposed=target_spacing_transposed,
            target_median_shape_transposed=median_shape_transposed,
        )

        anchors = self._plan_anchors(
            target_spacing_transposed=target_spacing_transposed,
            median_shape_transposed=median_shape_transposed,
            transpose_forward=transpose_forward,
        )        

        plan = {"patch_size": patch_size,
                "batch_size": self.batch_size,
                "architecture": {
                    "arch_name": self.network_cls.__name__,
                    **self.architecture_kwargs
                },
                "anchors": anchors,
                }
        logger.info(f"Using architecture plan: \n{plan}")
        return plan

    def _plan_anchors(self, **kwargs) -> dict:
        """
        Optimize anchors
        """
        boxes_np_full = self.all_boxes.astype(np.float32)
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

    @staticmethod
    def filter_boxes(boxes_np: np.ndarray,
                     upper_percentile: float = 99.5,
                     lower_percentile: float = 00.5,
                     ) -> np.ndarray:
        """
        Determine upper and lower percentiles of bounding box sizes for each
        axis and remove boxes which are outside the specified range

        Args:
            boxes_np (np.ndarray): bounding boxes [N, dim * 2](x1, y1, x2, y2, (z1, z2))
            upper_percentile: percentile for upper boundary. Defaults to 99.5.
            lower_percentile: percentile for lower boundary. Defaults to 00.5.
        
        Returns:
            np.ndarray: filtered boxes

        See Also:
            :func:`np.percentile`
        """
        mask = np.ones(boxes_np.shape[0]).astype(bool)
        box_sizes = box_size_np(boxes_np)
        for ax in range(box_sizes.shape[1]):
            ax_sizes = box_sizes[:, ax]
            upper_th = np.percentile(ax_sizes, upper_percentile)
            lower_th = np.percentile(ax_sizes, lower_percentile)
            ax_mask = (ax_sizes < upper_th) * (ax_sizes > lower_th)
            mask = mask * ax_mask
        return boxes_np[mask.astype(bool)]

    def find_anchors(self, 
                     boxes_torch: torch.Tensor,
                     strides: Sequence[Sequence[int]],
                     anchor_generator: AnchorGenerator,
                     ) -> Dict[str, Sequence[int]]:
        """
        Find anchors which maximize iou over dataset
        
        Args:
            boxes_torch: filtered ground truth boxes
            strides (Sequence[Sequence[int]]): strides of network to compute
                anchor sizes of lower levels
            anchor_generator (AnchorGenerator): anchor generator for generate
                the anchors

        Returns:
            Dict[Sequence[int]]: parameterization of anchors
                    `width` (Sequence[float]): width values for bounding boxes
                    `height` (Sequence[float]): height values for bounding boxes
                    (`depth` (Sequence[float]): dpeth values for bounding boxes)
        """
        import nevergrad as ng
        dim = int(boxes_torch.shape[1] // 2)

        sizes = box_size(boxes_torch)
        maxs = sizes.max(dim=0)[0]
        best_iou = 0
        # TBPSA, PSO
        for algo in ["TwoPointsDE", "TwoPointsDE", "TwoPointsDE"]:
            _best_iou = 0
            params = []
            for axis in range(dim):
                # TODO: find better initialization
                anchor_init = self.get_anchor_init(boxes_torch)
                p = ng.p.Array(init=np.asarray(anchor_init[axis]))
                p.set_integer_casting()
                # p.set_bounds(1, maxs[axis].item())
                p.set_bounds(lower=1)
                params.append(p)
            instrum = ng.p.Instrumentation(*params)
            optimizer = ng.optimizers.registry[algo](
                parametrization=instrum, budget=5000, num_workers=1)

            with torch.no_grad():
                pbar = tqdm(range(optimizer.budget), f"Anchor Opt {algo}")
                for _ in pbar:
                    x = optimizer.ask()
                    anchors = anchor_generator.generate_anchors(*x.args)
                    anchors = compute_anchors_for_strides(
                        anchors, strides=strides, cat=True)
                    anchors = anchors
                    # TODO: add checks if GPU is availabe and has enough VRAM
                    iou = box_iou(boxes_torch.cuda(), anchors.cuda())  # boxes x anchors
                    mean_iou = iou.max(dim=1)[0].mean().cpu()
                    optimizer.tell(x, -mean_iou.item())
                    pbar.set_postfix(mean_iou=mean_iou)
                    _best_iou = mean_iou
            if _best_iou > best_iou:
                best_iou = _best_iou
                recommendation = optimizer.provide_recommendation().value[0]
        return {key: list(val) for key, val in zip(["width", "height", "depth"], recommendation)}

    def get_anchor_init(self, boxes: torch.Tensor) -> Sequence[Sequence[int]]:
        """
        Initialize anchors sizes for optimization

        Args:
            boxes: scales and transposed boxes

        Returns:
            Sequence[Sequence[int]]: anchor initialization
        """
        return [(2, 4, 8)] * 3

    def _plan_architecture(self,
                           target_spacing_transposed: Sequence[float],
                           target_median_shape_transposed: Sequence[float],
                           **kwargs,
                           ) -> Sequence[int]:
        """
        Plan patchsize and main aspects of the architecture
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

            decoder_levels_start = min(max(0, num_resolutions - self.num_decoder_level), self.min_decoder_level)
            self.architecture_kwargs["decoder_levels"] = \
                tuple([i for i in range(decoder_levels_start, num_resolutions)])
            print(self.architecture_kwargs["decoder_levels"])
            print(self.get_anchors_for_estimation())
            _, fits_in_mem = self.estimator.estimate(
                min_shape=must_be_divisible_by,
                target_shape=patch_size,
                in_channels=self.architecture_kwargs["in_channels"],
                network=self.network_cls.from_config_plan(
                    model_cfg=self.model_cfg,
                    plan_arch=self.architecture_kwargs,
                    plan_anchors=self.get_anchors_for_estimation()),
                optimizer_cls=torch.optim.Adam,
            )
            if fits_in_mem:
                break
            first_run = False
        logger.info(f"decoder levels: {self.architecture_kwargs['decoder_levels']}; \n"
                    f"pooling strides: {self.architecture_kwargs['strides']}; \n"
                    f"kernel sizes: {self.architecture_kwargs['conv_kernels']}; \n"
                    f"patch size: {patch_size}; \n")
        return patch_size

    def _decrease_patch_size(self,
                             patch_size: np.ndarray,
                             target_median_shape_transposed: np.ndarray,
                             pooling: Sequence[Sequence[int]],
                             must_be_divisible_by: Sequence[int],
                             ) -> np.ndarray:
        """
        Decrease largest physical axis. If it larger than bottleneck size is
        is decreased by the minimum value to be divisable by computed pooling
        strides and will be halfed otherwise.

        Args:
            patch_size: current patch size
            target_median_shape_transposed: median shape of dataset
                correctly transposed
            pooling: pooling kernels of network
            must_be_divisible_by: necessary divisor per axis
        
        Returns:
            np.ndarray: new patch size
        """
        argsrt = np.argsort(patch_size / target_median_shape_transposed)[::-1]
        pool_fct_per_axis = np.prod(pooling, 0)
        bottleneck_size_per_axis = patch_size / pool_fct_per_axis
        reduction = []
        for i in range(len(patch_size)):
            if bottleneck_size_per_axis[i] > self.min_feature_map_size:
                reduction.append(must_be_divisible_by[i])
            else:
                reduction.append(must_be_divisible_by[i] / 2)
        patch_size[argsrt[0]] -= reduction[argsrt[0]]
        return patch_size

    @staticmethod
    def _get_initial_patch_size(target_spacing_transposed: np.ndarray,
                                target_median_shape_transposed: Sequence[int]) -> List[int]:
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

        # clip it to the median shape of the dataset because patches larger then that make not much sense
        input_patch_size = [min(i, j) for i, j in zip(
            input_patch_size, target_median_shape_transposed)]
        return np.round(input_patch_size).astype(np.int32)

    def plan_pool_and_conv_pool_late(self,
                                     patch_size: Sequence[int],
                                     spacing: Sequence[float],
                                     ) -> Tuple[List[int], List[Tuple[int]], List[Tuple[int]],
                                                Sequence[int], Sequence[int]]:
        """
        Plan pooling and convolutions of encoder network
        Axis which do not need pooling in every block are pooled as late as possible
        Uses kernel size 1 for anisotropic axis which are not reached by the fov yet

        Args:
            patch_size: target path size
            spacing: target spacing transposed

        Returns:
            List[int]: max number of pooling operations per axis
            List[Tuple[int]]: kernel sizes of pooling operations
            List[Tuple[int]]: kernel sizes of convolution layers
            Sequence[int]: patch size
            Sequence[int]: coefficient each axes needs to be divisable by
        """
        num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, \
            patch_size, must_be_divisible_by = get_pool_and_conv_props(
                spacing=spacing, patch_size=patch_size,
                min_feature_map_size=self.min_feature_map_size,
                max_numpool=self.max_num_pool)
        return num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, must_be_divisible_by

    def get_anchors_for_estimation(self):
        """
        Adjust anchor plan for varying  number of feature maps
        
        Returns:
            dict: adjusted anchor plan
        """
        num_levels = len(self.architecture_kwargs["decoder_levels"])
        anchor_plan = {"stride": 1, "aspect_ratios": (0.5, 1, 2)}
        if self.dim == 2:
            _sizes = [(16, 32, 64)] * num_levels
            anchor_plan["sizes"] = tuple(_sizes)
        else:
            _sizes = [(16, 32, 64)] * num_levels
            anchor_plan["sizes"] = tuple(_sizes)
            anchor_plan["zsizes"] = tuple(_sizes)
        return anchor_plan
