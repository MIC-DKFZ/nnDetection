import copy
from loguru import logger

import torch

from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001
from nndet.ptmodule import MODULE_REGISTRY

from nndet.core.retina import BaseRetinaNet
from nndet.core.boxes.matcher import IoUMatcher
from nndet.core.boxes.sampler import HardNegativeSamplerBatched
from nndet.core.boxes.anchors import AnchorGeneratorType
from nndet.core.boxes.coder import BoxCoderND
from nndet.core.boxes.anchors import get_anchor_generator
from nndet.core.boxes.utils import box_iou

from nndet.arch.conv import Generator, ConvInstanceRelu, ConvGroupRelu

from nndet.core.rois.head import RoIModule
from nndet.core.rois.pooler import RoIAlignNaiveAssign
from nndet.arch.heads.classifier.roi import RoIClassifierTwoMLP
from nndet.arch.heads.regressor.roi_single import RoIRegressorConv
from nndet.arch.heads.comb.roi import RoIBoxHead
from nndet.core.boxes.coder import BoxCoderND
from nndet.arch.conv import Generator, ConvInstanceRelu, ConvGroupRelu
from nndet.core.boxes.utils import box_iou
from nndet.core.boxes.matcher import IoUMatcher
from nndet.core.boxes.sampler import NegativeSampler

from nndet.core.rcnn import RCNN


@MODULE_REGISTRY.register
class DummyRCNN(RetinaUNetV001):
    @classmethod
    def from_config_plan(cls,
                         model_cfg: dict,
                         plan_arch: dict,
                         plan_anchors: dict,
                         **kwargs,
                         ):
        #####
        # Build PRN
        #####
        logger.info(f"Architecture overwrites: {model_cfg['plan_arch_overwrites']} "
                    f"Anchor overwrites: {model_cfg['plan_anchors_overwrites']}")
        logger.info(f"Building architecture according to plan of {plan_arch.get('arch_name', 'not_found')}")
        plan_arch.update(model_cfg["plan_arch_overwrites"])
        plan_anchors.update(model_cfg["plan_anchors_overwrites"])
        logger.info(f"Start channels: {plan_arch['start_channels']}; "
                    f"head channels: {plan_arch['head_channels']}; "
                    f"fpn channels: {plan_arch['fpn_channels']}")

        _plan_anchors = copy.deepcopy(plan_anchors)
        coder = BoxCoderND(weights=(1.,) * (plan_arch["dim"] * 2))
        s_param = False if ("aspect_ratios" in _plan_anchors) and \
                           (_plan_anchors["aspect_ratios"] is not None) else True
        anchor_generator = get_anchor_generator(
            plan_arch["dim"], s_param=s_param)(**_plan_anchors)

        encoder = cls._build_encoder(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            )
        decoder = cls._build_decoder(
            encoder=encoder,
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            )
        matcher = cls.matcher_cls(
            similarity_fn=box_iou,
            **model_cfg["matcher_kwargs"],
            )

        classifier = cls._build_head_classifier(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            anchor_generator=anchor_generator,
        )
        regressor = cls._build_head_regressor(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            anchor_generator=anchor_generator,
        )
        head = cls._build_head(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            classifier=classifier,
            regressor=regressor,
            coder=coder
        )
        segmenter = cls._build_segmenter(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            decoder=decoder,
        )

        detections_per_img = plan_arch.get("detections_per_img", 100)
        score_thresh = plan_arch.get("score_thresh", 0)
        topk_candidates = plan_arch.get("topk_candidates", 10000)
        remove_small_boxes = plan_arch.get("remove_small_boxes", 0.01)
        nms_thresh = plan_arch.get("nms_thresh", 0.6)

        logger.info(f"Model Inference Summary: \n"
                    f"detections_per_img: {detections_per_img} \n"
                    f"score_thresh: {score_thresh} \n"
                    f"topk_candidates: {topk_candidates} \n"
                    f"remove_small_boxes: {remove_small_boxes} \n"
                    f"nms_thresh: {nms_thresh}",
                    )

        rpn = BaseRetinaNet(
            dim=plan_arch["dim"],
            encoder=encoder,
            decoder=decoder,
            head=head,
            anchor_generator=anchor_generator,
            matcher=matcher,
            num_classes=plan_arch["classifier_classes"],
            decoder_levels=plan_arch["decoder_levels"],
            segmenter=segmenter,
            # model_max_instances_per_batch_element (in mdt per img, per class; here: per img)
            detections_per_img=detections_per_img,
            score_thresh=score_thresh,
            topk_candidates=topk_candidates,
            remove_small_boxes=remove_small_boxes,
            nms_thresh=nms_thresh,
        )

        #####
        # Build RoI Module
        #####
        output_size = (7, 7)
        conv = Generator(ConvInstanceRelu, 2)
        coder = BoxCoderND(weights=(1.,) * (2 * 2))
        classifier = RoIClassifierTwoMLP(
            conv=conv,
            in_channels = plan_arch["fpn_channels"] * 7 * 7,
            internal_channels=plan_arch["fpn_channels"],
            num_classes=1
        )
        regressor = RoIRegressorConv(
            conv=conv,
            in_channels=plan_arch["fpn_channels"],
            internal_channels=plan_arch["fpn_channels"],
        )
        roi_head = RoIBoxHead(
            classifier=classifier,
            regressor=regressor,
            coder=coder,
        )
        matcher = IoUMatcher(
                low_threshold=0.5,
                high_threshold=0.5,
                allow_low_quality_matches=False,
            )
        pooler = RoIAlignNaiveAssign(
            output_size=output_size
        )
        sampler = NegativeSampler(
            batch_size_per_image=32,
            positive_fraction=0.5,
        )
        roi_module = RoIModule(
            box_head=roi_head,
            matcher=matcher,
            pooler=pooler,
            sampler=sampler,
            num_classes=plan_arch["classifier_classes"],
            decoder_levels=plan_arch["decoder_levels"],
            gt_to_proposals=True,
        )

        return RCNN(
            rpn=rpn,
            roi_module=roi_module,
        )

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            batch = self.pre_trafo(**batch)
            targets = {
                    "target_boxes": batch["boxes"],
                    "target_classes": batch["classes"],
                    "target_seg": batch['target'][:, 0]  # Remove channel dimension
                }
            prediction = self.model.inference_step(
                images=batch["data"],
                targets=targets,
                predict=True,
                batch_num=batch_idx,
            )

        self.evaluation_step(prediction=prediction, targets=targets)
        return {"loss": 0}
