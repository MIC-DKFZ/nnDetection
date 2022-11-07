import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from nndet.core.abstract import AbstractDetector
from nndet.core.boxes import box_cxcywhczd_to_xyxyzz, box_iou


class AbstractDETR(AbstractDetector):
    def train_step(
        self,
        images: Tensor,
        targets: dict,
        batch_num: int,
    ) -> Dict[str, torch.Tensor]:
        """
        See `self.train_step_with_features` for more info
        """
        losses, _, _ = self.train_step_with_features(
            images=images,
            targets=targets,
            predict=False,
            batch_num=batch_num,
        )
        return losses

    @torch.no_grad()
    def validation_step(
        self,
        images: Tensor,
        targets: dict,
        batch_num: bool,
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        See `self.train_step_with_features` for more info
        """
        losses, prediction, _ = self.train_step_with_features(
            images=images,
            targets=targets,
            predict=True,
            batch_num=batch_num,
        )
        return losses, prediction

    @torch.no_grad()
    def inference_step(
        self,
        images: Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        See `inference_step_with_features` for more info
        """
        prediction, _ = self.inference_step_with_features(images=images, **kwargs)
        return prediction

    def log_queries(self, target_classes: List[Tensor], pred_logits: Tensor, batch_num: int):
        num_objects = np.array([len(class_tensor) for class_tensor in target_classes], dtype=int)
        query_preds = pred_logits.argmax(dim=2)
        query_scores = (
            torch.gather(
                pred_logits,
                2,
                query_preds.unsqueeze(-1),
            )
            .squeeze(-1)
            .detach()
            .cpu()
            .numpy()
            .T
        )
        query_preds = query_preds.detach().cpu().numpy().T
        df_dict = dict([(f"query{i}_pred", query_pred) for i, query_pred in enumerate(query_preds)])
        df_dict.update(dict([(f"query{i}_score:", query_score) for i, query_score in enumerate(query_scores)]))
        df_dict.update({"num gt": num_objects, "batch_num": np.repeat(batch_num, len(num_objects))})
        df = pd.DataFrame(df_dict)
        output_path = "query_preds.csv"
        df.to_csv(output_path, mode="a", header=not os.path.exists(output_path))
        return

    def log_ious(self, pred_boxes: Tensor):
        # Calculate iou matrix between box predictions
        box_ious = []
        for i, boxes in enumerate(pred_boxes):
            box_edges = box_cxcywhczd_to_xyxyzz(boxes)
            box_ious.append(box_iou(box_edges, box_edges))
        pred_iou = torch.stack(box_ious).flatten(1).detach().cpu().numpy()
        df = pd.DataFrame(pred_iou)
        output_path = "prediction_ious.csv"
        df.to_csv(output_path, mode="a", header=not os.path.exists(output_path))
        return

    @torch.no_grad()
    def postprocess_for_inference(
        self,
        images: torch.Tensor,
        pred_detection: Dict[str, torch.Tensor],
        anchors=None,
        pred_seg=None,
    ):
        """
            Returns:
                    Dict: post processed predictions
                        'pred_boxes': List[Tensor]: predicted bounding boxes for each
                            image List[[R, dim * 2]]
                        'pred_scores': List[Tensor]: predicted probability for
                            the class List[[R]]
                        'pred_labels': List[Tensor]: predicted class List[[R]]
                        'pred_seg': Tensor: predicted segmentation [N, C, dims]
            """ ""

        prediction = self.head.postprocess_for_inference(images, pred_detection)
        return prediction
