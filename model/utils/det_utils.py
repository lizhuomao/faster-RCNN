from torch import Tensor
import torch
import math
from typing import Tuple, List


def encode_boxes(reference_boxes, proposals, weights):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    """
    Encode a set of proposals with respect to some reference boxes
    :param reference_boxes: ground truth
    :param proposals: anchor
    :param weights: weights
    :return:
    """
    wx, wy, ww, wh = weights[0], weights[1], weights[2], weights[3]
    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)
    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    ex_ctr_x = proposals_x1 + 0.5 * exwidths



class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into the representation used for training the regressors
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        # type: (Tuple[float, float, float, float], float) -> None
        """
        initialization
        :param weights:
        :param bbox_xform_clip:
        """
        self.weight = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode_single(self, reference_boxes, proposals):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        """

        :param reference_boxes: ground truths
        :param proposals: anchors
        :return: targets
        """
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weight = torch.as_tensor(self.weight, dtype=dtype, device=device)

    def encode(self, reference_boxes, proposals):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        """
        combine anchors and corresponding ground truth to compute regression parameters
        :param reference_boxes: ground truths
        :param proposals: anchors
        :return: regression parameters
        """
        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)

        targets = self.encode_single(reference_boxes, proposals)
        return targets.split(boxes_per_image, 0)
