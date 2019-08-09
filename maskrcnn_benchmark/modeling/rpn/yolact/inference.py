import torch

import pycocotools.mask as mask_util
import numpy as np

from ..utils import permute_and_flatten
from .yolact import _ACTIVATION_FUNC

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.rpn.retinanet.inference import RetinaNetPostProcessor
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.utils import cat, jaccard
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist, boxlist_nms, remove_small_boxes
from maskrcnn_benchmark.structures.mask_ops import crop_zero_out, convert_binary_to_rle
from maskrcnn_benchmark.layers.misc import interpolate

DEBUG = False
if DEBUG:
    import matplotlib.pyplot as plt
    from torchvision.utils import save_image

class YolactPostProcessor(RetinaNetPostProcessor):
    """
    Performs post-processing on the outputs of the RetinaNet boxes and ProtoNet.
    This is only used in the testing.
    """
    def __init__(
        self,
        pre_nms_thresh,
        pre_nms_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        min_size,
        num_classes,
        mask_activation,
        mask_threshold,
        box_coder=None,
    ):
        super(YolactPostProcessor, self).__init__(
            pre_nms_thresh, pre_nms_top_n, nms_thresh, fpn_post_nms_top_n, min_size, \
                num_classes, box_coder
        )
        self.mask_activation = mask_activation
        self.mask_threshold = mask_threshold

    def forward_for_single_feature_map(self, anchors, box_cls, box_regression, coeffs):
        """
        Arguments:
            anchors: list[BoxList] N, A * H * W
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
            coeffs: tensor of size N, A * K, H, W
        """
        N, _, H, W = box_cls.shape
        A = box_regression.size(1) // 4
        C = box_cls.size(1) // A
        K = coeffs.size(1) // A

        # put in the same format as anchors (N, H*W*A, C)
        box_cls = permute_and_flatten(box_cls, N, A, C, H, W)
        box_cls = box_cls.sigmoid()
        # box regression is class-agnostic
        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)
        # Q: Seems redundant?
        # box_regression = box_regression.reshape(N, -1, 4)

        coeffs = permute_and_flatten(coeffs, N, A, K, H, W)

        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        results = []
        for per_box_cls, per_box_regression, per_coeffs, \
            per_pre_nms_top_n, per_candidate_inds, per_anchors in zip(
                box_cls, box_regression, coeffs, \
                    pre_nms_top_n, candidate_inds, anchors):

            if cfg.MODEL.YOLACT.USE_FAST_NMS:
                per_class = None
                detections = self.box_coder.decode(
                    per_box_regression,
                    per_anchors.bbox
                )
            else:
                # Sort and select TopN
                per_box_cls = per_box_cls[per_candidate_inds]

                per_box_cls, top_k_indices = \
                        per_box_cls.topk(per_pre_nms_top_n, sorted=False)

                per_candidate_nonzeros = \
                        per_candidate_inds.nonzero()[top_k_indices, :]

                per_box_loc = per_candidate_nonzeros[:, 0]
                per_class = per_candidate_nonzeros[:, 1]
                per_class += 1

                detections = self.box_coder.decode(
                    per_box_regression[per_box_loc, :].view(-1, 4),
                    per_anchors.bbox[per_box_loc, :].view(-1, 4)
                )

                per_coeffs = per_coeffs[per_box_loc, :].view(-1, K)

            image_size = per_anchors.size
            boxlist = BoxList(detections, image_size, mode="xyxy")
            if per_class is not None:
                boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", per_box_cls)
            boxlist.add_field("coeffs", per_coeffs)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

        return results
    
    def fast_nms(self, box_cls, bbox, coeffs):
        box_cls_max, _ = torch.max(box_cls, dim=1)
        keep = (box_cls_max > self.pre_nms_thresh)
        box_cls = box_cls[keep]
        bbox = bbox[keep]
        coeffs = coeffs[keep]

        box_cls = box_cls.permute(1, 0)

        box_cls, idx = box_cls.sort(1, descending=True)

        idx = idx[:, :self.pre_nms_top_n].contiguous()
        box_cls = box_cls[:, :self.pre_nms_top_n]
    
        num_classes, num_dets = idx.size()

        bbox = bbox[idx.view(-1)].view(num_classes, num_dets, 4)
        coeffs = coeffs[idx.view(-1)].view(num_classes, num_dets, -1)

        iou = jaccard(bbox, bbox)
        iou.triu_(diagonal=1)
        iou_max, _ = iou.max(dim=1)

        # Now just filter out the ones higher than the threshold
        keep = (iou_max <= self.nms_thresh)
        
        # Assign each kept detection to its corresponding class
        labels = torch.arange(num_classes, device=bbox.device)[:, None].expand_as(keep)
        labels = labels[keep]

        bbox = bbox[keep]
        coeffs = coeffs[keep]
        box_cls = box_cls[keep]
        
        # Only keep the top self.fpn_post_nms_top_n highest scores across all classes
        box_cls, idx = box_cls.sort(0, descending=True)
        idx = idx[:self.fpn_post_nms_top_n]
        box_cls = box_cls[:self.fpn_post_nms_top_n]

        labels = labels[idx]
        labels += 1
        bbox = bbox[idx]
        coeffs = coeffs[idx]

        return box_cls, bbox, coeffs, labels

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            scores = boxlists[i].get_field("scores")
            coeffs = boxlists[i].get_field("coeffs")
            boxes = boxlists[i].bbox
            boxlist = boxlists[i]
            if cfg.MODEL.YOLACT.USE_FAST_NMS:
                scores, boxes, coeffs, labels = self.fast_nms(scores, boxes, coeffs)
                result = BoxList(boxes, boxlist.size, mode="xyxy")
                result.add_field("scores", scores)
                result.add_field("coeffs", coeffs)
                result.add_field("labels", labels)
            else:
                labels = boxlists[i].get_field("labels")
                result = []
                # skip the background
                for j in range(1, self.num_classes):
                    inds = (labels == j).nonzero().squeeze(1)

                    # if inds.numel() == 0:
                    #     continue

                    scores_j = scores[inds]
                    coeffs_j = coeffs[inds, :]
                    boxes_j = boxes[inds, :]
                    boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
                    boxlist_for_class.add_field("scores", scores_j)
                    boxlist_for_class.add_field("coeffs", coeffs_j)
                    # per class nms
                    boxlist_for_class = boxlist_nms(
                        boxlist_for_class, self.nms_thresh,
                        score_field="scores"
                    )
                    num_labels = len(boxlist_for_class)
                    boxlist_for_class.add_field(
                        "labels", torch.full((num_labels,), j,
                                            dtype=torch.int64,
                                            device=scores.device)
                    )
                    result.append(boxlist_for_class)
                result = cat_boxlist(result)
            
                # Limit to max_per_image detections **over all classes**
                number_of_detections = len(result)
                if number_of_detections > self.fpn_post_nms_top_n > 0:
                    cls_scores = result.get_field("scores")
                    image_thresh, _ = torch.kthvalue(
                        cls_scores.cpu(),
                        number_of_detections - self.fpn_post_nms_top_n + 1
                    )
                    keep = cls_scores >= image_thresh.item()
                    keep = torch.nonzero(keep).squeeze(1)
                    result = result[keep]

            results.append(result)
        return results

    def forward(self, anchors, box_cls, box_regression, coeffs, prototypes):
        sampled_boxes = []
        num_levels = len(box_cls)
        anchors = list(zip(*anchors))
        
        for a, c, r, co in zip(anchors, box_cls, box_regression, coeffs):
            sampled_boxes.append(self.forward_for_single_feature_map(a, c, r, co))

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]

        if num_levels > 1:
            boxlists = self.select_over_all_levels(boxlists)

        results = []
        for prototypes_per_image, boxlists_per_image in zip(prototypes, boxlists):

            coeffs_per_image = boxlists_per_image.get_field("coeffs")

            # if DEBUG:
            #     print('range of prototypes_per_image:',\
            #         prototypes_per_image.min(), prototypes_per_image.max())

            # assemble mask
            masks_pred_per_image = prototypes_per_image.permute(1, 2, 0) @ coeffs_per_image.t()
            masks_pred_per_image = masks_pred_per_image.permute(2, 0, 1)
            masks_pred_per_image = self.mask_activation(masks_pred_per_image)

            # crop
            mask_h, mask_w = masks_pred_per_image.shape[1:]
            resized_pred_bbox = boxlists_per_image.resize((mask_w, mask_h))
            masks_pred_per_image = crop_zero_out(masks_pred_per_image, resized_pred_bbox.bbox)

            # binarize
            masks_pred_per_image = masks_pred_per_image > self.mask_threshold
            
            # convert mask predictions to polygon format to save memory
            if cfg.MODEL.YOLACT.CONVERT_MASK_TO_POLY:
                cpu_device = torch.device("cpu")
                masks_pred_per_image = SegmentationMask(masks_pred_per_image.to(cpu_device), \
                    (mask_w, mask_h), "mask")
                if DEBUG:
                    print(len(masks_pred_per_image), mask_w, mask_h)
                masks_pred_per_image = masks_pred_per_image.convert("poly")
            else:
                masks_pred_per_image = SegmentationMask(masks_pred_per_image, (mask_w, mask_h), "mask")
            
            if DEBUG:
                print(len(masks_pred_per_image), mask_w, mask_h)

            # resize
            img_w, img_h = boxlists_per_image.size
            masks_pred_per_image = masks_pred_per_image.resize((img_w, img_h))

            boxlists_per_image.add_field("masks", masks_pred_per_image)
            results.append(boxlists_per_image)

        return results

def make_yolact_postprocessor(config, box_coder):
    pre_nms_thresh = config.MODEL.RETINANET.INFERENCE_TH
    pre_nms_top_n = config.MODEL.YOLACT.PRE_NMS_TOP_N
    nms_thresh = config.MODEL.RETINANET.NMS_TH
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG
    min_size = 0

    postprocessor = YolactPostProcessor(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        min_size=min_size,
        num_classes=config.MODEL.RETINANET.NUM_CLASSES,
        mask_activation=_ACTIVATION_FUNC[config.MODEL.YOLACT.MASK_ACTIVATION],
        mask_threshold=config.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS_THRESHOLD,
        box_coder=box_coder,
    )

    return postprocessor
