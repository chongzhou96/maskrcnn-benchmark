import torch
from torch.nn import functional as F

from ..utils import concat_box_prediction_layers, concat_coeffs_prediction_layers
from .yolact import _ACTIVATION_FUNC


from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.layers import SigmoidFocalLoss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.modeling.rpn.retinanet.loss import RetinaNetLossComputation, generate_retinanet_labels
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.mask_ops import crop_zero_out

DEBUG = False
if DEBUG:
    import matplotlib.pyplot as plt
    from maskrcnn_benchmark.utils.timer import Timer, get_time_str

class YolactLossComputation(RetinaNetLossComputation):
    """
    This class computes the Yolact loss.
    """
    def __init__(self, proposal_matcher, box_coder,
                 generate_labels_func,
                 sigmoid_focal_loss,
                 mask_activation,
                 num_prototypes,
                 mask_to_train,
                 bbox_reg_beta=0.11,
                 regress_norm=1.0,
                 mask_norm=1.0,
                 mask_with_logits=False):
        super(YolactLossComputation, self).__init__(
                proposal_matcher, box_coder,
                generate_labels_func,
                sigmoid_focal_loss,
                bbox_reg_beta,
                regress_norm
        )
        if mask_with_logits:
            self.mask_activation = None
        else:
            self.mask_activation = mask_activation
        self.mask_with_logits = mask_with_logits
        self.num_prototypes = num_prototypes
        # don't copy masks because it is slow
        # self.copied_fields = ['labels', 'masks']
        self.mask_norm = mask_norm
        self.mask_to_train = mask_to_train

        if DEBUG:
            self.timer = Timer()

    def prepare_targets_and_assemble(self, anchors, targets, coeffs, prototypes):
        labels = []
        regression_targets = []
        mask_targets = []
        mask_pred = []
        gt_boxes_area = []

        for anchors_per_image, targets_per_image, coeffs_per_image, prototypes_per_image \
             in zip(anchors, targets, coeffs, prototypes):
            
            all_mask_targets_per_image = targets_per_image.get_field("masks")
            matched_targets = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image, self.copied_fields
            )

            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = self.generate_labels_func(matched_targets)
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0

            # discard anchors that go out of the boundaries of the image
            if "not_visibility" in self.discard_cases:
                labels_per_image[~anchors_per_image.get_field("visibility")] = -1

            # discard indices that are between thresholds
            if "between_thresholds" in self.discard_cases:
                inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, anchors_per_image.bbox
            )

            # if DEBUG:
            #     print('size of matched_targets:', matched_targets.size)
            #     print('size of anchors_per_image:', anchors_per_image.size)

            # mask scores are only computed on positive samples
            pos_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            # mask assembly
            mask_pred_per_image = prototypes_per_image.permute(1, 2, 0) @ coeffs_per_image[pos_inds].t()
            mask_pred_per_image = mask_pred_per_image.permute(2, 0, 1)
            if self.mask_activation is not None:
                mask_pred_per_image = self.mask_activation(mask_pred_per_image)

            # if DEBUG:
            #     with torch.no_grad():
            #         print('range of prototypes_per_image:',\
            #             prototypes_per_image.min(), prototypes_per_image.max())
            #         print('range of coeffs_per_image:',\
            #             coeffs_per_image.min(), coeffs_per_image.max())
            #         print('range of mask_pred_per_image:',\
            #             mask_pred_per_image.min(), mask_pred_per_image.max())
            #         print('range of mask_pred_per_image (sigmoid):',\
            #             mask_pred_per_image.sigmoid().min(), mask_pred_per_image.sigmoid().max())

            # if DEBUG:
            #     with torch.no_grad():
            #         plt.figure(1)
            #         plt.imshow(mask_pred_per_image.to("cpu")[0])

            # use resized ground-truth boxes to crop mask_pred
            mask_h, mask_w = mask_pred_per_image.shape[1:]
            gt_boxes_per_image = matched_targets[pos_inds].convert("xyxy").resize((mask_w, mask_h))
            mask_pred_per_image = crop_zero_out(mask_pred_per_image, gt_boxes_per_image.bbox)

            # used by mask loss reweighting
            gt_boxes_area_per_image = gt_boxes_per_image.area()
            # mask targets (on CPU)
            pos_matched_idxs = matched_idxs[pos_inds]
            mask_targets_per_image = all_mask_targets_per_image[pos_matched_idxs]
            mask_targets_per_image = mask_targets_per_image\
                .resize((mask_w, mask_h)).get_mask_tensor(do_squeeze=False)

            # if DEBUG:
            #     with torch.no_grad():
            #         plt.figure(2)
            #         plt.imshow(mask_pred_per_image.sigmoid().to("cpu")[0])
            #         plt.figure(3)
            #         plt.imshow((mask_pred_per_image > 0.5).to("cpu")[0])
            #         plt.figure(4)
            #         plt.imshow(mask_targets_per_image.to("cpu")[0])
            #         plt.show()

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
            mask_targets.append(mask_targets_per_image)
            mask_pred.append(mask_pred_per_image)
            gt_boxes_area.append(gt_boxes_area_per_image)

        return labels, regression_targets, mask_targets, mask_pred, gt_boxes_area

    def __call__(self, anchors, box_cls, box_regression, coeffs, prototypes, targets):
        coeffs = concat_coeffs_prediction_layers(coeffs, self.num_prototypes)

        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        labels, regression_targets, mask_targets, mask_pred, gt_boxes_area = \
            self.prepare_targets_and_assemble(anchors, targets, coeffs, prototypes)

        N = len(labels)
        box_cls, box_regression = \
                concat_box_prediction_layers(box_cls, box_regression)

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        pos_inds = torch.nonzero(labels > 0).squeeze(1)

        mask_pred = torch.cat(mask_pred, dim=0)
        device = mask_pred.device
        mask_targets = torch.cat(mask_targets, dim=0).to(device, dtype=torch.float32)
        gt_boxes_area = torch.cat(gt_boxes_area, dim=0)

        if mask_pred.size(0) > self.mask_to_train:
            perm = torch.randperm(mask_pred.size(0))
            select = perm[:self.mask_to_train]
            mask_pred = mask_pred[select]
            mask_targets = mask_targets[select]
            gt_boxes_area = gt_boxes_area[select]

        # only positive boxes contribute to regression loss and mask loss
        retinanet_regression_loss = smooth_l1_loss(
            box_regression[pos_inds],
            regression_targets[pos_inds],
            beta=self.bbox_reg_beta,
            size_average=False,
        ) / (max(1, pos_inds.numel() * self.regress_norm))

        # if DEBUG:
        #     print('retinanet_regression_loss', box_regression[pos_inds].shape)

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if mask_targets.numel() == 0:
            yolact_mask_loss = mask_pred.sum() * 0
        else:
            if self.mask_with_logits:
                yolact_mask_loss = F.binary_cross_entropy_with_logits(mask_pred, mask_targets, reduction='none')
            else:
                yolact_mask_loss = F.binary_cross_entropy(mask_pred, mask_targets, reduction='none')
                
        # if DEBUG:
        #     print("gt_boxes_area:", gt_boxes_area)
        # if DEBUG:
        #     print('yolact_mask_loss', mask_pred.shape)

        # reweight mask loss by dividing the area of ground-truth boxes
        yolact_mask_loss = yolact_mask_loss.sum(dim=(1, 2)) / gt_boxes_area
        yolact_mask_loss = yolact_mask_loss.sum() / (max(1, pos_inds.numel() * self.mask_norm))
        
        if DEBUG:
            print('pos_inds.numel():', pos_inds.numel())
            print('gt_boxes_area.shape:', gt_boxes_area.shape)

        labels = labels.int()
        retinanet_cls_loss = self.box_cls_loss_func(
            box_cls,
            labels
        ) / (pos_inds.numel() + N)

        # if DEBUG:
        #     print('retinanet_cls_loss', box_cls.shape)

        return retinanet_cls_loss, retinanet_regression_loss, yolact_mask_loss

def make_yolact_loss_evaluator(cfg, box_coder):
    matcher = Matcher(
        cfg.MODEL.RETINANET.FG_IOU_THRESHOLD,
        cfg.MODEL.RETINANET.BG_IOU_THRESHOLD,
        allow_low_quality_matches=True,
    )
    sigmoid_focal_loss = SigmoidFocalLoss(
        cfg.MODEL.RETINANET.LOSS_GAMMA,
        cfg.MODEL.RETINANET.LOSS_ALPHA
    )

    loss_evaluator = YolactLossComputation(
        matcher,
        box_coder,
        generate_retinanet_labels,
        sigmoid_focal_loss,
        # sigmoid operation is integrated into binary_cross_entropy_with_logits
        _ACTIVATION_FUNC[cfg.MODEL.YOLACT.MASK_ACTIVATION],
        cfg.MODEL.YOLACT.NUM_PROTOTYPES,
        cfg.MODEL.YOLACT.MASK_TO_TRAIN,
        bbox_reg_beta = cfg.MODEL.RETINANET.BBOX_REG_BETA,
        regress_norm = cfg.MODEL.RETINANET.BBOX_REG_WEIGHT,
        mask_norm = cfg.MODEL.YOLACT.MASK_WEIGHT,
        mask_with_logits = cfg.MODEL.YOLACT.MASK_WITH_LOGITS
    )
    return loss_evaluator