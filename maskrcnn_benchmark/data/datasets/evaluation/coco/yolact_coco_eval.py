import torch
import logging
from tqdm import tqdm
import numpy as np
from collections import OrderedDict

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.engine.bbox_aug import im_detect_bbox_aug
from maskrcnn_benchmark.layers.misc import interpolate
# from maskrcnn_benchmark.engine.inference import _accumulate_predictions_from_multiple_gpus
# from maskrcnn_benchmark.utils.comm import is_main_process

iou_thresholds = [x / 100 for x in range(50, 100, 5)]
logger = logging.getLogger("maskrcnn_benchmark.inference")

def do_yolact_coco_evaluation(
    model,
    data_loader,
    device,
):
    model.eval()
    # cpu_device = torch.device("cpu")
    dataset = data_loader.dataset

    logger.info("Preparing results for COCO format")
    num_categories = len(dataset.categories)
    ap_data = {
        'box' : [[APDataObject() for _ in range(num_categories)] for _ in iou_thresholds],
        'mask': [[APDataObject() for _ in range(num_categories)] for _ in iou_thresholds]
    }

    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        with torch.no_grad():
            if cfg.TEST.BBOX_AUG.ENABLED:
                predictions = im_detect_bbox_aug(model, images, device)
            else:
                predictions = model(images.to(device))
            

        for image_id, prediction, target in zip(image_ids, predictions, targets):
            # original_id = dataset.id_to_img_map[image_id]
            img_info = dataset.get_img_info(image_id)
            w = img_info["width"]
            h = img_info["height"]
            
            prediction = prediction.resize((w, h))
            target = target.to(device)
            target = target.resize((w, h))

            # boxes = prediction.bbox
            scores = prediction.get_field("scores")
            classes = prediction.get_field("labels")
            masks = prediction.get_field("masks")
            masks = interpolate(
                input=masks.float(),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).type_as(masks)
            masks = masks.view(-1, h*w)
            
            gt_classes = target.get_field("labels")
            gt_masks = target.get_field("masks")\
                .get_mask_tensor(do_squeeze=False)
            gt_masks = interpolate(
                input=gt_masks[:, None].float(),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).type_as(gt_masks)
            gt_masks = gt_masks.view(-1, h*w)
            
            num_pred = len(classes)
            num_gt   = len(gt_classes)
            
            mask_iou_cache = mask_iou(masks, gt_masks)
            bbox_iou_cache = boxlist_iou(prediction, target)

            iou_types = [
                ('box',  lambda i,j: bbox_iou_cache[i, j].item()),
                ('mask', lambda i,j: mask_iou_cache[i, j].item())
            ]

            for _class in set(classes + gt_classes):
                num_gt_for_class = sum([1 for x in gt_classes if x == _class])
                
                for iouIdx in range(len(iou_thresholds)):
                    iou_threshold = iou_thresholds[iouIdx]

                    for iou_type, iou_func in iou_types:
                        gt_used = [False] * len(gt_classes)
                        
                        ap_obj = ap_data[iou_type][iouIdx][_class]
                        ap_obj.add_gt_positives(num_gt_for_class)

                        for i in range(num_pred):
                            if classes[i] != _class:
                                continue
                            
                            max_iou_found = iou_threshold
                            max_match_idx = -1
                            for j in range(num_gt):
                                if gt_used[j] or gt_classes[j] != _class:
                                    continue
                                    
                                iou = iou_func(i, j)

                                if iou > max_iou_found:
                                    max_iou_found = iou
                                    max_match_idx = j
                            
                            if max_match_idx >= 0:
                                gt_used[max_match_idx] = True
                                ap_obj.push(scores[i], True)

    calc_map(ap_data)

class APDataObject:
    """
    Stores all the information necessary to calculate the AP for one IoU and one class.
    """

    def __init__(self):
        self.data_points = []
        self.num_gt_positives = 0

    def push(self, score:float, is_true:bool):
        self.data_points.append((score, is_true))
    
    def add_gt_positives(self, num_positives:int):
        """ Call this once per image. """
        self.num_gt_positives += num_positives

    def is_empty(self) -> bool:
        return len(self.data_points) == 0 and self.num_gt_positives == 0

    def get_ap(self) -> float:
        """ Warning: result not cached. """

        if self.num_gt_positives == 0:
            return 0

        # Sort descending by score
        self.data_points.sort(key=lambda x: -x[0])

        precisions = []
        recalls    = []
        num_true  = 0
        num_false = 0

        # Compute the precision-recall curve. The x axis is recalls and the y axis precisions.
        for datum in self.data_points:
            # datum[1] is whether the detection a true or false positive
            if datum[1]: num_true += 1
            else: num_false += 1
            
            precision = num_true / (num_true + num_false)
            recall    = num_true / self.num_gt_positives

            precisions.append(precision)
            recalls.append(recall)

        # Smooth the curve by computing [max(precisions[i:]) for i in range(len(precisions))]
        # Basically, remove any temporary dips from the curve.
        # At least that's what I think, idk. COCOEval did it so I do too.
        for i in range(len(precisions)-1, 0, -1):
            if precisions[i] > precisions[i-1]:
                precisions[i-1] = precisions[i]

        # Compute the integral of precision(recall) d_recall from recall=0->1 using fixed-length riemann summation with 101 bars.
        y_range = [0] * 101 # idx 0 is recall == 0.0 and idx 100 is recall == 1.00
        x_range = np.array([x / 100 for x in range(101)])
        recalls = np.array(recalls)

        # I realize this is weird, but all it does is find the nearest precision(x) for a given x in x_range.
        # Basically, if the closest recall we have to 0.01 is 0.009 this sets precision(0.01) = precision(0.009).
        # I approximate the integral this way, because that's how COCOEval does it.
        indices = np.searchsorted(recalls, x_range, side='left')
        for bar_idx, precision_idx in enumerate(indices):
            if precision_idx < len(precisions):
                y_range[bar_idx] = precisions[precision_idx]

        # Finally compute the riemann sum to get our integral.
        # avg([precision(x) for x in 0:0.01:1])
        return sum(y_range) / len(y_range)

def calc_map(ap_data):
    logger.info('Calculating mAP...')
    aps = [{'box': [], 'mask': []} for _ in iou_thresholds]

    # TODO replace 80 with variable
    for _class in range(80):
        for iou_idx in range(len(iou_thresholds)):
            for iou_type in ('box', 'mask'):
                ap_obj = ap_data[iou_type][iou_idx][_class]

                if not ap_obj.is_empty():
                    aps[iou_idx][iou_type].append(ap_obj.get_ap())

    all_maps = {'box': OrderedDict(), 'mask': OrderedDict()}

    # Looking back at it, this code is really hard to read :/
    for iou_type in ('box', 'mask'):
        all_maps[iou_type]['all'] = 0 # Make this first in the ordereddict
        for i, threshold in enumerate(iou_thresholds):
            mAP = sum(aps[i][iou_type]) / len(aps[i][iou_type]) * 100 if len(aps[i][iou_type]) > 0 else 0
            all_maps[iou_type][int(threshold*100)] = mAP
        all_maps[iou_type]['all'] = (sum(all_maps[iou_type].values()) / (len(all_maps[iou_type].values())-1))
    
    print_maps(all_maps)
    return all_maps

def print_maps(all_maps):
    # Warning: hacky 
    make_row = lambda vals: (' %5s |' * len(vals)) % tuple(vals)
    make_sep = lambda n:  ('-------+' * n)

    logger.info()
    logger.info(make_row([''] + [('.%d ' % x if isinstance(x, int) else x + ' ') for x in all_maps['box'].keys()]))
    logger.info(make_sep(len(all_maps['box']) + 1))
    for iou_type in ('box', 'mask'):
        logger.info(make_row([iou_type] + ['%.2f' % x for x in all_maps[iou_type].values()]))
    logger.info(make_sep(len(all_maps['box']) + 1))
    logger.info()

def mask_iou(mask1, mask2):
    """
    Inputs inputs are matricies of size _ x N. Output is size _1 x _2.
    Note: if iscrowd is True, then mask2 should be the crowd.
    """

    intersection = torch.matmul(mask1, mask2.t())
    area1 = torch.sum(mask1, dim=1).view(1, -1)
    area2 = torch.sum(mask2, dim=1).view(1, -1)
    union = (area1.t() + area2) - intersection

    ret = intersection / union
    return ret.cpu()