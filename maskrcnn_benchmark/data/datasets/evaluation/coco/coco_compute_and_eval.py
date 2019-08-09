import torch
import logging
import tempfile
import os
from tqdm import tqdm
import numpy as np

from .coco_eval import evaluate_predictions_on_coco, COCOResults

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.bbox_aug import im_detect_bbox_aug
from maskrcnn_benchmark.structures.mask_ops import convert_binary_to_rle
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.utils.comm import is_main_process

def do_coco_compute_and_evalute(
    model,
    data_loader,
    device,
    output_folder,
):
    model.eval()
    dataset = data_loader.dataset
    masker = Masker(threshold=0.5, padding=1)
    cpu_device = torch.device("cpu")
    logger = logging.getLogger("maskrcnn_benchmark.inference")

    logger.info("Preparing results for COCO format")

    coco_results = {"bbox":[], "segm":[]}
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        with torch.no_grad():
            if cfg.TEST.BBOX_AUG.ENABLED:
                predictions = im_detect_bbox_aug(model, images, device)
            else:
                predictions = model(images.to(device))
        
        predictions = [p.to(cpu_device) for p in predictions]

        for image_id, prediction, target in zip(image_ids, predictions, targets):
            original_id = dataset.id_to_img_map[image_id]
            img_info = dataset.get_img_info(image_id)
            w = img_info["width"]
            h = img_info["height"]
            
            prediction = prediction.resize((w, h))
            prediction = prediction.convert("xywh")

            boxes = prediction.bbox.tolist()
            scores = prediction.get_field("scores").tolist()
            classes = prediction.get_field("labels").tolist()
            masks = prediction.get_field("masks")
            if isinstance(masks, SegmentationMask):
                masks = masks.get_mask_tensor(do_squeeze=False)[:, None]

            # Masker is necessary only if masks haven't been already resized.
            if list(masks.shape[-2:]) != [h, w]:
                masks = masker(masks.expand(1, -1, -1, -1, -1), prediction)
                masks = masks[0]

            mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in classes]
            rles = convert_binary_to_rle(masks.cpu())

            coco_results["bbox"].extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": mapped_labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )

            coco_results["segm"].extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": mapped_labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )

    iou_types = ["bbox", "segm"]
    results = COCOResults(*iou_types)
    logger.info("Evaluating predictions")
    for iou_type in iou_types:
        with tempfile.NamedTemporaryFile() as f:
            file_path = f.name
            if output_folder:
                file_path = os.path.join(output_folder, iou_type + ".json")
            res = evaluate_predictions_on_coco(
                dataset.coco, coco_results[iou_type], file_path, dataset.ids, iou_type
            )
            results.update(res)
    logger.info(results)