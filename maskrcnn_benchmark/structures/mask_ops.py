import torch
import torch.nn.functional as F
import pycocotools.mask as mask_util
import numpy as np

def crop_zero_out(masks, boxes):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.

    Args:
        - masks should be a size [n, h, w] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in xyxy form
    """
    n, h, w = masks.size()
    x1, x2 = boxes[:, 0], boxes[:, 2]
    y1, y2 = boxes[:, 1], boxes[:, 3]

    rows = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, 1, -1).expand(n, h, w)
    cols = torch.arange(h, device=masks.device, dtype=x1.dtype).view(1, -1, 1).expand(n, h, w)
    
    masks_left  = rows >= x1.view(-1, 1, 1)
    masks_right = rows <  x2.view(-1, 1, 1)
    masks_up    = cols >= y1.view(-1, 1, 1)
    masks_down  = cols <  y2.view(-1, 1, 1)
    
    crop_mask = masks_left * masks_right * masks_up * masks_down
    
    return masks * crop_mask.float()

def convert_binary_to_rle(masks):
    """
    Convert binary mask to rle format

    Args:
        - masks should be a list whose elements are cpu tensors with the size of [1, img_h, img_w]
    """
    rles = [
        mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
        for mask in masks
    ]
    for rle in rles:
        rle["counts"] = rle["counts"].decode("utf-8")
    return rles

def convert_rle_to_binary(rles):
    """
    Convert a list of rle masks to a list of binary masks
    """
    masks = []
    for rle in rles:
        mask = mask_util.decode(rle)
        mask = torch.from_numpy(mask)
        masks.append(mask)
    return masks

def resize_rle(rles, size):
    """
    Resize a list of rle masks

    Args:
        - size should be a tuple of (new_height, new_width)
    """
    masks = convert_rle_to_binary(rles)
    masks = torch.stack(masks)

    masks = F.interpolate(
        input=masks[None].float(),
        size=size,
        mode="bilinear",
        align_corners=False,
    )[0].type_as(masks)

    masks = [mask[None] for mask in masks]
    rles = convert_binary_to_rle(masks)
    return rles