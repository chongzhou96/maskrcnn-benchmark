# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Utility functions minipulating the prediction layers
"""

from ..utils import cat

import torch

def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


def concat_box_prediction_layers(box_cls, box_regression):
    box_cls_flattened = []
    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(
        box_cls, box_regression
    ):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(
            box_cls_per_level, N, A, C, H, W
        )
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(
            box_regression_per_level, N, A, 4, H, W
        )
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).reshape(-1, C)
    box_regression = cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression

def concat_coeffs_prediction_layers(coeffs, num_prototypes):
    coeffs_flattened = []
    for coeffs_per_level in coeffs:
        N, AxC, H, W = coeffs_per_level.shape
        C = num_prototypes
        A = AxC // C
        coeffs_per_level = permute_and_flatten(coeffs_per_level, N, A, C, H, W)
        coeffs_flattened.append(coeffs_per_level)
    coeffs = cat(coeffs_flattened, dim=1)
    # coeffs = [coeffs_per_image for coeffs_per_image in coeffs]
    return coeffs