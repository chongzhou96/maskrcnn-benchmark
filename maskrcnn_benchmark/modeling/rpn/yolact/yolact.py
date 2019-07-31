import torch
import torch.nn.functional as F
from torch import nn
import math

from maskrcnn_benchmark.modeling.rpn.retinanet.retinanet import RetinaNetHead
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.layers.interpolate import InterpolateModule
from maskrcnn_benchmark.modeling.rpn.anchor_generator import make_anchor_generator_retinanet

DEBUG = False
if DEBUG:
    from maskrcnn_benchmark.utils.timer import Timer, get_time_str

_ACTIVATION_FUNC = {
    'tanh':    torch.tanh,
    'sigmoid': torch.sigmoid,
    'softmax': lambda x: F.softmax(x, dim=-1),
    'relu':    lambda x: F.relu(x, inplace=True),
    'none':    lambda x: x,
}

from .inference import make_yolact_postprocessor
from .loss import make_yolact_loss_evaluator

class YolactHead(torch.nn.Module):
    """
    Adds a Yolact predition head with classification, regression and coefficents preditction heads
    """
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            cfg {[type]} -- [description]
            in_channels {[type]} -- [description]
        """
        super(YolactHead, self).__init__()
        num_classes = cfg.MODEL.RETINANET.NUM_CLASSES - 1
        num_prototypes = cfg.MODEL.YOLACT.NUM_PROTOTYPES
        num_anchors = len(cfg.MODEL.RETINANET.ASPECT_RATIOS) \
                        * cfg.MODEL.RETINANET.SCALES_PER_OCTAVE
        
        self.share_tower = cfg.MODEL.RETINANET.SHARE_HEAD_TOWER

        if self.share_tower:
            shared_tower = []
            for _ in range(cfg.MODEL.RETINANET.NUM_CONVS):
                shared_tower.append(
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1
                    )
                )
                shared_tower.append(nn.ReLU())
            self.add_module('shared_tower', nn.Sequential(*shared_tower))
        else:
            cls_tower = []
            bbox_tower = []
            coeff_tower = []
            for _ in range(cfg.MODEL.RETINANET.NUM_CONVS):
                for tower in [cls_tower, bbox_tower, coeff_tower]:
                    tower.append(
                        nn.Conv2d(
                            in_channels,
                            in_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1
                        )
                    )
                    tower.append(nn.ReLU())

            self.add_module('cls_tower', nn.Sequential(*cls_tower))
            self.add_module('bbox_tower', nn.Sequential(*bbox_tower))            
            self.add_module('coeff_tower', nn.Sequential(*coeff_tower))

        self.cls_logits = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels,  num_anchors * 4, kernel_size=3, stride=1,
            padding=1
        )
        self.coeff_pred = nn.Conv2d(
            in_channels, num_anchors * num_prototypes, kernel_size=3, stride=1,
            padding=1
        )
        self.coeff_activation = _ACTIVATION_FUNC[cfg.MODEL.YOLACT.COEFF_ACTIVATION]
        # Initialization
        if self.share_tower:
            modules_list = [self.cls_logits, self.bbox_pred, self.coeff_pred, \
                self.shared_tower]
        else:
            modules_list = [self.cls_logits, self.bbox_pred, self.coeff_pred, \
                self.cls_tower, self.bbox_tower, self.coeff_tower]
        for modules in modules_list:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
        
        # retinanet_bias_init
        prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        
    def forward(self, features):
        logits = []
        bbox_reg = []
        coeffs = []
        for feature_per_level in features:
            if self.share_tower:
                shared_feature = self.shared_tower(feature_per_level)
                logits.append(self.cls_logits(shared_feature))
                bbox_reg.append(self.bbox_pred(shared_feature))
                coeffs.append(self.coeff_activation(self.coeff_pred(shared_feature)))
            else:
                logits.append(self.cls_logits(self.cls_tower(feature_per_level)))
                bbox_reg.append(self.bbox_pred(self.bbox_tower(feature_per_level)))
                coeffs.append(self.coeff_activation(self.coeff_pred(self.coeff_tower(feature_per_level))))
        return logits, bbox_reg, coeffs

def make_net(net_cfg, in_channels):
    """
    A helper function to take a config setting and turn it into a network.
    Used by protonet and extrahead. Returns (network, out_channels)
    
    Arguments:
        net_cfg {[type]} -- [description]
        in_channels {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    def make_layer(layer_cfg):
        nonlocal in_channels
        
        # Possible patterns:
        # ( 256, 3, {}) -> conv
        # ( 256,-2, {}) -> deconv
        # (None,-2, {}) -> bilinear interpolate
        
        num_channels = layer_cfg[0]
        kernel_size = layer_cfg[1]

        if kernel_size > 0:
            layer = nn.Conv2d(in_channels, num_channels, kernel_size, **layer_cfg[2])
        else:
            if num_channels is None:
                layer = InterpolateModule(scale_factor=-kernel_size, mode='bilinear', align_corners=False, **layer_cfg[2])
            else:
                layer = nn.ConvTranspose2d(in_channels, num_channels, -kernel_size, **layer_cfg[2])
        
        in_channels = num_channels if num_channels is not None else in_channels

        return [layer, nn.ReLU(inplace=True)]

    # Use sum to concat together all the component layer lists
    net = sum([make_layer(eval(layer_cfg)) for layer_cfg in net_cfg], [])

    return nn.Sequential(*(net)), in_channels

class YolactProtonet(torch.nn.Module):
    """
    [summary]
    """
    def __init__(self, cfg, in_channels):
        super(YolactProtonet, self).__init__()
        num_prototypes = cfg.MODEL.YOLACT.NUM_PROTOTYPES

        self.protonet_tower, in_channels = make_net(cfg.MODEL.YOLACT.PROTONET, in_channels)

        self.protonet_pred = nn.Conv2d(
            in_channels, num_prototypes, kernel_size=3, stride=1,
            padding=1
        )

        self.prototype_activation = _ACTIVATION_FUNC[cfg.MODEL.YOLACT.PROTOTYPE_ACTIVATION]

        # Initialization
        for modules in [self.protonet_tower, self.protonet_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        prototypes = self.protonet_pred(self.protonet_tower(x))
        prototypes = self.prototype_activation(prototypes)
        return prototypes

class YolactModule(torch.nn.Module):
    """
    [summary]
    """
    def __init__(self, cfg, in_channels):
        super(YolactModule, self).__init__()

        box_coder = BoxCoder(weights=(10., 10., 5., 5.))

        self.anchor_generator = make_anchor_generator_retinanet(cfg)
        self.head = YolactHead(cfg, in_channels)
        self.postprocessor = make_yolact_postprocessor(cfg, box_coder)
        self.loss_evaluator = make_yolact_loss_evaluator(cfg, box_coder)

        self.proto_src = cfg.MODEL.YOLACT.PROTO_SRC
        self.protonet = YolactProtonet(cfg, in_channels)

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images {[type]} -- [description]
            features {[type]} -- [description]
        
        Keyword Arguments:
            targets {[type]} -- [description] (default: {None})
        
        Returns:
            [type] -- [description]
        """
        # each element in the list is per feature level, for example: [box_cls_level1, box_cls_level2, ...],
        # where the shape of box_cls_level1 is (N, A_1xC, H, W)
        box_cls, box_regression, coeffs = self.head(features)
        # each element in the list is per image, for example: [anchors_image1, anchors_image2, ...]
        # the anchors_image1 is still a list, whose elements are per level, 
        # for example: [anchors_image1_level1, anchors_image1_level2, ...]
        # where anchors_image1_level2 is a BoxList with A_2XHxW elements
        anchors = self.anchor_generator(images, features)
        if DEBUG:
            print(type(anchors), len(anchors))
            print(type(anchors[0]), len(anchors[0]))
            print(type(anchors[0][0]), anchors[0][0].bbox.shape)
        prototypes = self.protonet(features[self.proto_src])
 
        if self.training:
            return self._forward_train(anchors, box_cls, box_regression, coeffs, prototypes, targets)
        else:
            return self._forward_test(anchors, box_cls, box_regression, coeffs, prototypes)
    
    def _forward_train(self, anchors, box_cls, box_regression, coeffs, prototypes, targets):
        loss_box_cls, loss_box_reg, loss_mask = self.loss_evaluator(
            anchors, box_cls, box_regression, coeffs, prototypes, targets
        )

        losses = {
            "loss_retina_cls": loss_box_cls,
            "loss_retina_reg": loss_box_reg,
            "loss_yolact_mask": loss_mask,
        }
        return anchors, losses

    def _forward_test(self, anchors, box_cls, box_regression, coeffs, prototypes):
        detections = self.postprocessor(
            anchors, box_cls, box_regression, coeffs, prototypes
        )
        return detections, {}

def build_yolact(cfg, in_channels):
    return YolactModule(cfg, in_channels)