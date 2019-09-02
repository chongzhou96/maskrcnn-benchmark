from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import cv2
from matplotlib.pyplot import imsave
import glob
import os
import numpy as np
import maskrcnn_benchmark.utils.futils as fu
import torch
import random
from tqdm import tqdm

# params
args = fu.Args()
args.dataset = 'COCO' 
args.config_file = "configs/yolact/yolact_retinanet_R-101-FPN_1x.yaml"
# args.config_file = "configs/e2e_mask_rcnn_R_101_FPN_1x.yaml"
args.data_dir = 'datasets/coco/images'
args.vis_dir = 'weights/yolact_retinanet/inference/coco_2017_val/output'
args.postfix = 'jpg'
args.rand_seed = 777
args.conf_thresh = 0.5
args.img_num = 10
args.opts = []
with open('weights/yolact_retinanet/last_checkpoint', 'r') as f:
    last_checkpoint = f.read()
args.opts += ['MODEL.WEIGHT', last_checkpoint]
args.opts += ['MODEL.YOLACT.CONVERT_MASK_TO_POLY', False]

# update the config options with the config file
cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

demo = COCODemo(
    cfg,
    confidence_threshold=args.conf_thresh,
)

# set random seed
np.random.seed(args.rand_seed)
random.seed(args.rand_seed)
torch.random.manual_seed(args.rand_seed)


# # load image and then run prediction
# image = cv2.imread('PATH_TO_IMAGE') # imread returns BGR output
# predictions = demo.run_on_opencv_image(image)
# predictions = predictions[:, :, ::-1]
# imsave('res.png', predictions) # imsave is assuming RGB input


# # detect in a directory
# img_list = glob.glob(os.path.join(args.data_dir, '*.%s' % args.postfix))
# for img_path in img_list:
#     image = cv2.imread(img_path) # imread returns BGR output
#     predictions = demo.run_on_opencv_image(image)
#     predictions = predictions[:, :, ::-1]
#     save_path = os.path.join(args.vis_dir, os.path.basename(img_path).rstrip('.%s' % args.postfix) + '.png')
#     if not os.path.isdir(os.path.dirname(save_path)):
#         os.makedirs(os.path.dirname(save_path))
#     # imsave(save_path, predictions) # imsave is assuming RGB input
#     cv2.imwrite(save_path, predictions[:, :, ::-1])


# loop inside a dataset 
img_list = glob.glob(args.data_dir + '/*.%s' % args.postfix)
rand_idx = np.arange(len(img_list))
np.random.shuffle(rand_idx)
img_list = np.array(img_list)[rand_idx[0:args.img_num]].tolist()
im_list, im_cap = fu.initHTML(len(img_list), 1)
for img_cursor, img_path in tqdm(enumerate(img_list), total=len(img_list)):
    img = cv2.imread(img_path) # imread returns BGR output
    predictions = demo.run_on_opencv_image(img)
    img_name = os.path.basename(img_path).rstrip('.%s' % args.postfix) + '.png'
    save_path = os.path.join(args.vis_dir, 'imgs', img_name)
    if not os.path.isdir(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    cv2.imwrite(save_path, predictions)
    im_list[img_cursor][0] = os.path.relpath(save_path, args.vis_dir)
    im_cap[img_cursor][0] = img_name
      
html_path = os.path.join(args.vis_dir, 'vis.html')
fu.writeHTML(html_path, im_list, im_cap)
print('Done.')

