# defaults
# config='configs/yolact/yolact_R-101-FPN_1x.yaml'
config='configs/yolact/yolact_retinanet_R-101-FPN_1x.yaml'
# config='configs/e2e_mask_rcnn_R_101_FPN_1x.yaml'
# config='configs/retinanet/retinanet_R-101-FPN_1x.yaml'

nproc=2

if [ "$#" -ge 1 ]; then
    nproc=$1
fi

if [ "$#" -ge 2 ]; then
    config=$2
fi

python -m torch.distributed.launch --nproc_per_node=$nproc tools/train_net.py --config-file $config