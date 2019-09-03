# defaults
config='configs/yolact/yolact_R-101-FPN_1x.yaml'
# config='configs/yolact/yolact_retinanet_R-101-FPN_1x.yaml'
# config='configs/e2e_mask_rcnn_R_101_FPN_1x.yaml'
# config='configs/retinanet/retinanet_R-101-FPN_1x.yaml'

nproc=1
batch=1
subset_size=-1

if [ "$#" -ge 1 ]; then
    subset_size=$1
fi

if [ "$#" -ge 2 ]; then
    nproc=$2
fi

if [ "$#" -ge 3 ]; then
    batch=$3
fi

if [ "$#" -ge 4 ]; then
    config=$4
fi

python -m torch.distributed.launch --nproc_per_node=$nproc tools/test_net.py \
    --config-file $config TEST.IMS_PER_BATCH $batch DATALOADER.SUBSET_SIZE $subset_size
