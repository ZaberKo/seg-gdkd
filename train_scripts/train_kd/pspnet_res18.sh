python -m torch.distributed.launch --nproc_per_node=8 \
    train_kd.py \
    --teacher-model deeplabv3 \
    --student-model psp \
    --teacher-backbone resnet101 \
    --student-backbone resnet18 \
    --work-dir work_dirs/dist_dv3-r101_psp_r18 \
    --teacher-pretrained ckpts/deeplabv3_resnet101_citys_best_model.pth \
    --student-pretrained-base ckpts/resnet18-imagenet.pth

torchrun --standalone --nnodes=1 --nproc-per-node=8 \
    --teacher-model deeplabv3 \
    --student-model psp \
    --teacher-backbone resnet101 \
    --student-backbone resnet18 \
    --work-dir work_dirs/dist_dv3-r101_psp_r18 \
    --teacher-pretrained ckpts/deeplabv3_resnet101_citys_best_model.pth \
    --student-pretrained-base ckpts/resnet18-imagenet.pth

# single GPU test
python train_kd.py \
    --teacher-model deeplabv3 \
    --student-model psp \
    --teacher-backbone resnet101 \
    --student-backbone resnet18 \
    --work-dir work_dirs/dist_dv3-r101_psp_r18 \
    --batch-size 4 \
    --teacher-pretrained ckpts/deeplabv3_resnet101_citys_best_model.pth \
    --student-pretrained-base ckpts/resnet18-imagenet.pth