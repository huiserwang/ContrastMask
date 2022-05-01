_base_ = [
    './contrastmask_res50_fpn_coco_1x_nonvoc2voc.py'
]

# Models
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')),
)