_base_ = [
    './contrastmask_res50_fpn_coco_1x_nonvoc2voc.py'
]
#dataset settings
dataset_type = 'CocoPSDataset'
data_root = '/home/huiser/ssd/Datasets/COCO/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadPSAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundlePS', collect_keys=['proposals', 'gt_bboxes', 'gt_bboxes_ids', 
                                                    'gt_bboxes_ignore', 'gt_labels', 'gt_is_novel']),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_bboxes_ids', 
                               'gt_labels', 'gt_masks', 'gt_is_novel'],
                         meta_keys=('filename', 'ori_filename', 'ori_shape',
                                    'img_shape', 'pad_shape', 'scale_factor', 'flip',
                                    'flip_direction', 'img_norm_cfg', 'ann_info')),
]
data=dict(
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_train2017.json',
            img_prefix=data_root + 'train2017/',
            pipeline=train_pipeline,
            base_set='nonvoc',
            novel_set='voc')),
    val=dict(
        base_set='nonvoc',
        novel_set='voc',
    ),
    test=dict(
        base_set='nonvoc',
        novel_set='voc',
    ),
)

# Schedule 3x, same as the default setting of mmdetection
lr_config = dict(
    step=[9, 11]
)

