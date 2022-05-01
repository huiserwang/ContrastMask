_base_ = [
    '../mmdetection-2.14.0/configs/_base_/default_runtime.py',
    '../mmdetection-2.14.0/configs/_base_/datasets/coco_instance.py',
    '../mmdetection-2.14.0/configs/_base_/schedules/schedule_1x.py',
    '../mmdetection-2.14.0/configs/_base_/models/mask_rcnn_r50_fpn.py'
]
#dataset settings
dataset_type = 'CocoPSDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data_root = '/home/huiser/ssd/Datasets/COCO/'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadPSAnnotations', with_bbox=True, with_mask=True),
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
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data=dict(
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline,
        base_set='nonvoc',
        novel_set='voc'),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline,
        base_set='nonvoc',
        novel_set='voc'),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline,
        base_set='nonvoc',
        novel_set='voc'),
)

# Models
model = dict(
    type='ContrastMask',
    roi_head=dict(
        type='StandardRoIHeadPS',
        bbox_head=dict(
            type='Shared2FCBBoxWithCAMHead',
            with_avg_pool=True,
        ),
        mask_head=dict(
            type="FCNMaskCamHead",
            in_channels=256,
            class_agnostic=True,
            contrastive_enable=True,
            contrastive_head=dict(
                type="ContrastiveHead",
                num_convs=8,
                num_projectfc=3,
                thred_u=0.1,
                scale_u=1.0,
                percent=0.3,
                fc_norm_cfg=dict(type='BN',)),
        ),
        contrastive_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=28, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
    )
)

# Default_runtime
checkpoint_config = dict(out_dir="./ckpt")
custom_imports = dict(
    imports=[
        'ilib.dataset.cocops', 'ilib.dataset.pipelines', 
        'ilib.utils.tensorboard_utils', 'ilib.utils.sample_utils',
        'ilib.model.contrastmask',
        'ilib.model.contrastmask_head.roi_head', 
        'ilib.model.contrastmask_head.mask_head',
        'ilib.model.contrastmask_head.bbox_head', 
        'ilib.model.contrastmask_head.contrastive_head'],
    allow_failed_imports=False
)

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='SelfTensorboardLoggerHook', 
             log_dir="./tensorboards", 
             contr_upper_ep=4.0, 
             contr_start_ep=0.0, 
             init_value=0.25)
    ]
)

# Schedule 3x, same as the default setting of mmdetection
lr_config = dict(
    step=[8, 11]
)
runner = dict(type='EpochBasedRunner', max_epochs=12)

