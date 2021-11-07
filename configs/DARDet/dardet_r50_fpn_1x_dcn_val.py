# model settings
model = dict(
    type='DARDet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        # dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        # stage_with_dcn=(False, True, True, True),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        extra_convs_on_inputs=False,  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='DARDetHead',
        num_classes=15,
        in_channels=256,
        stacked_convs=3,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=False,
        dcn_on_last_conv=False,#False  True
        use_atss=True,
        use_vfl=True,
        loss_cls=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0),
            loss_rbox=dict( type='RotatedIoULoss', loss_weight=1.5),
            loss_rbox_refine=dict( type='RotatedIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        rotate_test=True,
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.1),
        max_per_img=1500))

# data setting
dataset_type = 'DotaKDataset'
data_root = '/media/zf/E/Dataset/dota_1024_s2anet2/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True,poly2mask=False),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    # dict(type='Rotate', level=5, prob=0.5),
    dict(type='RandomFlip', direction=['horizontal','vertical', 'diagonal'], flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=6e-2,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/reppoint_keypoints_train2017.json',
            img_prefix=data_root + 'train2017/',
            pipeline=train_pipeline)),
    # train=dict(
    #     type=dataset_type,
    #     ann_file=data_root + 'annotations/reppoint_keypoints_train2017.json',
    #     img_prefix=data_root + 'train2017/',
    #     pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/reppoint_keypoints_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/reppoint_keypoints_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
work_dir = '/media/zf/E/Dataset/dota_1024_s2anet2/workdir/DARDet_r50_2080base'
load_from =None#'/media/zf/E/mmdetection213_1080/checkpoint/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-6879c318.pth'
resume_from =None# '/media/zf/E/Dataset/dota1-split-1024/workdir/vfnet_r50_50_ms/epoch_7.pth'

evaluation = dict(interval=3, metric='bbox',eval_dir= work_dir,
        gt_dir='/media/zf/E/Dataset/dota_1024_s2anet2/valGTtxt/')
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
