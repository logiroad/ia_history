model = dict(
    type='ImageClassifier',
    backbone=dict(type='EfficientNet', arch='b4'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiLabelLinearClsHead', num_classes=12, in_channels=1792))
dataset_type = 'CustomMulti'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=380,
        efficientnet_style=True,
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='CenterCrop',
        crop_size=380,
        efficientnet_style=True,
        interpolation='bicubic'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=12,
    workers_per_gpu=2,
    train=dict(
        type='CustomMulti',
        data_prefix='/home/finn/DATASET/CRACKS/Logiroad_10746_images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='RandomResizedCrop',
                size=380,
                efficientnet_style=True,
                interpolation='bicubic'),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ],
        ann_file='data/cracks/from_detection/cracks_train.json',
        classes=('Arrachement_pelade', 'Faiencage', 'Nid_de_poule',
                 'Transversale', 'Longitudinale', 'Pontage_de_fissures',
                 'Remblaiement_de_tranchees', 'Raccord_de_chaussee',
                 'Comblage_de_trou_ou_Projection_d_enrobe', 'Bouche_a_clef',
                 'Grille_avaloir', 'Regard_tampon')),
    val=dict(
        type='CustomMulti',
        data_prefix='/home/finn/DATASET/CRACKS/Logiroad_10746_images/',
        ann_file='data/cracks/from_detection/cracks_val_test.json',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='CenterCrop',
                crop_size=380,
                efficientnet_style=True,
                interpolation='bicubic'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        classes=('Arrachement_pelade', 'Faiencage', 'Nid_de_poule',
                 'Transversale', 'Longitudinale', 'Pontage_de_fissures',
                 'Remblaiement_de_tranchees', 'Raccord_de_chaussee',
                 'Comblage_de_trou_ou_Projection_d_enrobe', 'Bouche_a_clef',
                 'Grille_avaloir', 'Regard_tampon')),
    test=dict(
        type='CustomMulti',
        data_prefix=
        '/home/theo/workdir/mmseg/Logiroad_10746_images_road_filter/',
        ann_file='data/cracks/from_detection/cracks_val_test.json',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='CenterCrop',
                crop_size=380,
                efficientnet_style=True,
                interpolation='bicubic'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        classes=('Arrachement_pelade', 'Faiencage', 'Nid_de_poule',
                 'Transversale', 'Longitudinale', 'Pontage_de_fissures',
                 'Remblaiement_de_tranchees', 'Raccord_de_chaussee',
                 'Comblage_de_trou_ou_Projection_d_enrobe', 'Bouche_a_clef',
                 'Grille_avaloir', 'Regard_tampon')))
evaluation = dict(interval=2, metric='mAP')
optimizer = dict(
    type='SGD', lr=0.004687500000000001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[7, 15, 22])
runner = dict(type='EpochBasedRunner', max_epochs=25)
checkpoint_config = dict(interval=2)
log_config = dict(
    interval=250,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'checkpoints/efficientnet-b4_3rdparty_8xb32-aa_in1k_20220119-45b8bd2b.pth'
resume_from = None
workflow = [('train', 1)]
classes = ('Arrachement_pelade', 'Faiencage', 'Nid_de_poule', 'Transversale',
           'Longitudinale', 'Pontage_de_fissures', 'Remblaiement_de_tranchees',
           'Raccord_de_chaussee', 'Comblage_de_trou_ou_Projection_d_enrobe',
           'Bouche_a_clef', 'Grille_avaloir', 'Regard_tampon')
work_dir = 'output/cracks_efficientnet/'
gpu_ids = [0]
