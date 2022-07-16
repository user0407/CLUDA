# Obtained from: https://github.com/lhoyer/DAFormer
# This is the same as SegFormer but with 256 embed_dims
# SegF. with C_e=256 in Tab. 7

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b5.pth',
    backbone=dict(type='mit_b5', style='pytorch'),
    decode_head=dict(
        type='DAFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(
            embed_dims=512,
            embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            fusion_cfg=dict(
                type='conv',
                kernel_size=1,
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg),
        ),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),

    # pretrained='open-mmlab://resnet101_v1c',
    # backbone=dict(
    #     type='ResNetV1c',
    #     depth=101,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     dilations=(1, 1, 2, 4),
    #     strides=(1, 2, 1, 1),
    #     norm_cfg=norm_cfg,
    #     norm_eval=False,
    #     style='pytorch',
    #     contract_dilation=True),
    # decode_head=dict(
    #     type='DAFormerHead',
    #     in_channels=[256, 512, 1024, 2048],
    #     in_index=[0, 1, 2, 3],
    #     channels=256,
    #     dropout_ratio=0.1,
    #     num_classes=19,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     decoder_params=dict(
    #         embed_dims=512,
    #         embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
    #         embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
    #         fusion_cfg=dict(
    #             type='conv',
    #             kernel_size=1,
    #             act_cfg=dict(type='ReLU'),
    #             norm_cfg=norm_cfg),
    #     ),
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
