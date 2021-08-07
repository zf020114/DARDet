import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale
from mmcv.ops import DeformConv2d
from mmcv.runner import force_fp32
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init#ConvModule是卷积层、归一化层和激活层的捆绑，详细请参考api。
from mmdet.core import (bbox2distance, bbox_overlaps, build_anchor_generator,distance2rbox,multiclass_nms_rotated_bbox,rotated_box_to_poly,poly_to_rotated_box,
                        build_assigner, build_sampler, distance2bbox,
                        multi_apply, multiclass_nms, reduce_mean)
from ..builder import HEADS, build_loss
from .atss_head import ATSSHead
from .fcos_head import FCOSHead
from mmcv.ops import box_iou_rotated
INF = 1e8
import torch.nn.functional as F

@HEADS.register_module()
class VFNetHead(ATSSHead, FCOSHead):
    """Head of `VarifocalNet (VFNet): An IoU-aware Dense Object
    Detector.<https://arxiv.org/abs/2008.13367>`_.

    The VFNet predicts IoU-aware classification scores which mix the
    object presence confidence and object localization accuracy as the
    detection score. It is built on the FCOS architecture and uses ATSS
    for defining positive/negative training examples. The VFNet is trained
    with Varifocal Loss and empolys star-shaped deformable convolution to
    extract features for a bbox.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        sync_num_pos (bool): If true, synchronize the number of positive
            examples across GPUs. Default: True
        gradient_mul (float): The multiplier to gradients from bbox refinement
            and recognition. Default: 0.1.
        bbox_norm_type (str): The bbox normalization type, 'reg_denom' or
            'stride'. Default: reg_denom
        loss_cls_fl (dict): Config of focal loss.
        use_vfl (bool): If true, use varifocal loss for training.
            Default: True.
        loss_cls (dict): Config of varifocal loss.
        loss_bbox (dict): Config of localization loss, GIoU Loss.
        loss_bbox (dict): Config of localization refinement loss, GIoU Loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32,
            requires_grad=True).
        use_atss (bool): If true, use ATSS to define positive/negative
            examples. Default: True.
        anchor_generator (dict): Config of anchor generator for ATSS.
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> self = VFNetHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, bbox_pred_refine= self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 sync_num_pos=True,
                 gradient_mul=0.1,
                 bbox_norm_type='reg_denom',
                 loss_cls_fl=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 use_vfl=True,
                 loss_cls=dict(
                     type='VarifocalLoss',
                     use_sigmoid=True,
                     alpha=0.75,
                     gamma=2.0,
                     iou_weighted=True,
                     loss_weight=1.0),
                 loss_bbox=dict(type='GIoULoss', loss_weight=1.5),
                 loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0),
                 loss_rbox=dict( type='RotatedIoULoss', loss_weight=1.5),
                 loss_rbox_refine=dict( type='RotatedIoULoss', loss_weight=2.0),
                 loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 use_atss=True,
                 down_ratio=8,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     ratios=[1.0],
                     octave_base_scale=8,
                     scales_per_octave=1,
                     center_offset=0.0,
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='vfnet_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        # dcn base offsets, adapted from reppoints_head.py
        self.num_dconv_points = 9
        self.dcn_kernel = int(np.sqrt(self.num_dconv_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)
        self.reg_channel=5
        super(FCOSHead, self).__init__(
            num_classes,
            in_channels,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.regress_ranges = regress_ranges
        self.reg_denoms = [
            regress_range[-1] for regress_range in regress_ranges
        ]
        self.reg_denoms[-1] = self.reg_denoms[-2] * 2
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.sync_num_pos = sync_num_pos
        self.bbox_norm_type = bbox_norm_type
        self.gradient_mul = gradient_mul
        self.use_vfl = use_vfl
        if self.use_vfl:
            self.loss_cls = build_loss(loss_cls)
        else:
            self.loss_cls = build_loss(loss_cls_fl)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_bbox_refine = build_loss(loss_bbox_refine)
        self.loss_rbox = build_loss(loss_rbox)
        self.loss_rbox_refine = build_loss(loss_rbox_refine)
        self.loss_center_heatmap=build_loss(loss_center_heatmap)
        # for getting ATSS targets
        self.use_atss = use_atss
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.anchor_generator = build_anchor_generator(anchor_generator)
        self.anchor_center_offset = anchor_generator['center_offset']
        self.num_anchors = self.anchor_generator.num_base_anchors[0]
        self.sampling = False
        self.down_ratio = down_ratio
        self.alpha=0.54
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
    
    def _init_layers(self):
        """Initialize layers of the head."""
        super(FCOSHead, self)._init_cls_convs()
        super(FCOSHead, self)._init_reg_convs()
        self.relu = nn.ReLU(inplace=True)
        self.vfnet_reg_conv = ConvModule(
            self.feat_channels,
            self.feat_channels,
            3,
            stride=1,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            bias=self.conv_bias)
        self.vfnet_reg = nn.Conv2d(self.feat_channels, self.reg_channel, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

        self.vfnet_reg_refine_dconv = DeformConv2d(
            self.feat_channels,
            self.feat_channels,
            self.dcn_kernel,
            1,
            padding=self.dcn_pad)
        self.vfnet_reg_refine = nn.Conv2d(self.feat_channels, self.reg_channel, 3, padding=1)
        self.scales_refine = nn.ModuleList([Scale(1.0) for _ in self.strides])

        self.vfnet_cls_dconv = DeformConv2d(
            self.feat_channels,
            self.feat_channels,
            self.dcn_kernel,
            1,
            padding=self.dcn_pad)
        self.vfnet_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.vfnet_sem_conv = ConvModule(
            self.feat_channels,
            self.feat_channels,
            3,
            stride=1,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            bias=self.conv_bias)
        self.sem_out = nn.Conv2d(self.feat_channels, self.cls_out_channels,kernel_size=1)#这里获得了类别数 的通道数量 用的1*1卷积
        bias_init = bias_init_with_prob(0.1)
        self.sem_out.bias.data.fill_(bias_init)
    
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas) 
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore,gt_masks=gt_masks)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box iou-aware scores for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box offsets for each
                    scale level, each is a 4D-tensor, the channel number is
                    num_points * 4.
                bbox_preds_refine (list[Tensor]): Refined Box offsets for
                    each scale level, each is a 4D-tensor, the channel
                    number is num_points * 4.
        """
        return multi_apply(self.forward_single, feats, self.scales,
                           self.scales_refine, self.strides, self.reg_denoms)

    def forward_single(self, x, scale, scale_refine, stride, reg_denom):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            scale_refine (:obj: `mmcv.cnn.Scale`): Learnable scale module to
                resize the refined bbox prediction.
            stride (int): The corresponding stride for feature maps,
                used to normalize the bbox prediction when
                bbox_norm_type = 'stride'.
            reg_denom (int): The corresponding regression range for feature
                maps, only used to normalize the bbox prediction when
                bbox_norm_type = 'reg_denom'.

        Returns:
            tuple: iou-aware cls scores for each box, bbox predictions and
                refined bbox predictions of input feature maps.
        """
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)

        # predict the bbox_pred of different level
        reg_feat_init = self.vfnet_reg_conv(reg_feat)
        if self.bbox_norm_type == 'reg_denom':
            # bbox_pred = scale(self.vfnet_reg(reg_feat_init)).float().exp() * reg_denom
            bbox_pred_ini = scale(self.vfnet_reg(reg_feat_init)).float() #[:, 0:2, :, :]
            bbox_pred=bbox_pred_ini[:, 0:2, :, :]*reg_denom
            bbox_pred1=bbox_pred_ini[:, 2:4, :, :].exp() * reg_denom
            bbox_pred2=bbox_pred_ini[:, 4:, :, :]
            bbox_pred = torch.cat((bbox_pred,bbox_pred1,bbox_pred2),dim=1)
        elif self.bbox_norm_type == 'stride':
            bbox_pred = scale(
                self.vfnet_reg(reg_feat_init)).float().exp() * stride
        else:
            raise NotImplementedError
        
        if self.training and stride==8:
            # predict the bbox_pred of different level
            sem_feat= self.vfnet_sem_conv(cls_feat)
            sem_out = self.sem_out(sem_feat).sigmoid()
        else:
            sem_out=None
        # compute star deformable convolution offsets
        # converting dcn_offset to reg_feat.dtype thus VFNet can be
        # trained with FP16
        dcn_offset = self.r_dcn_offset(bbox_pred, self.gradient_mul,
                                          stride).to(reg_feat.dtype)

        # refine the bbox_pred
        reg_feat = self.relu(self.vfnet_reg_refine_dconv(reg_feat, dcn_offset))
        bbox_pred_refine = scale_refine(
            self.vfnet_reg_refine(reg_feat)).float()
        bbox_pred_refine = bbox_pred_refine + bbox_pred.detach()#变为加号

        # predict the iou-aware cls score
        cls_feat = self.relu(self.vfnet_cls_dconv(cls_feat, dcn_offset))
        cls_score = self.vfnet_cls(cls_feat)

        return cls_score, bbox_pred, bbox_pred_refine, sem_out

    def r_dcn_offset(self, bbox_pred, gradient_mul, stride):
        dcn_base_offset = self.dcn_base_offset.type_as(bbox_pred)
        bbox_pred_grad_mul = (1 - gradient_mul) * bbox_pred.detach() + \
            gradient_mul * bbox_pred
        # map to the feature map scale
        bbox_pred_grad_mul = bbox_pred_grad_mul / stride
        N, C, H, W = bbox_pred.size()
        rboxes=bbox_pred_grad_mul[:,0: 5, :, :]#torch.cat((cx,cy,w,h,angle),1)
        rboxes=rboxes.permute(0, 2, 3, 1).reshape(-1, 5)
        #cx cy需要变个符号
        rboxes[:, 0:2]*=-1.0
        ploy_points=rotated_box_to_poly(rboxes)
        ploy_points=ploy_points.reshape(N,H,W,8).permute(0, 3, 1, 2)
        x1 = ploy_points[:, 0, :, :]
        y1 = ploy_points[:, 1, :, :]
        x2 = ploy_points[:, 2, :, :]
        y2 = ploy_points[:, 3, :, :]
        x3 = ploy_points[:, 4, :, :]
        y3 = ploy_points[:, 5, :, :]
        x4 = ploy_points[:, 6, :, :]
        y4 = ploy_points[:, 7, :, :]

        bbox_pred_grad_mul_offset = bbox_pred.new_zeros(
            N, 2 * self.num_dconv_points, H, W)
        bbox_pred_grad_mul_offset[:, 0, :, :] = y1  #1
        bbox_pred_grad_mul_offset[:, 1, :, :] = x1  
        bbox_pred_grad_mul_offset[:, 2, :, :] = (y2+y1)/2.0#2
        bbox_pred_grad_mul_offset[:, 3, :, :] = (x2+x1)/2.0
        bbox_pred_grad_mul_offset[:, 4, :, :] = y2#3
        bbox_pred_grad_mul_offset[:, 5, :, :] = x2  
        bbox_pred_grad_mul_offset[:, 6, :, :] =(y4+y1)/2.0#4
        bbox_pred_grad_mul_offset[:, 7, :, :] =   (x4+x1)/2.0
        # bbox_pred_grad_mul_offset[:, 8, :, :] =(y4+y1+y2+y3)/2.0#4
        # bbox_pred_grad_mul_offset[:, 9, :, :] =   (x4+x1+x2+x3)/2.0
        bbox_pred_grad_mul_offset[:, 10, :, :] =  (y2+y3)/2.0#6
        bbox_pred_grad_mul_offset[:, 11, :, :] =  (x2+x3)/2.0#6
        bbox_pred_grad_mul_offset[:, 12, :, :] = y4  # 7
        bbox_pred_grad_mul_offset[:, 13, :, :] =  x4  # 7
        bbox_pred_grad_mul_offset[:, 14, :, :] = (y4+y3)/2.0 # 8
        bbox_pred_grad_mul_offset[:, 15, :, :] = (x4+x3)/2.0#
        bbox_pred_grad_mul_offset[:, 16, :, :] = -1.0 * y3  # 9
        bbox_pred_grad_mul_offset[:, 17, :, :] = -1.0 * x3  # 9
        return bbox_pred_grad_mul_offset 

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'bbox_preds_refine'))
    def loss(self,
             cls_scores,
             bbox_preds,
             bbox_preds_refine,sem_scores,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None,
             gt_masks=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box offsets for each
                scale level, each is a 4D-tensor, the channel number is
                num_points * 4.
            bbox_preds_refine (list[Tensor]): Refined Box offsets for
                each scale level, each is a 4D-tensor, the channel
                number is num_points * 4.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
                Default: None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(bbox_preds_refine)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        # gt_bboxes_new=[]
        # for gt_bbox ,gtmask in zip(gt_bboxes, gt_masks) :
        #     gt_boxes_new.append(gt_bbox[:, 0:4])
        # print(gt_bboxes)
        # print(gt_masks)
        
        labels, label_weights, bbox_targets, rbox_targets,bbox_weights = self.get_targets(
            cls_scores, all_level_points, gt_bboxes, gt_masks,gt_labels, img_metas,
            gt_bboxes_ignore)
        
        gt_sem_map,hm_weight = self.seg_target_generator(gt_masks,gt_labels, img_metas)
        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and bbox_preds_refine
        flatten_cls_scores = [   cls_score.permute(0, 2, 3,   1).reshape(-1,self.cls_out_channels).contiguous()     for cls_score in cls_scores    ]
        flatten_bbox_preds = [  bbox_pred.permute(0, 2, 3, 1).reshape(-1, self.reg_channel).contiguous()  for bbox_pred in bbox_preds    ]
        flatten_bbox_preds_refine = [  bbox_pred_refine.permute(0, 2, 3, 1).reshape(-1, self.reg_channel).contiguous()     for bbox_pred_refine in bbox_preds_refine    ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_bbox_preds_refine = torch.cat(flatten_bbox_preds_refine)
        flatten_labels = torch.cat(labels)
        # flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_rbox_targets = torch.cat(rbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(  [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes - 1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = torch.where(
            ((flatten_labels >= 0) & (flatten_labels < bg_class_ind)) > 0)[0]
        num_pos = len(pos_inds)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_bbox_preds_refine = flatten_bbox_preds_refine[pos_inds]
        pos_labels = flatten_labels[pos_inds]

        # sync num_pos across all gpus
        if self.sync_num_pos:
            num_pos_avg_per_gpu = reduce_mean(
                pos_inds.new_tensor(num_pos).float()).item()
            num_pos_avg_per_gpu = max(num_pos_avg_per_gpu, 1.0)
        else:
            num_pos_avg_per_gpu = num_pos

        # pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_rbox_targets = flatten_rbox_targets[pos_inds]
        pos_points = flatten_points[pos_inds]

        # pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
        # pos_decoded_target_preds = distance2bbox(pos_points, pos_bbox_targets)
        pos_decoded_rbox_preds_angle = distance2rbox(pos_points, pos_bbox_preds)#解码直接预测的rbox
        pos_decoded_rbox_target= distance2bbox(pos_points, pos_rbox_targets)#[...,4:12])#解码回归的四个点
        pos_decoded_rbox_target_angle=poly_to_rotated_box(pos_decoded_rbox_target,pos_labels,refine_angle_with_label=True)
        # iou_targets_ini = bbox_overlaps(
        #     pos_decoded_bbox_preds,
        #     pos_decoded_target_preds.detach(),
        #     is_aligned=True).clamp(min=1e-6)
        iou_targets_ini = box_iou_rotated(
            pos_decoded_rbox_preds_angle,
            pos_decoded_rbox_target_angle.detach(),
            aligned=True).clamp(min=1e-6)
        bbox_weights_ini = iou_targets_ini.clone().detach()
        bbox_avg_factor_ini = reduce_mean(
            bbox_weights_ini.sum()).clamp_(min=1).item()
        bbox_weights_ini=bbox_weights_ini[:, None].repeat(1, 5)
         #refine stage
        # pos_decoded_bbox_preds_refine = distance2bbox(pos_points, pos_bbox_preds_refine)
        pos_decoded_rbox_preds_refine_angle= distance2rbox(pos_points, pos_bbox_preds_refine)
        # iou_targets_rf = bbox_overlaps(
        #     pos_decoded_bbox_preds_refine,
        #     pos_decoded_target_preds.detach(),
        #     is_aligned=True).clamp(min=1e-6)
        iou_targets_rf = box_iou_rotated(
            pos_decoded_rbox_preds_refine_angle,
            pos_decoded_rbox_target_angle.detach(),
            aligned=True).clamp(min=1e-6)
        bbox_weights_rf = iou_targets_rf.clone().detach()
        bbox_avg_factor_rf = reduce_mean(
            bbox_weights_rf.sum()).clamp_(min=1).item()
        bbox_weights_rf=bbox_weights_rf[:, None].repeat(1, 5)
        if num_pos > 0:
            loss_rbox = self.loss_rbox(
                pos_decoded_rbox_preds_angle,
                pos_decoded_rbox_target_angle.detach(),
                weight=bbox_weights_ini,
                avg_factor=bbox_avg_factor_ini)

            loss_rbox_refine = self.loss_rbox_refine(
                pos_decoded_rbox_preds_refine_angle,
                pos_decoded_rbox_target_angle.detach(),
                weight=bbox_weights_rf,
                avg_factor=bbox_avg_factor_rf)

            # build IoU-aware cls_score targets
            if self.use_vfl:
                pos_ious = iou_targets_rf.clone().detach()
                cls_iou_targets = torch.zeros_like(flatten_cls_scores)
                cls_iou_targets[pos_inds, pos_labels] = pos_ious
        else:
            loss_rbox = pos_bbox_preds.sum() * 0
            loss_rbox_refine = pos_bbox_preds_refine.sum() * 0
            if self.use_vfl:
                cls_iou_targets = torch.zeros_like(flatten_cls_scores)

        if self.use_vfl:
            loss_cls = self.loss_cls(
                flatten_cls_scores,
                cls_iou_targets,
                avg_factor=num_pos_avg_per_gpu)
        else:
            loss_cls = self.loss_cls(
                flatten_cls_scores,
                flatten_labels,
                weight=label_weights,
                avg_factor=num_pos_avg_per_gpu)
            
        #compute sem loss
        # gt_lvl_sem_map=#torch.stack(gt_sem_map,dim=0)#cat batch
        # loss_sem=pos_bbox_preds.new_tensor(0)
        # m = nn.AvgPool2d((2, 2), stride=(2, 2))
        H, W = sem_scores[0].shape[2:]
        mask = hm_weight.view(-1, H, W)
        avg_factor = mask.sum() + 1e-4
        loss_sem = self.loss_center_heatmap(sem_scores[0], gt_sem_map.detach(),avg_factor=avg_factor)
        # gt_lvl_sem_map=m(gt_lvl_sem_map)

        return dict(
            loss_cls=loss_cls,
            loss_rbox=loss_rbox,
            loss_rbox_rf=loss_rbox_refine,
            loss_sem=loss_sem)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'bbox_preds_refine'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   bbox_preds_refine,sem_out,
                   img_metas,
                   cfg=None,
                   rescale=None,
                   with_nms=True):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for each scale
                level with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box offsets for each scale
                level with shape (N, num_points * 4, H, W).
            bbox_preds_refine (list[Tensor]): Refined Box offsets for
                each scale level with shape (N, num_points * 4, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before returning boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds) == len(bbox_preds_refine)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds_refine[i][img_id].detach()
                for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(cls_score_list,
                                                 bbox_pred_list, mlvl_points,
                                                 img_shape, scale_factor, cfg,
                                                 rescale, with_nms)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for a single scale
                level with shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box offsets for a single scale
                level with shape (num_points * 4, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before returning boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes,mlvl_rboxes = [],[]
        mlvl_scores = []
        for cls_score, bbox_pred, points in zip(cls_scores, bbox_preds,
                                                mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).contiguous().sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, self.reg_channel).contiguous()

            nms_pre = cfg.get('nms_pre', -1)
            if 0 < nms_pre < scores.shape[0]:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            
            # bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            rboxes = distance2rbox(points, bbox_pred, max_shape=img_shape)
            bboxes = rotated_box_to_poly(rboxes)
            mlvl_bboxes.append(bboxes)
            mlvl_rboxes.append(rboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_rboxes = torch.cat(mlvl_rboxes)
        if rescale:
            # mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
            mlvl_bboxes[...,0:4] /= mlvl_bboxes.new_tensor(scale_factor)
            mlvl_bboxes[...,4:8] /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        if with_nms:
            # det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
            #                                         cfg.score_thr, cfg.nms,
            #                                         cfg.max_per_img)
            det_bboxes,  det_labels = multiclass_nms_rotated_bbox(mlvl_rboxes,mlvl_bboxes,
                                                        mlvl_scores,
                                                        cfg.score_thr, cfg.nms,
                                                        cfg.max_per_img)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map sizes."""
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        # to be compatible with anchor points in ATSS
        if self.use_atss:
            points = torch.stack(
                (x.reshape(-1), y.reshape(-1)), dim=-1) + \
                     stride * self.anchor_center_offset
        else:
            points = torch.stack(
                (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points

    def get_targets(self, cls_scores, mlvl_points, gt_bboxes, gt_masks, gt_labels,
                    img_metas, gt_bboxes_ignore):
        """A wrapper for computing ATSS and FCOS targets for points in multiple
        images.

        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for each scale
                level with shape (N, num_points * num_classes, H, W).
            mlvl_points (list[Tensor]): Points of each fpn level, each has
                shape (num_points, 2).
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level.
                label_weights (Tensor/None): Label weights of all levels.
                bbox_targets_list (list[Tensor]): Regression targets of each
                    level, (l, t, r, b).
                bbox_weights (Tensor/None): Bbox weights of all levels.
        """
        # gt_boxes_new = []
        # gt_rboxes = []
        # for gt_bbox in gt_bboxes:
        #     # gt_rboxes.append(gt_bbox[:, 4:])
        #     gt_rboxes.append(gt_bbox)
        #     gt_boxes_new.append(gt_bbox[:, 0:4])
        # gt_bboxes = gt_boxes_new
        gt_rboxes=gt_masks
        if self.use_atss:
            return self.get_atss_targets(cls_scores, mlvl_points, gt_bboxes,gt_rboxes,
                                         gt_labels, img_metas,
                                         gt_bboxes_ignore)
        else:
            self.norm_on_bbox = False
            return self.get_fcos_targets(mlvl_points, gt_bboxes, gt_labels)

    def _get_target_single(self, *args, **kwargs):
        """Avoid ambiguity in multiple inheritance."""
        if self.use_atss:
            return ATSSHead._get_target_single(self, *args, **kwargs)
        else:
            return FCOSHead._get_target_single(self, *args, **kwargs)

    def get_fcos_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute FCOS regression and classification targets for points in
        multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                labels (list[Tensor]): Labels of each level.
                label_weights: None, to be compatible with ATSS targets.
                bbox_targets (list[Tensor]): BBox targets of each level.
                bbox_weights: None, to be compatible with ATSS targets.
        """
        labels, bbox_targets = FCOSHead.get_targets(self, points,
                                                    gt_bboxes_list,
                                                    gt_labels_list)
        label_weights = None
        bbox_weights = None
        return labels, label_weights, bbox_targets, bbox_weights

    def get_atss_targets(self,
                         cls_scores,
                         mlvl_points,
                         gt_bboxes,
                         gt_rboxes,
                         gt_labels,
                         img_metas,
                         gt_bboxes_ignore=None):
        """A wrapper for computing ATSS targets for points in multiple images.

        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for each scale
                level with shape (N, num_points * num_classes, H, W).
            mlvl_points (list[Tensor]): Points of each fpn level, each has
                shape (num_points, 2).
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4). Default: None.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level.
                label_weights (Tensor): Label weights of all levels.
                bbox_targets_list (list[Tensor]): Regression targets of each
                    level, (l, t, r, b).
                bbox_weights (Tensor): Bbox weights of all levels.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets = ATSSHead.get_targets(
            self,
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            gt_rboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            unmap_outputs=True)
        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list, rbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets
        #这里的list 是根据bathsize设置的一张图一个元素,接下来的操作就是去掉图的这个以维度 按照level组织list
        bbox_targets_list = [ bbox_targets.reshape(-1, 4) for bbox_targets in bbox_targets_list ]
        # bbox_targets_list1 = [ bbox_targets.reshape(-1, 12)[...,:4]for bbox_targets in rbox_targets_list ]
        rbox_targets_list = [ bbox_targets.reshape(-1, 8)for bbox_targets in rbox_targets_list ]#[...,4:12]

        num_imgs = len(img_metas)

        # transform bbox_targets (x1, y1, x2, y2) into (l, t, r, b) format
        bbox_targets_list = self.transform_bbox_targets( bbox_targets_list, mlvl_points, num_imgs)
        #TODO insert
        rbox_targets_list = self.transform_bbox_targets( rbox_targets_list, mlvl_points, num_imgs)


        labels_list = [labels.reshape(-1) for labels in labels_list]
        # labels_k_list = [labels.reshape(-1) for labels in labels_k_list]
        label_weights_list = [  label_weights.reshape(-1) for label_weights in label_weights_list ]
        bbox_weights_list = [ bbox_weights.reshape(-1) for bbox_weights in bbox_weights_list  ]
        label_weights = torch.cat(label_weights_list)
        bbox_weights = torch.cat(bbox_weights_list)
        return labels_list,  label_weights, bbox_targets_list, rbox_targets_list, bbox_weights


    def transform_bbox_targets(self, decoded_bboxes, mlvl_points, num_imgs):
        """Transform bbox_targets (x1, y1, x2, y2) into (l, t, r, b) format.

        Args:
            decoded_bboxes (list[Tensor]): Regression targets of each level,
                in the form of (x1, y1, x2, y2).
            mlvl_points (list[Tensor]): Points of each fpn level, each has
                shape (num_points, 2).
            num_imgs (int): the number of images in a batch.

        Returns:
            bbox_targets (list[Tensor]): Regression targets of each level in
                the form of (l, t, r, b).
        """
        # TODO: Re-implemented in Class PointCoder
        assert len(decoded_bboxes) == len(mlvl_points)
        num_levels = len(decoded_bboxes)
        mlvl_points = [points.repeat(num_imgs, 1) for points in mlvl_points]
        bbox_targets = []
        for i in range(num_levels):
            bbox_target = bbox2distance(mlvl_points[i], decoded_bboxes[i])
            bbox_targets.append(bbox_target)

        return bbox_targets

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Override the method in the parent class to avoid changing para's
        name."""
        pass
    
    #add centernet model
    def gaussian2D(self, shape, sigma_x=1, sigma_y=1, dtype=torch.float32, device='cpu'):
        """Generate 2D gaussian kernel.

        Args:
            radius (int): Radius of gaussian kernel.
            sigma (int): Sigma of gaussian function. Default: 1.
            dtype (torch.dtype): Dtype of gaussian tensor. Default: torch.float32.
            device (str): Device of gaussian tensor. Default: 'cpu'.

        Returns:
            h (Tensor): Gaussian kernel with a
                ``(2 * radius + 1) * (2 * radius + 1)`` shape.
        """
        m, n = [(ss - 1.) / 2. for ss in shape]
        x = torch.arange(
            -n, n + 1, dtype=dtype, device=device).view(1, -1)
        y = torch.arange(
            -m, m + 1, dtype=dtype, device=device).view(-1, 1)
        h = (-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y))).exp()
        # h = (-(x * x + y * y) / (2 * sigma * sigma)).exp()

        h[h < torch.finfo(h.dtype).eps * h.max()] = 0
        return h

    def imshow_gpu_tensor(self, tensor):#调试中显示表标签图
        from PIL import Image
        from torchvision import transforms
        image = tensor.cpu().clone().numpy()
        image = image.astype(np.uint8)
        image = transforms.ToPILImage()(image)
        # image = image.resize((1024, 1024),Image.ANTIALIAS)
        image.show(image)
        image.save('./hm.jpg')

    def rot_img(self,x, theta):
        # theta = torch.tensor(theta)theta
        theta*=-1.0#表示这里是反的顺时针旋转 是负号
        rot_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                            [torch.sin(theta), torch.cos(theta), 0]],device=x.device)
        rot_mat=rot_mat[None, ...]
        # rot_mat = x.new_tensor(rot_mat)
        
        out_size = torch.Size((1, 1,  x.size()[0],  x.size()[1]))
        grid = F.affine_grid(rot_mat, out_size)
        # grid =  x.new_tensor(grid)
        rotate = F.grid_sample(x.unsqueeze(0).unsqueeze(0), grid)
        return rotate.squeeze(0).squeeze(0)
    
    
    def draw_truncate_gaussian(self, heatmap, center, h_radius, w_radius,angle, k=1):
        h, w = 2 * h_radius + 1, 2 * w_radius + 1
        sigma_x = w / 6
        sigma_y = h / 6
        # gaussian = self.gaussian_2d((h, w), sigma_x=sigma_x, sigma_y=sigma_y)
        gaussian = self.gaussian2D((h, w), sigma_x=sigma_x, sigma_y=sigma_y, device=heatmap.device)
        # angle= math.atan2((pts[0, 0] - ct[0]), (pts[0, 1] - ct[1]))
        # gaussian = self.gaussian2D_rotate_ellipse((h, w), sigma_x=sigma_x, sigma_y=sigma_y,angle=)
        # gaussian = heatmap.new_tensor(gaussian)
        
        #这里添加转换函数
        max_size=max(gaussian.size()[0],gaussian.size()[1])
        gaussian_expand=torch.zeros((max_size,max_size),device=heatmap.device)
        if gaussian.size()[0]==max_size:
            start=int((max_size-gaussian.size()[1])/2)
            gaussian_expand[:,start:start+gaussian.size()[1]]=gaussian
        else:
            start=int((max_size-gaussian.size()[0])/2)
            gaussian_expand[start:start+gaussian.size()[0],:]=gaussian
            
        gaussian_expand = heatmap.new_tensor(gaussian_expand)
        # dtype =  torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        rotated_im = self.rot_img(gaussian_expand, angle) 
        # self.imshow_gpu_tensor(gaussian_expand*255)
        # self.imshow_gpu_tensor(rotated_im*255)
        
        x, y = int(center[0]), int(center[1])
        height, width = heatmap.shape[0:2]
        # left, right = min(x, w_radius), min(width - x, w_radius + 1)#为了不越界
        # top, bottom = min(y, h_radius), min(height - y, h_radius + 1)
        max_radius=max(w_radius,h_radius)
        left, right = min(x, max_radius), min(width - x, max_radius + 1)#为了不越界
        top, bottom = min(y, max_radius), min(height - y, max_radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = rotated_im[max_radius - top:max_radius + bottom,
                          max_radius - left:max_radius + right]
    
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def target_single_image(self, gt_rboxes, gt_labels, feat_shape):
        device=gt_rboxes.device
        if gt_rboxes.shape[0]<1:# !!!!
            gt_rboxes=torch.zeros((gt_rboxes.shape[0],5)).to(device)
        else:
            gt_rboxes=poly_to_rotated_box(gt_rboxes,gt_labels,refine_angle_with_label=True)

        x_ctr, y_ctr, width, height, angle = gt_rboxes.select(1, 0), gt_rboxes.select(
            1, 1), gt_rboxes.select(1, 2), gt_rboxes.select(1, 3), gt_rboxes.select(1, 4)
  
        gt_boxes= torch.stack([x_ctr - width * 0.5, y_ctr-height * 0.5,x_ctr + width * 0.5, y_ctr + height * 0.5], dim=0).permute(1, 0)
        output_h, output_w = feat_shape
        heatmap_channel = self.num_classes
        heatmap = gt_boxes.new_zeros((heatmap_channel, output_h, output_w))
        fake_heatmap = gt_boxes.new_zeros((output_h, output_w))
        reg_weight = gt_boxes.new_zeros((1, output_h, output_w))

        boxes_areas_log = (width*height).log()
        boxes_area_topk_log, boxes_ind = torch.topk(boxes_areas_log, boxes_areas_log.size(0))

        gt_boxes = gt_boxes[boxes_ind]
        gt_labels = gt_labels[boxes_ind]
        gt_rboxes = gt_rboxes[boxes_ind]
        
        feat_gt_boxes = gt_boxes / self.down_ratio #8
        feat_gt_boxes[:, [0, 2]] = torch.clamp(feat_gt_boxes[:, [0, 2]], min=0,
                                               max=output_w - 1)
        feat_gt_boxes[:, [1, 3]] = torch.clamp(feat_gt_boxes[:, [1, 3]], min=0,
                                               max=output_h - 1)
        feat_hs, feat_ws = (feat_gt_boxes[:, 3] - feat_gt_boxes[:, 1],
                            feat_gt_boxes[:, 2] - feat_gt_boxes[:, 0])

        # we calc the center and ignore area based on the gt-boxes of the origin scale
        # no peak will fall between pixels
        ct_ints = (torch.stack([(gt_boxes[:, 0] + gt_boxes[:, 2]) / 2,
                                (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2],
                               dim=1) / self.down_ratio).to(torch.int)

        h_radiuses_alpha = (feat_hs / 2. * self.alpha).int()
        w_radiuses_alpha = (feat_ws / 2. * self.alpha).int()
        angle=gt_rboxes[:,4]

        # larger boxes have lower priority than small boxes.#这里开始计算bbox的回归框
        for k in range(boxes_ind.shape[0]):
            cls_id = gt_labels[k] 
            fake_heatmap = fake_heatmap.zero_()
            self.draw_truncate_gaussian(fake_heatmap, ct_ints[k],
                                        h_radiuses_alpha[k].item(), w_radiuses_alpha[k].item(),angle[k])
            heatmap[cls_id] = torch.max(heatmap[cls_id], fake_heatmap)
     
            box_target_inds = fake_heatmap > 0
            cls_id = 0
            
            local_heatmap = fake_heatmap[box_target_inds]
            ct_div = local_heatmap.sum()
            local_heatmap *= boxes_area_topk_log[k]
            reg_weight[cls_id, box_target_inds] = local_heatmap / ct_div
           
        return heatmap, heatmap, reg_weight
    

    def seg_target_generator(self, gt_rboxes, gt_labels, img_metas):
        """

        Args:
            gt_boxes: list(tensor). tensor <=> image, (gt_num, 4).
            gt_labels: list(tensor). tensor <=> image, (gt_num,).
            img_metas: list(dict).

        Returns:
            heatmap: tensor, (batch, 80, h, w).
            box_target: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
            reg_weight: tensor, same as box_target.
        """
        with torch.no_grad():
            feat_shape = (img_metas[0]['pad_shape'][0] // self.down_ratio,
                          img_metas[0]['pad_shape'][1] // self.down_ratio)
            heatmap, box_target, reg_weight = multi_apply(
                self.target_single_image,
                gt_rboxes,
                gt_labels,
                feat_shape=feat_shape
            )
            heatmap, box_target = [torch.stack(t, dim=0).detach() for t in [heatmap, box_target]]
            reg_weight = torch.stack(reg_weight, dim=0).detach()
            heatmap_max=torch.max(heatmap[0],dim=0)[0].squeeze()
            self.imshow_gpu_tensor(heatmap_max*255)
            # heatmaps=[]
            # for gt_rbox,gt_label in zip(gt_rboxes,gt_labels):
            #     heatmap, reg_weight= self.target_single_image(gt_rbox,gt_label,feat_shape)
            #     heatmaps.append(heatmap)
            return heatmap, reg_weight

        