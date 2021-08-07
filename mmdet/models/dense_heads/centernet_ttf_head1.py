import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32
import numpy as np
from mmdet.core import multi_apply,poly_to_rotated_box, rotated_box_to_poly,rotated_box_to_bbox
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from ..utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                     transpose_and_gather_feat)
from .base_dense_head import BaseDenseHead
from mmcv.cnn import  kaiming_init
from mmcv.ops import DeformConv2d

def bbox_areas(bboxes, keep_axis=False):
    x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    areas = (y_max - y_min + 1) * (x_max - x_min + 1)
    if keep_axis:
        return areas[:, None]
    return areas

def simple_nms(heat, kernel=3, out_heat=None):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    out_heat = heat if out_heat is None else out_heat
    return out_heat * keep
def get_topk_from_heatmap(scores, k=20):
    """Get top k positions from heatmap.

    Args:
        scores (Tensor): Target heatmap with shape
            [batch, num_classes, height, width].
        k (int): Target number. Default: 20.

    Returns:
        tuple[torch.Tensor]: Scores, indexes, categories and coords of
            topk keypoint. Containing following Tensors:

        - topk_scores (Tensor): Max scores of each topk keypoint.
        - topk_inds (Tensor): Indexes of each topk keypoint.
        - topk_clses (Tensor): Categories of each topk keypoint.
        - topk_ys (Tensor): Y-coord of each topk keypoint.
        - topk_xs (Tensor): X-coord of each topk keypoint.
    """
    batch, _, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), k)
    topk_clses = topk_inds // (height * width)
    topk_inds = topk_inds % (height * width)
    topk_ys = topk_inds // width
    topk_xs = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

@HEADS.register_module()
class CenterTTFNetHead(BaseDenseHead):
    """Objects as Points Head. CenterHead use center_point to indicate object's
    position. Paper link <https://arxiv.org/abs/1904.07850>

    Args:
        in_channel (int): Number of channel in the input feature map.
        feat_channel (int): Number of channel in the intermediate feature map.
        num_classes (int): Number of categories excluding the background
            category.
        loss_center_heatmap (dict | None): Config of center heatmap loss.
            Default: GaussianFocalLoss.
        loss_wh (dict | None): Config of wh loss. Default: L1Loss.
        loss_offset (dict | None): Config of offset loss. Default: L1Loss.
        train_cfg (dict | None): Training config. Useless in CenterNet,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CenterNet. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channel,
                 feat_channel,
                 num_classes,
                 loss_center_heatmap=dict(
                     type='GaussianFocalLoss', loss_weight=1.0),
                 loss_wh=dict(type='L1Loss', loss_weight=0.1),
                 loss_offset=dict(type='L1Loss', loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 base_down_ratio=32,
                 alpha=0.54,
                 beta=0.54,):
        super(CenterTTFNetHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.wh_planes = 5
        self.heatmap_head = self._build_head(in_channel, feat_channel,
                                             num_classes)
        self.wh_head = self._build_head(in_channel, feat_channel, self.wh_planes )
        # self.offset_head = self._build_head(in_channel, feat_channel, 2)

        self.loss_center_heatmap = build_loss(loss_center_heatmap)
        self.loss_wh = build_loss(loss_wh)
        # self.loss_offset = build_loss(loss_offset)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.down_ratio = 4
        self.wh_offset_base=16
        self.alpha = alpha
        self.beta = beta
        self.wh_gaussian = True
        self.wh_agnostic = True
        self.hm_weight=1.0
        self.base_loc = None

    def _build_head(self, in_channel, feat_channel, out_channel):
        """Build head for each branch."""
        layer = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, out_channel, kernel_size=1))
        return layer

    def init_weights(self):
        """Initialize weights of the head."""
        bias_init = bias_init_with_prob(0.1)
        self.heatmap_head[-1].bias.data.fill_(bias_init)
        # for head in [self.wh_head, self.offset_head]:
        for m in self.wh_head:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)

    def forward(self, feats):
        """Forward features. Notice CenterNet head does not use FPN.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            center_heatmap_preds (List[Tensor]): center predict heatmaps for
                all levels, the channels number is num_classes.
            wh_preds (List[Tensor]): wh predicts for all levels, the channels
                number is 2.
            offset_preds (List[Tensor]): offset predicts for all levels, the
               channels number is 2.
        """
        if isinstance(feats,tuple):
            feats=feats[0]
        center_heatmap_pred = self.heatmap_head(feats).sigmoid()
        wh_pred = self.wh_head(feats)
        
        wh =wh_pred[:,2:4,:,:].exp() * self.wh_offset_base

        bbox_pred = torch.cat((wh_pred[:,:2,:,:],wh,wh_pred[:,4:,:,:]),dim=1)

        return center_heatmap_pred, bbox_pred

    @force_fp32(apply_to=('pred_heatmap', 'pred_wh'))
    def get_bboxes(self,
                   pred_heatmap,
                   pred_wh,
                   img_metas,
                   rescale=False):
        batch, cat, height, width = pred_heatmap.size()
        pred_heatmap = pred_heatmap.detach().sigmoid_()
        wh = pred_wh.detach()

        # perform nms on heatmaps
        heat = simple_nms(pred_heatmap)  # used maxpool to filter the max score
        topk=self.test_cfg.topk
        scores, inds, clses, ys, xs = get_topk_from_heatmap(heat, k=topk)
        xs = xs.view(batch, topk, 1) * self.down_ratio
        ys = ys.view(batch, topk, 1) * self.down_ratio

        wh = wh.permute(0, 2, 3, 1).contiguous()
        wh = wh.view(wh.size(0), -1, wh.size(3))
        inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), wh.size(2))
        wh = wh.gather(1, inds)

        if not self.wh_agnostic:
            wh = wh.view(-1, topk, self.num_fg, 4)
            wh = torch.gather(wh, 2, clses[..., None, None].expand(
                clses.size(0), clses.size(1), 1, 4).long())

        wh = wh.view(batch, topk, self.wh_planes )
        clses = clses.view(batch, topk, 1).float()
        scores = scores.view(batch, topk, 1)

        bboxes = torch.cat([xs + wh[..., [0]], ys + wh[..., [1]], wh[...,2:]], dim=2)

        result_list = []
        score_thr =0.01
        for batch_i in range(bboxes.shape[0]):
            scores_per_img = scores[batch_i]
            scores_keep = (scores_per_img > score_thr).squeeze(-1)

            scores_per_img = scores_per_img[scores_keep]
            bboxes_per_img = bboxes[batch_i][scores_keep]
            labels_per_img = clses[batch_i][scores_keep]
            img_shape = img_metas[batch_i]['pad_shape']
            bboxes_per_img[:, 0:1] = bboxes_per_img[:, 0:1].clamp(min=0, max=img_shape[1] - 1)
            # bboxes_per_img[:, 1::2] = bboxes_per_img[:, 1::2].clamp(min=0, max=img_shape[0] - 1)

            if rescale:
                scale_factor = img_metas[batch_i]['scale_factor']
                bboxes_per_img[..., :4]  /= bboxes_per_img.new_tensor(scale_factor)

            batch_det_bboxes = torch.cat([bboxes_per_img, scores_per_img], dim=1)
            labels_per_img = labels_per_img.squeeze(-1)


            rbox=batch_det_bboxes[...,0:5].squeeze()
            bboxes = rotated_box_to_bbox(rbox)
            points = rotated_box_to_poly(rbox)
            batch_det_bboxes=torch.cat([bboxes[None,...], batch_det_bboxes[...,5:6],batch_det_bboxes[...,:5], points[None,...]],dim=2)
     

            result_list.append((batch_det_bboxes, labels_per_img))

        return result_list


    # def get_bboxes(self,
    #                center_heatmap_preds,
    #                wh_preds,
    #                img_metas,
    #                rescale=True,
    #                with_nms=False):
    #     """Transform network output for a batch into bbox predictions.

    #     Args:
    #         center_heatmap_preds (list[Tensor]): center predict heatmaps for
    #             all levels with shape (B, num_classes, H, W).
    #         wh_preds (list[Tensor]): wh predicts for all levels with
    #             shape (B, 2, H, W).
    #         offset_preds (list[Tensor]): offset predicts for all levels
    #             with shape (B, 2, H, W).
    #         img_metas (list[dict]): Meta information of each image, e.g.,
    #             image size, scaling factor, etc.
    #         rescale (bool): If True, return boxes in original image space.
    #             Default: True.
    #         with_nms (bool): If True, do nms before return boxes.
    #             Default: False.

    #     Returns:
    #         list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
    #             The first item is an (n, 5) tensor, where 5 represent
    #             (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
    #             The shape of the second tensor in the tuple is (n,), and
    #             each element represents the class label of the corresponding
    #             box.
    #     """
    #     assert len(center_heatmap_preds) == len(wh_preds) 
    #     scale_factors = [img_meta['scale_factor'] for img_meta in img_metas]
    #     # border_pixs = [img_meta['border'] for img_meta in img_metas]

    #     batch_det_bboxes, batch_labels = self.decode_heatmap(
    #         center_heatmap_preds,
    #         wh_preds,
    #         img_metas[0]['batch_input_shape'],
    #         k=self.test_cfg.topk,
    #         kernel=self.test_cfg.local_maximum_kernel)

    #     # batch_border = batch_det_bboxes.new_tensor(
    #     #     border_pixs)[:, [2, 0, 2, 0]].unsqueeze(1)
    #     # batch_det_bboxes[..., :4] -= batch_border

    #     if rescale:
    #         batch_det_bboxes[..., :4] /= batch_det_bboxes.new_tensor(
    #             scale_factors).unsqueeze(1)
    #     rbox=batch_det_bboxes[...,0:5].squeeze()
    #     bboxes = rotated_box_to_bbox(rbox)
    #     points = rotated_box_to_poly(rbox)
    #     batch_det_bboxes=torch.cat([bboxes[None,...], batch_det_bboxes[...,5:6],batch_det_bboxes[...,:5], points[None,...]],dim=2)
    #     if with_nms:
    #         det_results = []
    #         for (det_bboxes, det_labels) in zip(batch_det_bboxes,
    #                                             batch_labels):
    #             det_bbox, det_label = self._bboxes_nms(det_bboxes, det_labels,
    #                                                    self.test_cfg)
    #             det_results.append(tuple([det_bbox, det_label]))
    #     else:
    #         det_results = [
    #             tuple(bs) for bs in zip(batch_det_bboxes, batch_labels)
    #         ]
    #     return det_results

    # def decode_heatmap(self,
    #                    center_heatmap_pred,
    #                    wh_pred,
    #                    img_shape,
    #                    k=100,
    #                    kernel=3):
    #     """Transform outputs into detections raw bbox prediction.

    #     Args:
    #         center_heatmap_pred (Tensor): center predict heatmap,
    #            shape (B, num_classes, H, W).
    #         wh_pred (Tensor): wh predict, shape (B, 2, H, W).
    #         offset_pred (Tensor): offset predict, shape (B, 2, H, W).
    #         img_shape (list[int]): image shape in [h, w] format.
    #         k (int): Get top k center keypoints from heatmap. Default 100.
    #         kernel (int): Max pooling kernel for extract local maximum pixels.
    #            Default 3.

    #     Returns:
    #         tuple[torch.Tensor]: Decoded output of CenterNetHead, containing
    #            the following Tensors:

    #           - batch_bboxes (Tensor): Coords of each box with shape (B, k, 5)
    #           - batch_topk_labels (Tensor): Categories of each box with \
    #               shape (B, k)
    #     """
    #     height, width = center_heatmap_pred.shape[2:]
    #     inp_h, inp_w = img_shape

    #     center_heatmap_pred = get_local_maximum(
    #         center_heatmap_pred, kernel=kernel)

    #     *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
    #         center_heatmap_pred, k=k)
    #     batch_scores, batch_index, batch_topk_labels = batch_dets
    #     #因为target搞错了
    #     # batch_topk_labels+=1

    #     wh = transpose_and_gather_feat(wh_pred, batch_index)
    #     # offset = transpose_and_gather_feat(offset_pred, batch_index)
    #     # topk_xs = topk_xs + offset[..., 0]
    #     # topk_ys = topk_ys + offset[..., 1]
    #     c_x = (topk_xs + wh[..., 0] ) * (inp_w / width)
    #     c_y = (topk_ys + wh[..., 1] ) * (inp_h / height)

    #     batch_bboxes = torch.stack([c_x, c_y, wh[..., 2], wh[..., 3],wh[..., 4]], dim=2)
    #     batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]), dim=-1)
    #     return batch_bboxes, batch_topk_labels

    @force_fp32(apply_to=('pred_heatmap', 'pred_wh'))
    def loss(self,
             pred_heatmap,
             pred_wh,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        all_targets = self.target_generator(gt_bboxes, gt_labels, img_metas)
        heatmap,box_target,wh_weight=all_targets
        # hm_loss, wh_loss = self.loss_calc(pred_heatmap, pred_wh, *all_targets)
        
        # pred_hm=pred_heatmap[0]
        # pred_wh=pred_wh[0]
        H, W = pred_heatmap.shape[2:]
        mask = wh_weight.view(-1, H, W)
        avg_factor = mask.sum() + 1e-4

        hm_loss=self.loss_center_heatmap(pred_heatmap, heatmap.detach(),avg_factor=avg_factor)

        if self.base_loc is None or H != self.base_loc.shape[1] or W != self.base_loc.shape[2]:#True
            base_step = self.down_ratio
            shifts_x = torch.arange(0, (W - 1) * base_step + 1, base_step,
                                    dtype=torch.float32, device=heatmap.device)
            shifts_y = torch.arange(0, (H - 1) * base_step + 1, base_step,
                                    dtype=torch.float32, device=heatmap.device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            self.base_loc = torch.stack((shift_x, shift_y), dim=0)  # (2, h, w)

        # (batch, h, w, 4)
        pred_boxes = torch.cat((self.base_loc + pred_wh[:, [0, 1]],
                                pred_wh[:, [2, 3,4]]), dim=1).permute(0, 2, 3, 1).reshape(-1,self.wh_planes)
        # (batch, h, w, 4)
        boxes = box_target.permute(0, 2, 3, 1).detach().reshape(-1,self.wh_planes)
        weight_wh=mask[:,:,:,None].repeat((1,1,1,self.wh_planes)).reshape(-1,self.wh_planes)
        wh_loss=self.loss_wh( pred_boxes, boxes,  weight=weight_wh, avg_factor=avg_factor)

        return {'loss_heatmap': hm_loss, 'loss_rbox': wh_loss}

    # def _topk(self, scores, topk):
    #     batch, cat, height, width = scores.size()

    #     # both are (batch, 80, topk)
    #     topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), topk)

    #     topk_inds = topk_inds % (height * width)
    #     topk_ys = (topk_inds / width).int().float()
    #     topk_xs = (topk_inds % width).int().float()

    #     # both are (batch, topk). select topk from 80*topk
    #     topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), topk)
    #     topk_clses = (topk_ind / topk).int()
    #     topk_ind = topk_ind.unsqueeze(2)
    #     topk_inds = topk_inds.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)
    #     topk_ys = topk_ys.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)
    #     topk_xs = topk_xs.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)

    #     return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def gaussian_2d(self, shape, sigma_x=1, sigma_y=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

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
        image.show(image)
        # image.save('./heatmap.jpg')
        

    def rot_img(self,x, theta):
        # theta = torch.tensor(theta)theta
        theta=theta*(-1.0)#.to(device)#表示这里是反的顺时针旋转 是负号
        rot_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                            [torch.sin(theta), torch.cos(theta), 0]]).to(x.device)
        rot_mat=rot_mat[None, ...]
        out_size = torch.Size((1, 1,  x.size()[0],  x.size()[1]))
        grid = F.affine_grid(rot_mat, out_size)
        rotate = F.grid_sample(x.unsqueeze(0).unsqueeze(0), grid)
        return rotate.squeeze()
    
    
    def draw_truncate_gaussian(self, heatmap, center, h_radius, w_radius,angle, k=1):
        h, w = 2 * h_radius + 1, 2 * w_radius + 1
        sigma_x = w / 6
        sigma_y = h / 6
        gaussian = self.gaussian2D((h, w), sigma_x=sigma_x, sigma_y=sigma_y, device=heatmap.device)
        # gaussian2 = self.gaussian_2d((h, w), sigma_x=sigma_x, sigma_y=sigma_y)
        # gaussian = heatmap.new_tensor(gaussian)
        #这里添加转换函数
        max_size=max(gaussian.size()[0],gaussian.size()[1])
        gaussian_expand=torch.zeros((max_size,max_size),device=heatmap.device)#.to(device)
        if gaussian.size()[0]==max_size:
            start=int((max_size-gaussian.size()[1])/2)
            gaussian_expand[:,start:start+gaussian.size()[1]]=gaussian
        else:
            start=int((max_size-gaussian.size()[0])/2)
            gaussian_expand[start:start+gaussian.size()[0],:]=gaussian

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
        if max_size==1:
            masked_gaussian=rotated_im.unsqueeze(0).unsqueeze(0)#[y - top:y + bottom]
        else:
            masked_gaussian = rotated_im[max_radius - top:max_radius + bottom,
                                max_radius - left:max_radius + right]
        # masked_gaussian = rotated_im[max_radius - top:max_radius + bottom,
        #                   max_radius - left:max_radius + right]
    
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def target_single_image(self, gt_boxes, gt_labels, feat_shape):
        """

        Args:
            gt_boxes: tensor, tensor <=> img, (num_gt, 4).
            gt_labels: tensor, tensor <=> img, (num_gt,).
            feat_shape: tuple.

        Returns:
            heatmap: tensor, tensor <=> img, (80, h, w).
            box_target: tensor, tensor <=> img, (4, h, w) or (80 * 4, h, w).
            reg_weight: tensor, same as box_target
        """

        # gt_rboxes=gt_boxes[...,8:16].clone()
        gt_rboxes=gt_boxes[:, 8:16]
        device=gt_boxes.device
        if gt_rboxes.shape[1]<8:
            gt_rboxes=torch.zeros((gt_boxes.shape[0],5)).to(device)
        else:
            gt_rboxes=poly_to_rotated_box(gt_boxes[:, 8:16])
        gt_boxes=gt_boxes[...,:4]

        #计算头部关键点
        # N = gt_rboxes.shape[0]
        x_ctr, y_ctr, width, height, angle = gt_rboxes.select(1, 0), gt_rboxes.select(
            1, 1), gt_rboxes.select(1, 2), gt_rboxes.select(1, 3), gt_rboxes.select(1, 4)
        # tl_x, tl_y, br_x, br_y = -width * 0, -height * 0.5, width * 0.5, height * 0.5
        # rects = torch.stack([-height * 0, -height * 0.5], dim=0).reshape(2, 1, N).permute(2, 0, 1)
        # sin, cos = torch.sin(angle), torch.cos(angle)
        # M = torch.stack([cos, -sin, sin, cos],
        #                 dim=0).reshape(2, 2, N).permute(2, 0, 1)     # M.shape=[N,2,2]
        # polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
        # polys[:, ::2] += x_ctr.unsqueeze(1)
        # polys[:, 1::2] += y_ctr.unsqueeze(1)
        #end
        
        #计算回归长宽的关键点
        gt_boxes= torch.stack([x_ctr - width * 0.5, y_ctr-height * 0.5,x_ctr + width * 0.5, y_ctr + height * 0.5], dim=0).permute(1, 0)
        output_h, output_w = feat_shape
        heatmap_channel = self.num_classes
        heatmap = gt_boxes.new_zeros((heatmap_channel, output_h, output_w))
        fake_heatmap = gt_boxes.new_zeros((output_h, output_w))
        box_target = gt_boxes.new_ones((self.wh_planes, output_h, output_w)) * -1
        reg_weight = gt_boxes.new_zeros((1, output_h, output_w))

        boxes_areas_log = bbox_areas(gt_boxes).log()
        boxes_area_topk_log, boxes_ind = torch.topk(boxes_areas_log, boxes_areas_log.size(0))

        gt_boxes = gt_boxes[boxes_ind]
        gt_labels = gt_labels[boxes_ind]
        gt_rboxes = gt_rboxes[boxes_ind]
        
        feat_gt_boxes = gt_boxes / self.down_ratio #4
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

            if self.wh_gaussian:#True
                box_target_inds = fake_heatmap > 0
            else:
                ctr_x1, ctr_y1, ctr_x2, ctr_y2 = ctr_x1s[k], ctr_y1s[k], ctr_x2s[k], ctr_y2s[k]
                box_target_inds = torch.zeros_like(fake_heatmap, dtype=torch.uint8)
                box_target_inds[ctr_y1:ctr_y2 + 1, ctr_x1:ctr_x2 + 1] = 1

            if self.wh_agnostic:
                # gt_boxes_reg=torch.cat((gt_boxes,gt_rboxes[:,4:5]),dim=1)
                box_target[:, box_target_inds] = gt_rboxes[k][:, None]#让所有的点都回归这四个点
                cls_id = 0
            else:
                box_target[(cls_id * 4):((cls_id + 1) * 4), box_target_inds] = gt_boxes[k][:, None]

            if self.wh_gaussian:
                local_heatmap = fake_heatmap[box_target_inds]
                ct_div = local_heatmap.sum()
                local_heatmap *= boxes_area_topk_log[k]
                reg_weight[cls_id, box_target_inds] = local_heatmap / ct_div
            else:
                reg_weight[cls_id, box_target_inds] = \
                    boxes_area_topk_log[k] / box_target_inds.sum().float()
        # heatmap_max=torch.max(heatmap,dim=0)[0].squeeze()
        # self.imshow_gpu_tensor(heatmap_max*255)
        return heatmap, box_target, reg_weight

    def target_generator(self, gt_boxes, gt_labels, img_metas):
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
                gt_boxes,
                gt_labels,
                feat_shape=feat_shape
            )

            heatmap, box_target = [torch.stack(t, dim=0).detach() for t in [heatmap, box_target]]
            reg_weight = torch.stack(reg_weight, dim=0).detach()

            return heatmap, box_target, reg_weight
