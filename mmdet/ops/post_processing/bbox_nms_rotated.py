import torch

from mmdet.ops import ml_nms_rotated
# def multiclass_nms_rotated_bbox(multi_bboxes,multi_r2boxes,multi_points,
#                            multi_scores,
#                            score_thr,
#                            nms_cfg,
#                            max_num=-1,
#                            score_factors=None):
#     """NMS for multi-class bboxes.
#     Args:
#         multi_bboxes (Tensor): shape (n, #class*5) or (n, 5)
#         multi_scores (Tensor): shape (n, #class), where the 0th column
#             contains scores of the background class, but this will be ignored.
#         score_thr (float): bbox threshold, bboxes with scores lower than it
#             will not be considered.
#         nms_thr (float): NMS IoU threshold
#         max_num (int): if there are more than max_num bboxes after NMS,
#             only top max_num will be kept.
#         score_factors (Tensor): The factors multiplied to scores before
#             applying NMS
#     Returns:
#         tuple: (bboxes, labels), tensors of shape (k, 6) and (k, 1). Labels
#             are 0-based.
#     """
#     num_classes = multi_scores.size(1) - 1
#     # exclude background category
#     if multi_bboxes.shape[1] > 5:
#         bboxes = multi_bboxes.view(multi_scores.size(0), -1, 5)[:, :-1]
#         r2bboxes = multi_r2boxes.view(multi_scores.size(0), -1, 4)[:, :-1]
#         points=multi_points.view(multi_scores.size(0), -1, 8)[:, :-1]
#     else:
#         bboxes = multi_bboxes[:, None].expand(-1, num_classes, 5)
#         r2boxes = multi_r2boxes[:, None].expand(-1, num_classes, 4)
#         points=multi_points[:, None].expand(-1, num_classes, 8)
#     scores = multi_scores[:, :-1]

#     # filter out boxes with low scores
#     valid_mask = scores > score_thr
#     bboxes = bboxes[valid_mask]
#     r2boxes = r2boxes[valid_mask]
#     points = points[valid_mask]
#     if score_factors is not None:
#         scores = scores * score_factors[:, None]
#     scores = scores[valid_mask]
#     labels = valid_mask.nonzero()[:, 1]

#     if bboxes.numel() == 0:
#         bboxes = multi_bboxes.new_zeros((0, 6))
#         r2boxes = multi_bboxes.new_zeros((0, 6))
#         points = multi_bboxes.new_zeros((0, 8))
#         labels = multi_bboxes.new_zeros((0,), dtype=torch.long)
#         return r2boxes, bboxes, points, labels
#     nms_cfg_ = nms_cfg.copy()
#     nms_type = nms_cfg_.pop('type', 'nms')
#     iou_thr = nms_cfg_.pop('iou_thr', 0.1)
#     labels = labels.to(bboxes)
#     keep = ml_nms_rotated(bboxes, scores, labels, iou_thr)
#     bboxes = bboxes[keep]
#     r2boxes = r2boxes[keep]
#     scores = scores[keep]
#     labels = labels[keep]
#     points = points[keep]
#     if keep.size(0) > max_num:
#         _, inds = scores.sort(descending=True)
#         inds = inds[:max_num]
#         r2boxes = r2boxes[inds]
#         bboxes = bboxes[inds]
#         scores = scores[inds]
#         labels = labels[inds]
#         points = points[inds]

#     return  torch.cat([bboxes, scores[:, None]], 1),  torch.cat([r2boxes, scores[:, None],points], 1),labels

def multiclass_nms_rotated_bbox_eqv2(multi_bboxes,multi_points,
                           multi_scores,multi_scores_rf,
                           score_thr,
                           nms_cfg,
                           max_num=-1,
                           score_factors=None):
    """NMS for multi-class bboxes.
    Args:
        multi_bboxes (Tensor): shape (n, #class*5) or (n, 5)
        multi_scores (Tensor): shape (n, #class), where the 0th column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS
    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 6) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    bboxes = multi_bboxes[:, None].expand(-1, num_classes, 5)#增加了四个点坐标
    size_bbox=multi_points.shape
    points = multi_points[:, None].expand(-1, num_classes, size_bbox[1])
    scores = multi_scores[:, :-1]
    scores_rf = multi_scores_rf[:, :-1]
    # filter out boxes with low scores
    valid_mask = scores > score_thr
    bboxes = bboxes[valid_mask]
    points = points[valid_mask]
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = scores[valid_mask]
    scores_rf = scores_rf[valid_mask]
    labels = valid_mask.nonzero()[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 18))
        labels = multi_bboxes.new_zeros((0,), dtype=torch.long)
        return bboxes,  labels
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    iou_thr = nms_cfg_.pop('iou_threshold', 0.1)
    labels = labels.to(bboxes[...,:0:5])
    keep = ml_nms_rotated(bboxes[...,0:5], scores, labels, iou_thr)
    bboxes = bboxes[keep]
    points = points[keep]
    scores = scores[keep]
    labels = labels[keep]
    scores_rf =scores_rf[keep]
    if keep.size(0) > max_num:
        _, inds = scores.sort(descending=True)
        inds = inds[:max_num]
        points = points[inds]
        bboxes = bboxes[inds]
        scores = scores[inds]
        labels = labels[inds]
        scores_rf=scores_rf[inds]
    t4bx, t4by = points[:,0:8:2],points[:,1:8:2]
    t2xmin, _= torch.min(t4bx,1,keepdim=True)
    t2ymin, _= torch.min(t4by,1,keepdim=True)
    t2xmax, _= torch.max(t4bx,1,keepdim=True)
    t2ymax,_= torch.max(t4by,1,keepdim=True)
    r2bboxes=torch.cat((t2xmin,t2ymin,t2xmax,t2ymax),1)       

    return  torch.cat([r2bboxes, scores_rf[:, None], bboxes, points], 1), labels

def multiclass_nms_rotated_bbox(multi_bboxes,multi_points,
                           multi_scores,
                           score_thr,
                           nms_cfg,
                           max_num=-1,
                           score_factors=None):
    """NMS for multi-class bboxes.
    Args:
        multi_bboxes (Tensor): shape (n, #class*5) or (n, 5)
        multi_scores (Tensor): shape (n, #class), where the 0th column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS
    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 6) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    bboxes = multi_bboxes[:, None].expand(-1, num_classes, 5)#增加了四个点坐标
    size_bbox=multi_points.shape
    points = multi_points[:, None].expand(-1, num_classes, size_bbox[1])
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    valid_mask = scores > score_thr
    bboxes = bboxes[valid_mask]
    points = points[valid_mask]
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = scores[valid_mask]
    labels = valid_mask.nonzero()[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 18))
        labels = multi_bboxes.new_zeros((0,), dtype=torch.long)
        return bboxes,  labels
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    iou_thr = nms_cfg_.pop('iou_threshold', 0.1)
    labels = labels.to(bboxes[...,:0:5])
    keep = ml_nms_rotated(bboxes[...,0:5], scores, labels, iou_thr)
    bboxes = bboxes[keep]
    points = points[keep]
    scores = scores[keep]
    labels = labels[keep]

    if keep.size(0) > max_num:
        _, inds = scores.sort(descending=True)
        inds = inds[:max_num]
        points = points[inds]
        bboxes = bboxes[inds]
        scores = scores[inds]
        labels = labels[inds]
    
    t4bx, t4by = points[:,0:8:2],points[:,1:8:2]
    t2xmin, _= torch.min(t4bx,1,keepdim=True)
    t2ymin, _= torch.min(t4by,1,keepdim=True)
    t2xmax, _= torch.max(t4bx,1,keepdim=True)
    t2ymax,_= torch.max(t4by,1,keepdim=True)
    r2bboxes=torch.cat((t2xmin,t2ymin,t2xmax,t2ymax),1)       

    return  torch.cat([r2bboxes, scores[:, None], bboxes, points], 1), labels

# def multiclass_nms_rotated_bbox(multi_bboxes,multi_r2boxes,#uesd in exp4
#                            multi_scores,
#                            score_thr,
#                            nms_cfg,
#                            max_num=-1,
#                            score_factors=None):
#     """NMS for multi-class bboxes.
#     Args:
#         multi_bboxes (Tensor): shape (n, #class*5) or (n, 5)
#         multi_scores (Tensor): shape (n, #class), where the 0th column
#             contains scores of the background class, but this will be ignored.
#         score_thr (float): bbox threshold, bboxes with scores lower than it
#             will not be considered.
#         nms_thr (float): NMS IoU threshold
#         max_num (int): if there are more than max_num bboxes after NMS,
#             only top max_num will be kept.
#         score_factors (Tensor): The factors multiplied to scores before
#             applying NMS
#     Returns:
#         tuple: (bboxes, labels), tensors of shape (k, 6) and (k, 1). Labels
#             are 0-based.
#     """
#     num_classes = multi_scores.size(1) - 1
#     # exclude background category
#     # if multi_bboxes.shape[1] > 5:
#     #     bboxes = multi_bboxes.view(multi_scores.size(0), -1, 5)[:, :-1]
#     #     r2bboxes = multi_r2boxes.view(multi_scores.size(0), -1, 4)[:, :-1]
#     # else:
#     bboxes = multi_bboxes[:, None].expand(-1, num_classes, 13)#增加了四个点坐标
#     r2boxes = multi_r2boxes[:, None].expand(-1, num_classes, 4)
#     scores = multi_scores[:, :-1]

#     # filter out boxes with low scores
#     valid_mask = scores > score_thr
#     bboxes = bboxes[valid_mask]
#     r2boxes = r2boxes[valid_mask]
#     if score_factors is not None:
#         scores = scores * score_factors[:, None]
#     scores = scores[valid_mask]
#     labels = valid_mask.nonzero()[:, 1]

#     if bboxes.numel() == 0:
#         bboxes = multi_bboxes.new_zeros((0, 14))
#         r2boxes = multi_bboxes.new_zeros((0, 5))
#         labels = multi_bboxes.new_zeros((0,), dtype=torch.long)
#         return bboxes, r2boxes,  labels
#     nms_cfg_ = nms_cfg.copy()
#     nms_type = nms_cfg_.pop('type', 'nms')
#     iou_thr = nms_cfg_.pop('iou_thr', 0.1)
#     labels = labels.to(bboxes[...,:0:5])
#     keep = ml_nms_rotated(bboxes[...,0:5], scores, labels, iou_thr)
#     bboxes = bboxes[keep]
#     r2boxes = r2boxes[keep]
#     scores = scores[keep]
#     labels = labels[keep]

#     if keep.size(0) > max_num:
#         _, inds = scores.sort(descending=True)
#         inds = inds[:max_num]
#         r2boxes = r2boxes[inds]
#         bboxes = bboxes[inds]
#         scores = scores[inds]
#         labels = labels[inds]

#     return  torch.cat([bboxes[...,0:5], scores[:, None], bboxes[...,5:]], 1),  torch.cat([r2boxes, scores[:, None]], 1),labels

def multiclass_nms_rotated(multi_bboxes,
                           multi_scores,
                           score_thr,
                           nms_cfg,
                           max_num=-1,
                           score_factors=None):
    """NMS for multi-class bboxes.
    Args:
        multi_bboxes (Tensor): shape (n, #class*5) or (n, 5)
        multi_scores (Tensor): shape (n, #class), where the 0th column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS
    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 6) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 5:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 5)[:, :-1]
    else:
        bboxes = multi_bboxes[:, None].expand(-1, num_classes, 5)
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    valid_mask = scores > score_thr
    bboxes = bboxes[valid_mask]
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = scores[valid_mask]
    labels = valid_mask.nonzero()[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 6))
        labels = multi_bboxes.new_zeros((0,), dtype=torch.long)
        return bboxes, labels
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    iou_thr = nms_cfg_.pop('iou_thr', 0.1)
    labels = labels.to(bboxes)
    keep = ml_nms_rotated(bboxes, scores, labels, iou_thr)
    bboxes = bboxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    if keep.size(0) > max_num:
        _, inds = scores.sort(descending=True)
        inds = inds[:max_num]
        bboxes = bboxes[inds]
        scores = scores[inds]
        labels = labels[inds]

    return torch.cat([bboxes, scores[:, None]], 1), labels
