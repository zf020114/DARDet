from ..builder import DETECTORS
from .single_stage import SingleStageDetector
import torch
import numpy as np
import mmcv 
import os
from mmdet.core import bbox2result,multiclass_nms_rotated,gt_mask_bp_obbs_list
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import *
import copy
import itertools
from skimage import measure

def write_rotate_xml(output_floder,img_name,size,gsd,imagesource,gtbox_label,CLASSES):#size,gsd,imagesource#将检测结果表示为中科星图比赛格式的程序,这里用folder字段记录gsd
    ##添加写出为xml函数
    voc_headstr = """\
    <annotation>
        <folder>{}</folder>
        <filename>{}</filename>
        <path>{}</path>
        <source>
            <database>{}</database>
        </source>
        <size>
            <width>{}</width>
            <height>{}</height>
            <depth>{}</depth>
        </size>
        <segmented>0</segmented>
        """
    voc_rotate_objstr = """\
    <object>
        <name>{}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>{}</difficult>
        <robndbox>
            <cx>{}</cx>
            <cy>{}</cy>
            <w>{}</w>
            <h>{}</h>
            <angle>{}</angle>
        </robndbox>
        <extra>{:.2f}</extra>
    </object>
    """
    voc_tailstr = '''\
        </annotation>
        '''
    [floder,name]=os.path.split(img_name)
    # filename=name.replace('.jpg','.xml')
    filename=os.path.join(floder,os.path.splitext(name)[0]+'.xml')
    foldername=os.path.split(img_name)[0]
    head=voc_headstr.format(gsd,name,foldername,imagesource,size[1],size[0],size[2])
    rotate_xml_name=os.path.join(output_floder,os.path.split(filename)[1])
    f = open(rotate_xml_name, "w",encoding='utf-8')
    f.write(head)
    for i,box in enumerate (gtbox_label):
        obj=voc_rotate_objstr.format(CLASSES[int(box[6])],0,box[0],box[1],box[2],box[3],box[4],box[5])
        f.write(obj)
    f.write(voc_tailstr)
    f.close()

@DETECTORS.register_module()
class VFNet(SingleStageDetector):
    """Implementation of `VarifocalNet
    (VFNet).<https://arxiv.org/abs/2008.13367>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(VFNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                    test_cfg, pretrained)

    def rbox2result(self, bboxes, labels, num_classes):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor | np.ndarray): shape (n, 5)
            labels (torch.Tensor | np.ndarray): shape (n, )
            num_classes (int): class number, including background class

        Returns:
            list(ndarray): bbox results of each class
        """
        if bboxes.shape[0] == 0:
            return [np.zeros((0, 9), dtype=np.float32) for i in range(num_classes)]#TODOinsert
        else:
            if isinstance(bboxes, torch.Tensor):
                bboxes = bboxes.detach().cpu().numpy()#dets,rboxes[keep],scores_k[keep]
                labels = labels.detach().cpu().numpy()

            return [bboxes[labels == i, :] for i in range(num_classes)]
        
    def imshow_gpu_tensor(self, tensor):#调试中显示表标签图
        from PIL import Image
        from torchvision import transforms
        device=tensor[0].device
        mean= torch.tensor([123.675, 116.28, 103.53])
        std= torch.tensor([58.395, 57.12, 57.375])
        mean=mean.to(device)
        std=std.to(device)
        tensor = (tensor[0].squeeze() * std[:,None,None]) + mean[:,None,None]
        tensor=tensor[0:1]
        if len(tensor.shape)==4:
            image = tensor.permute(0,2, 3,1).cpu().clone().numpy()
        else:
            image = tensor.permute(1, 2,0).cpu().clone().numpy()
        image = image.astype(np.uint8).squeeze()
        image = transforms.ToPILImage()(image)
        image = image.resize((256, 256),Image.ANTIALIAS)
        image.show(image)
            # image.save('./img.jpg')
    
    def load_semantic_map_from_mask(self, gt_boxes, gt_masks, gt_labels,feature_shape):
        pad_shape=feature_shape[-2:]
        gt_areas=gt_masks.areas
        # heatmap = gt_boxes.new_zeros((heatmap_channel, output_h, output_w))
        gt_sem_map = gt_boxes.new_zeros((self.bbox_head.num_classes, int(pad_shape[0] ), int(pad_shape[1] )))
        gt_sem_weights = gt_boxes.new_zeros((self.bbox_head.num_classes, int(pad_shape[0] ), int(pad_shape[1] )))
        box_masks=gt_masks.rescale(1/8).masks
        indexs = torch.sort(gt_areas)
        for ind in indexs[::-1]:
            box_mask=box_masks[ind]
            gt_sem_map[gt_labels[ind]][box_mask > 0] = 1
            gt_sem_weights[gt_labels[ind]][box_mask > 0] = np.min([1 / (gt_areas[ind]+0.000001),1])
        return gt_sem_map, gt_sem_weights
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None
                      ):

        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        # self.imshow_gpu_tensor(img)
    
        gt_mask_arrays=[]
        for i in  gt_masks:
            gt_mask_array=np.array(i.masks).squeeze()
            gt_mask_array = gt_mask_array.astype(float)  # numpy强制类型转换
            gt_mask_array=gt_bboxes[0].new_tensor(gt_mask_array)
            gt_mask_arrays.append(gt_mask_array)
        gt_masks=gt_mask_arrays
        
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore,gt_masks)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return bbox_list
        bbox_results = [
            self.rbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes,  det_labels  in bbox_list
        ]
        return bbox_results

    def result2rotatexml(self,bboxes,labels,thr):
        if bboxes.shape[0]>0:
            index=bboxes[:,5]>thr
            bboxes=bboxes[index]
            labels=labels[index]

        rotateboxes=[]
        for i in range(bboxes.shape[0]):
            if(bboxes.size != 0):
                [cx, cy, w, h,angle, score]=bboxes[i,:]
                # angle -= np.pi/2
                rotateboxes.append([cx, cy, w,h,angle,score,labels[i],score])
                    # rotateboxes=np.vstack((cx, cy, w,h,Angle,result[:,7],result[:,4])).T
                    # rotateboxes.append([bbox[0],bbox[1],bbox[3],bbox[2],bbox[4]+np.pi/2,bbox[5],class_id])
        return np.array(rotateboxes)

    def box2rotatexml(self,bboxes,labels):
        index=bboxes[:,4]>0.3
        bboxes=bboxes[index]
        labels=labels[index]
        rotateboxes=[]
        for i in range(bboxes.shape[0]):
            if(bboxes.size != 0):
                [xmin, ymin, xmax, ymax, score, x1, y1, x2, y2,x3,y3,x4,y4]=bboxes[i,:]
                cx,cy = (x1 + x2+x3+x4)/4,(y1+y2+y3+y4)/4
                det=[x3-x1, y3-y1]
                h=np.linalg.norm(det)
                w=np.linalg.norm([x4-x2, y4-y2])
                if det[0]==0:
                    if det[1]>0:
                        Angle = np.pi/2
                    else:
                        Angle = -np.pi/2
                elif det[0]<0:
                    Angle = np.arctan(det[1]/det[0])+np.pi/2
                else:
                    Angle = np.arctan(det[1]/det[0])-np.pi/2
                rotateboxes.append([cx, cy, w,h,Angle,score,labels[i],score])
                    # rotateboxes=np.vstack((cx, cy, w,h,Angle,result[:,7],result[:,4])).T
                    # rotateboxes.append([bbox[0],bbox[1],bbox[3],bbox[2],bbox[4]+np.pi/2,bbox[5],class_id])
        return np.array(rotateboxes)

    def drow_points(self,img,points,score_thr=0.3):
        if points.shape[0]>0:
            index=points[:,0]>score_thr
            points=points[index]
            for i in range(points.shape[0]):
                rbox=points[i][1:]                                                       #BRG
                cv2.circle(img, (int(rbox[0]), int(rbox[1])), 5, (0,255,0), -1)
                cv2.circle(img, (int(rbox[2]), int(rbox[3])), 4, (0,0,255), -1)
                cv2.circle(img, (int(rbox[4]), int(rbox[5])), 3, (255,0,0), -1)
                cv2.circle(img, (int(rbox[6]), int(rbox[7])), 2, (255,0,255), -1)
                # if rbox.shape[1]>7:
                #     cv2.circle(img, (int(rbox[8]), int(rbox[9])), 4, (0,0,0), -1)
                #     for j in range(9):
                #         cv2.circle(img, (int(rbox[10+j*2]), int(rbox[11+j*2])), 3, (0,255,255), -1)
                
        return img

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color='green',
                    text_color='green',
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > score_thr)[0]
            np.random.seed(42)
            color_masks = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
            for i in inds:
                i = int(i)
                color_mask = color_masks[labels[i]]
                mask = segms[i]
                img[mask] = img[mask] * 0.5 + color_mask * 0.5
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
            #这里开始让他一开始就显示rotatexml
        if bbox_result[0].shape[1]>5:
            # rotateboxes=self.box2rotatexml(bboxes,labels)
            rbox=np.hstack((bboxes[...,5:10],bboxes[...,4:5]))
            rotateboxes=self.result2rotatexml(rbox,labels,score_thr)
            write_rotate_xml(os.path.dirname(out_file),out_file,[1024 ,1024,3],0.5,'0.5',rotateboxes.reshape((-1,8)),self.CLASSES)

        showboxs=np.hstack((bboxes[...,4:5],bboxes[...,10:]))
        img=self.drow_points(img,showboxs,score_thr=score_thr)

        #draw bounding boxes
        mmcv.imshow_det_bboxes(
            img,
            bboxes[:,0:5],
            labels,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            thickness=thickness,
            font_scale=font_scale,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                            'result image will be returned')
            return img
