import torch
import numpy as np
from mmdet.core import bbox2result
from mmdet.models.builder import DETECTORS
from ...core.utils import flip_tensor
from .single_stage import SingleStageDetector
import mmcv 
import os, cv2
import matplotlib.pyplot as plt
from shapely.geometry import *
import copy
import itertools
# from mmdet.core import bbox2result,multiclass_nms_rotated

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
class TTFNet(SingleStageDetector):
    """Implementation of CenterNet(Objects as Points)

    <https://arxiv.org/abs/1904.07850>.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TTFNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained, init_cfg)
        self.mean=torch.tensor([123.675, 116.28, 103.53])
        self.mean=self.mean[None,None,None,:]
        self.std=torch.tensor([58.395, 57.12, 57.375])
        self.std=self.std[None,None,None,:]

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

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)

        # self.imshow_gpu_tensor(img)
        return x

    def merge_aug_results(self, aug_results, with_nms):
        """Merge augmented detection bboxes and score.

        Args:
            aug_results (list[list[Tensor]]): Det_bboxes and det_labels of each
                image.
            with_nms (bool): If True, do nms before return boxes.

        Returns:
            tuple: (out_bboxes, out_labels)
        """
        recovered_bboxes, aug_labels = [], []
        for single_result in aug_results:
            recovered_bboxes.append(single_result[0][0])
            aug_labels.append(single_result[0][1])

        bboxes = torch.cat(recovered_bboxes, dim=0).contiguous()
        labels = torch.cat(aug_labels).contiguous()
        if with_nms:
            out_bboxes, out_labels = self.bbox_head._bboxes_nms(
                bboxes, labels, self.bbox_head.test_cfg)
        else:
            out_bboxes, out_labels = bboxes, labels

        return out_bboxes, out_labels

    def aug_test(self, imgs, img_metas, rescale=True):
        """Augment testing of CenterNet. Aug test must have flipped image pair,
        and unlike CornerNet, it will perform an averaging operation on the
        feature map instead of detecting bbox.

        Args:
            imgs (list[Tensor]): Augmented images.
            img_metas (list[list[dict]]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: True.

        Note:
            ``imgs`` must including flipped image pairs.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        img_inds = list(range(len(imgs)))
        assert img_metas[0][0]['flip'] + img_metas[1][0]['flip'], (
            'aug test must have flipped image pair')
        aug_results = []
        for ind, flip_ind in zip(img_inds[0::2], img_inds[1::2]):
            flip_direction = img_metas[flip_ind][0]['flip_direction']
            img_pair = torch.cat([imgs[ind], imgs[flip_ind]])
            x = self.extract_feat(img_pair)
            center_heatmap_preds, wh_preds, offset_preds = self.bbox_head(x)
            assert len(center_heatmap_preds) == len(wh_preds) == len(
                offset_preds) == 1

            # Feature map averaging
            center_heatmap_preds[0] = (
                center_heatmap_preds[0][0:1] +
                flip_tensor(center_heatmap_preds[0][1:2], flip_direction)) / 2
            wh_preds[0] = (wh_preds[0][0:1] +
                           flip_tensor(wh_preds[0][1:2], flip_direction)) / 2

            bbox_list = self.bbox_head.get_bboxes(
                center_heatmap_preds,
                wh_preds, [offset_preds[0][0:1]],
                img_metas[ind],
                rescale=rescale,
                with_nms=False)
            aug_results.append(bbox_list)

        nms_cfg = self.bbox_head.test_cfg.get('nms_cfg', None)
        if nms_cfg is None:
            with_nms = False
        else:
            with_nms = True
        bbox_list = [self.merge_aug_results(aug_results, with_nms)]
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

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
                # points = np.hstack((bboxes[...,0:4],bboxes[...,5:9]))
                # r2bboxes=np.concatenate((points[...,0::2].min(1)[:, np.newaxis],
                #                                                         points[...,1::2].min(1)[:, np.newaxis], 
                #                                                         points[...,0::2].max(1)[:, np.newaxis],
                #                                                         points[...,1::2].max(1)[:, np.newaxis] ,bboxes[...,4:5]),axis=1)
                # bboxes = np.hstack((r2bboxes,points))
                labels = labels.detach().cpu().numpy()

            return [bboxes[labels == i, :] for i in range(num_classes)]
    # def rbox2result(self, bboxes, labels, num_classes):#这是实验八的处理方式
    #     """Convert detection results to a list of numpy arrays.

    #     Args:
    #         bboxes (torch.Tensor | np.ndarray): shape (n, 5)
    #         labels (torch.Tensor | np.ndarray): shape (n, )
    #         num_classes (int): class number, including background class

    #     Returns:
    #         list(ndarray): bbox results of each class
    #     """
    #     if bboxes.shape[0] == 0:
    #         return [np.zeros((0, 9), dtype=np.float32) for i in range(num_classes)]#TODOinsert
    #     else:
    #         if isinstance(bboxes, torch.Tensor):
    #             bboxes = bboxes.detach().cpu().numpy()#dets,rboxes[keep],scores_k[keep]
    #             points = np.hstack((bboxes[...,0:4],bboxes[...,5:9]))
    #             r2bboxes=np.concatenate((points[...,0::2].min(1)[:, np.newaxis],
    #                                                                     points[...,1::2].min(1)[:, np.newaxis], 
    #                                                                     points[...,0::2].max(1)[:, np.newaxis],
    #                                                                     points[...,1::2].max(1)[:, np.newaxis] ,bboxes[...,4:5]),axis=1)
    #             labels = labels.detach().cpu().numpy()
    #             bboxes = np.hstack((r2bboxes,points))
    #         return [bboxes[labels == i, :] for i in range(num_classes)]

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
        #BGR2RGB
        # device=img.device
        # img=torch.cat([img[...,2],img[...,1],img[...,0]],dim=-1)
        # self.mean=self.mean.to(device)
        # self.std=self.std.to(device)
        # img=(img-self.mean)/self.std

        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return bbox_list
        # bbox_results = [
        #     bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
        #     for det_bboxes, det_labels in bbox_list
        # ]
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