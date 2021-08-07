import itertools
import logging
import os.path as osp
import tempfile
import cv2
import mmcv
import numpy as np
from mmcv.utils import print_log
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable
import os
import cv2
from mmdet.core import eval_recalls
from .builder import DATASETS
from .custom import CustomDataset
import xml.etree.ElementTree as ET
from DOTA_devkit.ResultMerge_multi_process import mergebypoly
from DOTA_devkit.dota_evaluation_task1 import voc_eval
# from mmdet.core import poly_to_rotated_box_single 
import shutil
@DATASETS.register_module()
class DotaKDataset(CustomDataset):
    # NAME_LABEL_MAP = {
    # 'E-3':       1,
    # 'E-8':       2,
    # 'RC-135VW': 3,
    # 'RC-135S':   4,
    # 'KC-135': 5,
    # 'B-52': 6,
    # 'C-5': 7,
    # 'C-17': 8,
    # 'Il-76': 9,
    # 'A-50':  10,
    # 'Tu-95': 11,
    # 'P-8A':      12,
    # 'KC-10': 13,
    # 'F-22':      14,
    # 'F-35':      15,
    # 'F-16':      16,
    # 'F-15':      17,
    # 'F-18':    18,
    # 'L-39':      19, 
    # 'MiG-29': 20,
    # 'MiG-31': 21,
    # 'Su-35': 22,
    # 'Su-30': 23,
    # 'Su-27': 24,
    # 'Typhoon': 25,
    # 'Su-24': 26,
    # 'Su-34': 27,
    # 'A-10': 28,
    # 'Su-25': 29,
    # 'Tu-22M': 30,
    # 'Yak-130': 31,
    # 'B-1B': 32,
    # 'B-2': 33,
    # 'Tu-160': 34,
    # 'C-130': 35,
    # 'An-12': 36,
    # 'An-24': 37,
    # 'EP-3':      38,
    # 'P-3C':      39,
    # 'E-2':       40,
    # 'C-2': 41,
    # 'V-22': 42,
    # 'RQ-4': 43,
    # 'helicopter': 44,
    # 'other': 45
    # }
    # CLASSES = []
    # for name, label in NAME_LABEL_MAP.items():
    #     CLASSES.append(name)
    CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field',
               'roundabout', 'harbor', 'swimming-pool', 'helicopter')
    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()


        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.data_infos):
            if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def get_subset_by_classes(self):
        """Get img ids that contain any category in class_ids.

        Different from the coco.getImgIds(), this function returns the id if
        the img contains one of the categories rather than all.

        Args:
            class_ids (list[int]): list of category ids

        Return:
            ids (list[int]): integer list of img ids
        """

        ids = set()
        for i, class_id in enumerate(self.cat_ids):
            ids |= set(self.coco.cat_img_map[class_id])
        self.img_ids = list(ids)

        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
        return data_infos


    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann
    
    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def _proposal2json(self, results):
        """Convert proposal results to COCO json style"""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def _det2json(self, results):
        """Convert detection results to COCO json style"""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style"""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label][0:5]#TODO 为了适应测评工作这里限制了result的维度
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else: 
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self.coco.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 6)))#TODO change 4
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 6)) #TODO change 4
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05),
                 eval_dir= None,
        gt_dir='/media/zf/E/Dataset/dota_1024_s2anet2/valGTtxt/'):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float]): IoU threshold used for evaluating
                recalls. If set to a list, the average recall of all IoUs will
                also be computed. Default: 0.5.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = {}
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                cocoDt = cocoGt.loadRes(result_files[metric])
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            iou_type = 'bbox' if metric == 'proposal' else metric
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.params.maxDets = list(proposal_nums)
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                metric_items = [
                    'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000', 'AR_m@1000',
                    'AR_l@1000'
                ]
                for i, item in enumerate(metric_items):
                    val = float(f'{cocoEval.stats[i + 6]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                metric_items = [
                    'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                ]
                for i in range(len(metric_items)):
                    key = f'{metric}_{metric_items[i]}'
                    val = float(f'{cocoEval.stats[i]:.3f}')
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')
        if tmp_dir is not None:
            tmp_dir.cleanup()
        self.evaluate_rbox(results, eval_dir, gt_dir)
        return eval_results


    def evaluate_rbox(self, results, work_dir=None, gt_dir=None):
        dst_raw_path = osp.join(work_dir, 'results_before_nms')
        dst_merge_path = osp.join(work_dir, 'results_after_nms')
        if os.path.exists(dst_raw_path):
            shutil.rmtree(dst_raw_path,True)
        os.makedirs(dst_raw_path)
        if os.path.exists(dst_merge_path):
            shutil.rmtree(dst_merge_path,True)
        os.makedirs(dst_merge_path)

        imagesetfile=osp.join(osp.dirname(gt_dir), 'gt_list.txt')
        generate_file_list(gt_dir,imagesetfile)

        print('Saving results to {}'.format(dst_raw_path))
        # self.result_to_xml(results, os.path.join(work_dir,'pkl2xml'))
        # self.xml2dota_txt(work_dir,dst_raw_path)
        self.result_to_txt(results, os.path.join(dst_raw_path,'result2txtdirect'))

        print('Merge results to {}'.format(dst_merge_path))
        mergebypoly(os.path.join(dst_raw_path,'result2txtdirect'), dst_merge_path)

        print('Start evaluation')
        detpath = osp.join(dst_merge_path, 'Task1_{:s}.txt')
        annopath = osp.join(gt_dir, '{:s}.txt')

        classaps = []
        map = 0
        for classname in self.CLASSES:
            rec, prec, ap = voc_eval(detpath,
                                     annopath,
                                     imagesetfile,
                                     classname,
                                     ovthresh=0.5,
                                     use_07_metric=True)
            map = map + ap
            print(classname, ': ', ap)
            classaps.append(ap)

        map = map / len(self.CLASSES)
        print('map:', map)
        classaps = 100 * np.array(classaps)
        print('classaps: ', classaps)
        # Saving results to disk
        with open(osp.join(work_dir, 'eval_results.txt'), 'w') as f: 
            res_str = 'mAP:' + str(map) + '\n'
            res_str += 'classaps: ' + ' '.join([str(x) for x in classaps])
            f.write(res_str)
        return map

    def result_to_txt(self, results, results_path):
        #这里需要更改 这里不生成xml 直接转换成txt
        img_names = [img_info['filename'] for img_info in self.data_infos]
        assert len(results) == len(img_names), 'len(results) != len(img_names)'
        os.makedirs(results_path)
        for classname in self.CLASSES:
            f_out = open(osp.join(results_path, 'Task1_'+classname + '.txt'), 'w')
            print(classname + '.txt')
            # per result represent one image
            for img_id, result in enumerate(results):
                for class_id, bboxes in enumerate(result):
                    if self.CLASSES[class_id] != classname:
                        continue
                    # elif self.CLASSES[class_id] !='baseball-diamond' :
                    if bboxes.size != 0:
                        for bbox in bboxes:
                            score= bbox[4]
                            rbox_points =bbox[10:]
                            temp_txt = '{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(
                                osp.splitext(img_names[img_id])[0], score, rbox_points[0], rbox_points[1],
                                                                                                                        rbox_points[2], rbox_points[3],
                                                                                                                        rbox_points[4], rbox_points[5],
                                                                                                                        rbox_points[6], rbox_points[7])
                            # rbox= bbox[5:10]
                            # rbox_cv=rotate_rect2cv(rbox)
                            # rbox_points = cv2.boxPoints(rbox_cv)
                            # temp_txt = '{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(
                            #     osp.splitext(img_names[img_id])[0], score, rbox_points[0][0], rbox_points[0][1],
                            #                                                                                             rbox_points[1][0], rbox_points[1][1],
                            #                                                                                             rbox_points[2][0], rbox_points[2][1],
                            #                                                                                             rbox_points[3][0], rbox_points[3][1])
                            f_out.write(temp_txt)
                    # else:
                    #     if bboxes.size != 0:
                    #         for bbox in bboxes:
                    #             [xmin, ymin, xmax,ymax,score]= bbox[:5]
                    #             # angle -= np.pi/2
                    #             cx,cy = (xmin+xmax)/2.0, (ymin+ymax)/2.0
                    #             w,h = xmax-xmin, ymax-ymin
                    #             angle=0
                    #             rbox=[cx, cy, w,h,angle]
                    #             rbox_cv=rotate_rect2cv(rbox)
                    #             rbox_points = cv2.boxPoints(rbox_cv)
                    #             temp_txt = '{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(
                    #                 osp.splitext(img_names[img_id])[0], score, rbox_points[0][0], rbox_points[0][1],
                    #                                                                                                         rbox_points[1][0], rbox_points[1][1],
                    #                                                                                                         rbox_points[2][0], rbox_points[2][1],
                    #                                                                                                         rbox_points[3][0], rbox_points[3][1])
                    #             f_out.write(temp_txt)
            f_out.close()

    def result_to_xml(self, results, dst_path, score_threshold=0.3, nms_threshold=0.1,nms_maxnum=400 ):
        CLASSES = self.CLASSES#dataset.CLASSESself.CLASSES#dataset.CLASSES
        # img_names = [img_info['filename'] for img_info in self.img_infos]
        # assert len(results) == len(img_names), 'len(results) != len(img_names)'
        if not osp.exists(dst_path):
            os.mkdir(dst_path)
        for idx in range(len(self.img_ids)):
            img_id = self.img_ids[idx]
            img_name=self.data_infos[idx]['filename']
            result = results[idx]
            img_boxes=np.zeros((0,7))
            for label in range(len(result)):
                bboxes = result[label]
                #过滤小阈值的目标
                keep= bboxes[:,4]>score_threshold
                bboxes=bboxes[keep]
                #这里开始写转换回来的函数
                if bboxes.shape[0]>0:
                    rotateboxes,cv_rboxes=self.box2rotatexml(bboxes[...,5:],label)
                    #rotate nms
                    keep=nms_rotate_cpu(cv_rboxes,rotateboxes[:,5 ],nms_threshold, nms_maxnum)
                    rotateboxes=rotateboxes[keep]
                    img_boxes= np.vstack((img_boxes, rotateboxes))
            write_rotate_xml(dst_path,img_name,[1024 ,1024,3],0.5,'0.5',img_boxes.reshape((-1,7)),CLASSES)

    def box2rotatexml(self,bboxes,label):
        rotateboxes=[]
        cv_rboxes=[]
        for i in range(bboxes.shape[0]):
            if(bboxes.size != 0):
                # [xmin, ymin, xmax, ymax, score, x1, y1, x2, y2,x3,y3,x4,y4]=bboxes[i,:]
                [cx, cy, w,h,angle,score]=bboxes[i,:]
                # angle -= np.pi/2
                rotatebox=[cx, cy, w,h,angle,score,label]
                rotateboxes.append(rotatebox)
                cv_rboxes.append(rotate_rect2cv_np(rotatebox))
        return np.array(rotateboxes), np.array(cv_rboxes)
   
    def xml2dota_txt(self,dst_path, dst_raw_path):
        CLASSES = self.CLASSES
        NAME_LABEL_MAP={}
        LABEl_NAME_MAP={}
        for index, one_class in  enumerate(CLASSES):
            NAME_LABEL_MAP[one_class]=index+1
            LABEl_NAME_MAP[index+1]=one_class
        file_paths = get_file_paths_recursive(dst_path, '.xml')
        # Task2 # 建立写入句柄
        write_handle_h = {}
        for sub_class in CLASSES:
            if sub_class == 'back_ground':
                continue
            write_handle_h[sub_class] = open(os.path.join(dst_raw_path, 'Task1_%s.txt' % sub_class), 'a+')
        # 循环写入
        for count, xml_path in enumerate(file_paths):
            img_size, gsd, imagesource, gtbox_label, extra =read_rotate_xml(xml_path,NAME_LABEL_MAP)
            for i, rbox in enumerate(gtbox_label):
                rbox_cv=rotate_rect2cv(rbox)
                rect_box = cv2.boxPoints(rbox_cv)
                xmin,ymin,xmax,ymax=np.min(rect_box[:,0]),np.min(rect_box[:,1]),np.max(rect_box[:,0]),np.max(rect_box[:,1])
                # xmin,ymin,xmax,ymax,score=rbox[0:5]
                # command = '%s %.3f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n' % (os.path.splitext(os.path.split(xml_path)[1])[0],
                #                                             score,
                #                                             xmin, ymin, xmax, ymin,
                #                                             xmax, ymax, xmin, ymax,)
                command = '%s %.3f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n' % (os.path.splitext(os.path.split(xml_path)[1])[0],
                                                                np.float(extra[i]),
                                                                rect_box[0][0], rect_box[0][1], rect_box[1][0], rect_box[1][1],
                                                                rect_box[2][0], rect_box[2][1], rect_box[3][0], rect_box[3][1])
                write_handle_h[LABEl_NAME_MAP[rbox[5]]].write(command)
        #关闭句柄
        for sub_class in CLASSES:
            if sub_class == 'back_ground':
                continue
            write_handle_h[sub_class].close()


def get_file_paths_recursive(folder=None, file_exts=None):
    """ Get the absolute path of all files in given folder recursively
    :param folder:
    :param file_ext:
    :return:
    """
    file_list = []
    if folder is None:
        return file_list
    file_list = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith(file_exts)]
    return file_list

def nms_rotate_cpu(boxes, scores, iou_threshold, max_output_size):
    keep = []#保留框的结果集合
    order = scores.argsort()[::-1]#对检测结果得分进行降序排序
    num = boxes.shape[0]#获取检测框的个数
    suppressed = np.zeros((num), dtype=np.int)
    for _i in range(num):
        if len(keep) >= max_output_size:#若当前保留框集合中的个数大于max_output_size时，直接返回
            break
        i = order[_i]
        if suppressed[i] == 1:#对于抑制的检测框直接跳过
            continue
        keep.append(i)#保留当前框的索引
        # (midx,midy),(width,height), angle)
        r1 = ((boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), boxes[i, 4])
        #        r1 = ((boxes[i, 1], boxes[i, 0]), (boxes[i, 3], boxes[i, 2]), boxes[i, 4]) #根据box信息组合成opencv中的旋转bbox
        #        print("r1:{}".format(r1))
        area_r1 = boxes[i, 2] * boxes[i, 3]#计算当前检测框的面积
        for _j in range(_i + 1, num):#对剩余的而进行遍历
            j = order[_j]
            if suppressed[i] == 1:
                continue
            r2 = ((boxes[j, 0], boxes[j, 1]), (boxes[j, 2], boxes[j, 3]), boxes[j, 4])
            area_r2 = boxes[j, 2] * boxes[j, 3]
            inter = 0.0
            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]#求两个旋转矩形的交集，并返回相交的点集合
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)#求点集的凸边形
                int_area = cv2.contourArea(order_pts)#计算当前点集合组成的凸边形的面积
                inter = int_area * 1.0 / (area_r1 + area_r2 - int_area + 0.0000001)
            if inter >= iou_threshold:#对大于设定阈值的检测框进行滤除
                suppressed[j] = 1
    return np.array(keep, np.int64)

def rotate_rect2cv_np(rotatebox):
    #此程序将rotatexml中旋转矩形的表示，转换为cv2的RotateRect
    [x_center,y_center,w,h,angle]=rotatebox[0:5]
    angle_mod=angle*180/np.pi%180
    if angle_mod>=0 and angle_mod<90:
        [cv_w,cv_h,cv_angle]=[h,w,angle_mod-90]
    if angle_mod>=90 and angle_mod<180:
        [cv_w,cv_h,cv_angle]=[w,h,angle_mod-180]
    cvbox=np.array([ x_center,y_center,cv_w,cv_h,cv_angle ])
    return cvbox

def rotate_rect2cv(rotatebox):
    #此程序将rotatexml中旋转矩形的表示，转换为cv2的RotateRect
    [x_center,y_center,w,h,angle]=rotatebox[0:5]
    angle_mod=angle*180/np.pi%180
    if angle_mod>=0 and angle_mod<90:
        [cv_w,cv_h,cv_angle]=[h,w,angle_mod-90]
    if angle_mod>=90 and angle_mod<180:
        [cv_w,cv_h,cv_angle]=[w,h,angle_mod-180]
    return ((x_center,y_center),(cv_w,cv_h),cv_angle)

def read_rotate_xml(xml_path,NAME_LABEL_MAP):
    """
    :param xml_path: the path of voc xml
    :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 5],
            and has [xmin, ymin, xmax, ymax, label] in a per row
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = None
    img_height = None
    box_list = []
    extra=[]
    for child_of_root in root:
        if child_of_root.tag == 'folder':#读取gsd之前把它赋予到了folder字段
            try:
                gsd = float(child_of_root.text)
            except:
                gsd =0
        if child_of_root.tag == 'size':
            for child_item in child_of_root:
                if child_item.tag == 'width':
                    img_width = int(child_item.text)
                if child_item.tag == 'height':
                    img_height = int(child_item.text)
                if child_item.tag == 'depth':
                    img_depth = 3#int(child_item.text)
        if child_of_root.tag == 'source':
            for child_item in child_of_root:
                if child_item.tag == 'database':
                    imagesource=child_item.text
        if child_of_root.tag == 'object':
            label = None
            for child_item in child_of_root:
                if child_item.tag == 'name':
                    #TODO change
                    #                    label_name=child_item.text.replace('plane','other').replace('\ufeffB-1B','B-1B').replace('F-31','F-35').replace('L-39','L-159')
                    label_name=child_item.text.replace('\ufeff','')#.replace("其它","其他")#.replace('plane','bridge')#.replace('尼米兹级','航母').replace('圣安东尼奥','圣安东尼奥级').replace('圣安东尼奥级级','圣安东尼奥级')#.replace('塔瓦拉级','黄蜂级')
                    label =NAME_LABEL_MAP[label_name]#float(child_item.text) #训练VOC用NAME_LABEL_MAP[child_item.text]#因为用自己的这边的ID是编号  训练卫星数据用1
                if child_item.tag == 'difficult':
                    difficult=int(child_item.text)
                if child_item.tag == 'extra':
                    extra.append(child_item.text)
                if child_item.tag == 'robndbox':
                    tmp_box = [0, 0, 0, 0, 0,0,0]
                    for node in child_item:
                        if node.tag == 'cx':
                            tmp_box[0] = float(node.text)
                        if node.tag == 'cy':
                            tmp_box[1] = float(node.text)
                        if node.tag == 'w':
                            tmp_box[2] = float(node.text)
                        if node.tag == 'h':
                            tmp_box[3] = float(node.text)
                        if node.tag == 'angle':
                            tmp_box[4] = float(node.text)
                    assert label is not None, 'label is none, error'
                    tmp_box[5]=label
                    tmp_box[6]=difficult
                    box_list.append(tmp_box)
    #    gtbox_label = np.array(box_list, dtype=np.int32)
    img_size=[img_height,img_width,img_depth]
    return img_size,gsd,imagesource,box_list,extra

##添加写出为xml函数
def write_rotate_xml(output_floder,img_name,size,gsd,imagesource,gtbox_label,CLASSES):#size,gsd,imagesource#将检测结果表示为中科星图比赛格式的程序,这里用folder字段记录gsd
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

def generate_file_list(img_dir,output_txt,file_ext='.txt'):
    #读取原图路径
    # img_dir=os.path.split(img_dir)[0]
    imgs_path = get_file_paths_recursive(img_dir, file_ext)
    f = open(output_txt, "w",encoding='utf-8')
    for num,img_path in enumerate(imgs_path,0):
        obj='{}\n'.format(os.path.splitext(os.path.split(img_path)[1])[0])
        f.write(obj)
    f.close()
    print('Generate {} down!'.format(output_txt))