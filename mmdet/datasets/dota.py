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
from .dota_k import DotaKDataset
import shutil
@DATASETS.register_module()

class DotaDataset(DotaKDataset):
    NAME_LABEL_MAP={
    'Roundabout':1,
    'Intersection':2,
    'Bridge':3,
    'Tennis-Court':4, 
    'Basketball-Court':5,
    'Football-Field':6,
    'Baseball-Field' :7,
    'ship':8,
    'vehicle':9,
    'plane':10,
    }
    CLASSES = []
    for name, label in NAME_LABEL_MAP.items():
        CLASSES.append(name)
    