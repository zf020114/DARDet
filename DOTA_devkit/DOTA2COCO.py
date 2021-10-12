import dota_utils as util
import os
import cv2
import json
from PIL import Image

wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']

wordname_16 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']

wordname_text = ['text']
# NAME_LABEL_MAP = {
#     'Liquid_Cargo_Ship':1, 
#     'Passenger_Ship':2,
#     'Dry_Cargo_Ship':3,
#     'Motorboat':4, 
#     'Fishing_Boat':5,
#     'Engineering_Ship':6,
#     'Warship':7,
#     'Tugboat':8,
#     'other-ship':9,
#     'Cargo_Truck':10, 
#     'Small_Car':11, 
#     'Dump_Truck':12, 
#     'Tractor':13,
#     'Bus':14, 
#     'Trailer':15,  
#     'Truck_Tractor':16, 
#     'Van':17, 
#     'Excavator':18, 
#     'other-vehicle':19,
#     'Boeing737': 20,
#     'A220': 21,
#     'Boeing787': 22,
#     'Boeing777':23,
#     'A350': 24,
#     'A330': 25,
#     'A321': 26,
#     'Boeing747': 27,
#     'ARJ21': 28,
#     'other-airplane': 29,
#     'Intersection':30, 
#     'Bridge':31,
#     'Tennis_Court':32, 
#     'Basketball_Court':33,
#     'Football_Field':34,
#     'Baseball_Field':35 }
# NAME_LABEL_MAP = {
#         'E-3':       1,
#         'E-8':       2,
#         'RC-135V/W': 3,
#         'RC-135S':   4,
#         'E-2':       5,
#         'EP-3':      6,
#         'P-3C':      7,
#         'A-50':      8,
#         'P-8A':      9,
#         'F-22':      10,
#         'F-35':      11,
#         'F-16':      12,
#         'F-15':      13,
#         'F/A-18':    14,
#         'F/A-18E/F': 15,
#         'L-39': 16,
#         'MiG-29': 17,
#         'MiG-31': 18,
#         'Su-35': 19,
#         'Su-30': 20,
#         'Su-27': 21,
#         'Typhoon': 22,
#         'Su-24': 23,
#         'Su-34': 24,
#         'A-10': 25,
#         'Su-25': 26,
#         'B-52': 27,
#         'B-1B': 28,
#         'B-2': 29,
#         'Tu-95': 30,
#         'Tu-160': 31,
#         'KC-135': 32,
#         'KC-10': 33,
#         'C-130': 34,
#         'C-5': 35,
#         'C-2': 36,
#         'C-17': 37,
#         'Il-76': 38,
#         'V-22': 39,
#         'Tu-22M': 40,
#         'An-12': 41,
#         'An-24': 42,
#         'Yak-130': 43,
#         'RQ-4': 44,
#         'helicopter': 45,
#         'other': 46
#         }
# NAME_LABEL_MAP = {
#     'E-3':       1,
#     'E-8':       2,
#     'RC-135VW': 3,
#     'RC-135S':   4,
#     'KC-135': 5,
#     'B-52': 6,
#     'C-5': 7,
#     'C-17': 8,
#     'Il-76': 9,
#     'A-50':  10,
#     'Tu-95': 11,
#     'P-8A':      12,
#     'KC-10': 13,
#     'F-22':      14,
#     'F-35':      15,
#     'F-16':      16,
#     'F-15':      17,
#     'F-18':    18,
#     'L-39':      19,
#     'MiG-29': 20,
#     'MiG-31': 21,
#     'Su-35': 22,
#     'Su-30': 23,
#     'Su-27': 24,
#     'Typhoon': 25,
#     'Su-24': 26,
#     'Su-34': 27,
#     'A-10': 28,
#     'Su-25': 29,
#     'Tu-22M': 30,
#     'Yak-130': 31,
#     'B-1B': 32,
#     'B-2': 33,
#     'Tu-160': 34,
#     'C-130': 35,
#     'An-12': 36,
#     'An-24': 37,
#     'EP-3':      38,
#     'P-3C':      39,
#     'E-2':       40,
#     'C-2': 41,
#     'V-22': 42,
#     'RQ-4': 43,
#     'helicopter': 44,
#     'other': 45
#     }
# wordname_15=[]
# for name, label in NAME_LABEL_MAP.items():
#     wordname_15.append(name)

def DOTA2COCOTrain(srcpath, destfile, cls_names, difficult='2', ext='.png'):
    # set difficult to filter '2', '1', or do not filter, set '-1'

    imageparent = os.path.join(srcpath, 'train2017')
    labelparent = os.path.join(srcpath, 'train2017labelTxt')

    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = util.GetFileFromThisRootDir(labelparent)
        for file in filenames:
            basename = util.custombasename(file)
            # image_id = int(basename[1:])

            imagepath = os.path.join(imageparent, basename + ext)
            img = cv2.imread(imagepath)
            height, width, c = img.shape
            # height, width, c =1024,1024,3
            
            single_image = {}
            single_image['file_name'] = basename + ext
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            # annotations
            objects = util.parse_dota_poly2(file)
            for obj in objects:
                if obj['difficult'] == difficult:
                    print('difficult: ', difficult)
                    continue
                single_obj = {}
                # single_obj['area'] = obj['area']
                single_obj['category_id'] = cls_names.index(obj['name']) + 1
                single_obj['segmentation'] = []
                single_obj['segmentation'].append(obj['poly'])
                single_obj['iscrowd'] = 0
                xmin, ymin, xmax, ymax = min(obj['poly'][0::2]), min(obj['poly'][1::2]), \
                                         max(obj['poly'][0::2]), max(obj['poly'][1::2])

                width, height = xmax - xmin, ymax - ymin
                single_obj['bbox'] = xmin, ymin, width, height
                #modified
                single_obj['area'] = width*height
                single_obj['image_id'] = image_id
                data_dict['annotations'].append(single_obj)
                single_obj['id'] = inst_count
                inst_count = inst_count + 1
            image_id = image_id + 1
        json.dump(data_dict, f_out)

def DOTA2COCOval(srcpath, destfile, cls_names, difficult='2', ext='.png'):
    # set difficult to filter '2', '1', or do not filter, set '-1'

    imageparent = os.path.join(srcpath, 'val2017')
    labelparent = os.path.join(srcpath, 'val2017labelTxt')

    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = util.GetFileFromThisRootDir(labelparent)
        for file in filenames:
            basename = util.custombasename(file)
            # image_id = int(basename[1:])

            imagepath = os.path.join(imageparent, basename + ext)
            # img = cv2.imread(imagepath)
            # height, width, c = img.shape
            height, width, c =1024,1024,3
            
            single_image = {}
            single_image['file_name'] = basename + ext
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            # annotations
            objects = util.parse_dota_poly2(file)
            for obj in objects:
                if obj['difficult'] == difficult:
                    print('difficult: ', difficult)
                    continue
                single_obj = {}
                # single_obj['area'] = obj['area']
                single_obj['category_id'] = cls_names.index(obj['name']) + 1
                single_obj['segmentation'] = []
                single_obj['segmentation'].append(obj['poly'])
                single_obj['iscrowd'] = 0
                xmin, ymin, xmax, ymax = min(obj['poly'][0::2]), min(obj['poly'][1::2]), \
                                         max(obj['poly'][0::2]), max(obj['poly'][1::2])

                width, height = xmax - xmin, ymax - ymin
                single_obj['bbox'] = xmin, ymin, width, height
                #modified
                single_obj['area'] = width*height
                single_obj['image_id'] = image_id
                data_dict['annotations'].append(single_obj)
                single_obj['id'] = inst_count
                inst_count = inst_count + 1
            image_id = image_id + 1
        json.dump(data_dict, f_out)
        
def DOTA2COCOTest(srcpath, destfile, cls_names, ext='.png'):
    imageparent = os.path.join(srcpath, 'images')
    data_dict = {}

    data_dict['images'] = []
    data_dict['categories'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = util.GetFileFromThisRootDir(imageparent)
        for file in filenames:
            basename = util.custombasename(file)
            imagepath = os.path.join(imageparent, basename + ext)
            img = Image.open(imagepath)
            height = img.height
            width = img.width

            single_image = {}
            single_image['file_name'] = basename + ext
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            image_id = image_id + 1
        json.dump(data_dict, f_out)

if __name__ == '__main__':

    # DOTA2COCOTrain(r'/home/dj/code/mmdetection_DOTA/data/dota1_1024_v2/trainval1024',
    #                r'/home/dj/code/mmdetection_DOTA/data/dota1_1024_v2/trainval1024/DOTA_trainval1024.json',
    #                wordname_15)
    # DOTA2COCOTrain(r'/home/dj/code/mmdetection_DOTA/data/dota1_1024_v2/trainval1024_ms',
    #                r'/home/dj/code/mmdetection_DOTA/data/dota1_1024_v2/trainval1024_ms/DOTA_trainval1024_ms.json',
    #                wordname_15)
    # DOTA2COCOTest(r'/workfs/jmhan/dota15_1024_ms/test1024',
    #               r'/workfs/jmhan/dota15_1024_ms/test1024/DOTA_test1024.json',
    #               wordname_16)
    # DOTA2COCOTest(r'/workfs/jmhan/dota15_1024_ms/test1024_ms',
    #               r'/workfs/jmhan/dota15_1024_ms/test1024_ms/DOTA_test1024_ms.json',
    #               wordname_16)
    # DOTA2COCOTrain(r'data/MSRA500_DOTA/train',
    #                r'data/MSRA500_DOTA/train/train.json',
    #                wordname_text, ext='.JPG')

    # DOTA2COCOTest(r'data/MSRA500_DOTA/test',
    #               r'data/MSRA500_DOTA/test/test.json',
    #               wordname_text, ext='.JPG')

    # DOTA2COCOTrain(r'data/RCTW/train',
    #                r'data/RCTW/train/train.json',
    #                wordname_text, ext='.jpg')
    #
    # DOTA2COCOTest(r'data/RCTW/test',
    #                r'data/RCTW/test/test.json',
    #                wordname_text, ext='.jpg')
    dstpath='/media/zf/E/Dataset/dota1-split-1024'
    DOTA2COCOTrain(dstpath,os.path.join(
        dstpath, 'annotations', 'DOTA_trainval1024.json'), wordname_15, difficult='2', ext='.jpg')

    DOTA2COCOval(dstpath, os.path.join(
        dstpath, 'annotations', 'DOTA_test1024.json'), wordname_15,difficult='2', ext='.jpg')
