import utils as util
import os
import ImgSplit_multi_process
import SplitOnlyImage_multi_process
import shutil
from multiprocessing import Pool
from DOTA2COCO import DOTA2COCOTest, DOTA2COCOTrain
import argparse



# wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
#                'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']

NAME_LABEL_MAP = {
        'E-3':       1,
        'E-8':       2,
        'RC-135V/W': 3,
        'RC-135S':   4,
        'E-2':       5,
        'EP-3':      6,
        'P-3C':      7,
        'A-50':      8,
        'P-8A':      9,
        'F-22':      10,
        'F-35':      11,
        'F-16':      12,
        'F-15':      13,
        'F/A-18':    14,
        'F/A-18E/F': 15,
        'L-39': 16,
        'MiG-29': 17,
        'MiG-31': 18,
        'Su-35': 19,
        'Su-30': 20,
        'Su-27': 21,
        'Typhoon': 22,
        'Su-24': 23,
        'Su-34': 24,
        'A-10': 25,
        'Su-25': 26,
        'B-52': 27,
        'B-1B': 28,
        'B-2': 29,
        'Tu-95': 30,
        'Tu-160': 31,
        'KC-135': 32,
        'KC-10': 33,
        'C-130': 34,
        'C-5': 35,
        'C-2': 36,
        'C-17': 37,
        'Il-76': 38,
        'V-22': 39,
        'Tu-22M': 40,
        'An-12': 41,
        'An-24': 42,
        'Yak-130': 43,
        'RQ-4': 44,
        'helicopter': 45,
        'other': 46
        }
wordname_15=[]
def get_label_name_map():
    reverse_dict = {}
    for name, label in NAME_LABEL_MAP.items():
        reverse_dict[label] = name
        wordname_15.append(name)
    return reverse_dict
LABEl_NAME_MAP = get_label_name_map()

def parse_args():
    parser = argparse.ArgumentParser(description='prepare dota1')
    parser.add_argument('--srcpath', default='/media/zf/E/0plane_ori/PLANE')
    parser.add_argument('--dstpath', default=r'/media/zf/E/0plane_ori/plane_train_aug',
                        help='prepare data')
    args = parser.parse_args()

    return args


def single_copy(src_dst_tuple):
    shutil.copyfile(*src_dst_tuple)


def filecopy(srcpath, dstpath, num_process=32):
    pool = Pool(num_process)
    filelist = util.GetFileFromThisRootDir(srcpath)

    name_pairs = []
    for file in filelist:
        basename = os.path.basename(file.strip())
        dstname = os.path.join(dstpath, basename)
        name_tuple = (file, dstname)
        name_pairs.append(name_tuple)

    pool.map(single_copy, name_pairs)


def singel_move(src_dst_tuple):
    shutil.move(*src_dst_tuple)


def filemove(srcpath, dstpath, num_process=32):
    pool = Pool(num_process)
    filelist = util.GetFileFromThisRootDir(srcpath)

    name_pairs = []
    for file in filelist:
        basename = os.path.basename(file.strip())
        dstname = os.path.join(dstpath, basename)
        name_tuple = (file, dstname)
        name_pairs.append(name_tuple)

    pool.map(filemove, name_pairs)


def getnamelist(srcpath, dstfile):
    filelist = util.GetFileFromThisRootDir(srcpath)
    with open(dstfile, 'w') as f_out:
        for file in filelist:
            basename = util.mybasename(file)
            f_out.write(basename + '\n')


def prepare(srcpath, dstpath):
    """
    :param srcpath: train, val, test
          train --> trainval1024, val --> trainval1024, test --> test1024
    :return:
    """
    if not os.path.exists(os.path.join(dstpath, 'test1024')):
        os.makedirs(os.path.join(dstpath, 'test1024'))
    if not os.path.exists(os.path.join(dstpath, 'trainval1024')):
        os.makedirs(os.path.join(dstpath, 'trainval1024'))

    split_train = ImgSplit_multi_process.splitbase(os.path.join(srcpath, 'train'),
                                                   os.path.join(
                                                       dstpath, 'trainval1024'),
                                                   gap=150,
                                                   subsize=1024,
                                                   num_process=16,ext = '.tif'
                                                   )
    # split_train.splitdata(1)

    # split_train_ms = ImgSplit_multi_process.splitbase(os.path.join(srcpath, 'train'),
    #                                                   os.path.join(
    #                                                       dstpath, 'trainval1024_ms'),
    #                                                   gap=500,
    #                                                   subsize=1024,
    #                                                   num_process=16)
    # split_train_ms.splitdata(0.5)
    # split_train_ms.splitdata(1.5)

    split_val = ImgSplit_multi_process.splitbase(os.path.join(srcpath, 'val'),
                                                 os.path.join(
                                                     dstpath, 'test1024'),
                                                 gap=150,
                                                 subsize=1024,
                                                 num_process=16,ext = '.tif')
    # split_val.splitdata(1)

    # split_val_ms = ImgSplit_multi_process.splitbase(os.path.join(srcpath, 'val'),
    #                                                 os.path.join(
    #                                                     dstpath, 'trainval1024_ms'),
    #                                                 gap=500,
    #                                                 subsize=1024,
    #                                                 num_process=16)
    # split_val_ms.splitdata(0.5)
    # split_val_ms.splitdata(1.5)
    # split_test = SplitOnlyImage_multi_process.splitbase(os.path.join(srcpath, 'test', 'images'),
    #                                                     os.path.join(
    #                                                         dstpath, 'test1024', 'images'),
    #                                                     gap=500,
    #                                                     subsize=1024,
    #                                                     num_process=16)
    # split_test.splitdata(1)
    # split_test_ms = SplitOnlyImage_multi_process.splitbase(os.path.join(srcpath, 'test', 'images'),
    #                                                        os.path.join(
    #                                                            dstpath, 'test1024_ms', 'images'),
    #                                                        gap=500,
    #                                                        subsize=1024,
    #                                                        num_process=16
    #                                                        )
    # split_test_ms.splitdata(0.5)
    # split_test_ms.splitdata(1.5)

    DOTA2COCOTrain(os.path.join(dstpath, 'trainval1024'), os.path.join(
        dstpath, 'trainval1024', 'DOTA_trainval1024.json'), wordname_15, difficult='2', ext='.jpg')

    DOTA2COCOTrain(os.path.join(dstpath, 'test1024'), os.path.join(
        dstpath, 'test1024', 'DOTA_test1024.json'), wordname_15,difficult='2', ext='.jpg')

if __name__ == '__main__':
    args = parse_args()
    srcpath = args.srcpath
    dstpath = args.dstpath
    prepare(srcpath, dstpath)
