import os
import os.path as osp
import cv2
import mmcv
import numpy as np
from PIL import Image
# from dota_utils import GetFileFromThisRootDir
# from mmdet.core import poly_to_rotated_box_single
from timeit import default_timer as timer
import json
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
# def get_label_name_map():
#     reverse_dict = {}
#     for name, label in NAME_LABEL_MAP.items():
#         reverse_dict[label] = name
#     return reverse_dict
# LABEl_NAME_MAP = get_label_name_map()
wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
               'harbor', 'swimming-pool', 'helicopter']
LABEl_NAME_MAP={}
for i,classname in enumerate(wordname_15) :
    LABEl_NAME_MAP[i+1]=classname
NAME_LABEL_MAP = {name: i + 1 for i, name in enumerate(wordname_15)}

NAME_up =['bridge',
        'ground-track-field',
        'tennis-court',
        'basketball-court',
        'soccer-ball-field',
        'swimming-pool']
NAME_90 =['storage-tank', 'roundabout']

def parse_ann_info(label_base_path, img_name):
    lab_path = osp.join(label_base_path, img_name + '.txt')
    bboxes, masks, labels, bboxes_ignore,masks_ignore,  labels_ignore = [], [], [], [],[],[]
    with open(lab_path, 'r') as f:
        for ann_line in f.readlines():
            ann_line = ann_line.strip().split(' ')
            #TODO insert
            mask = [float(ann_line[i]) for i in range(8)]
            # 8 point to 5 point xywha
            bbox = tuple(poly_to_rotated_box_single(mask).tolist())
            #end
            class_name = ann_line[8]
            difficult = int(ann_line[9])
            # ignore difficult =2
            if difficult == 0:
                bboxes.append(bbox)
                labels.append(NAME_LABEL_MAP[class_name])
                masks.append(mask)
            elif difficult >1:
                bboxes_ignore.append(bbox)
                labels_ignore.append(NAME_LABEL_MAP[class_name])
                masks_ignore.append(mask)
    ann={}
    ann['bboxes']=bboxes
    ann['labels']=labels
    ann['masks']=masks
    ann['bboxes_ignore']=bbbboxes_ignoreoxes
    ann['labels_ignore']=labels_ignore
    ann['masks_ignore']=masks_ignore
    return ann#bboxes, labels, bboxes_ignore, labels_ignore

def rotate_rect2cv(rotatebox):
    #此程序将rotatexml中旋转矩形的表示，转换为cv2的RotateRect
    [x_center,y_center,w,h,angle]=rotatebox[0:5]
    angle_mod=angle*180/np.pi%180
    if angle_mod>=0 and angle_mod<90:
        [cv_w,cv_h,cv_angle]=[h,w,angle_mod-90]
    if angle_mod>=90 and angle_mod<180:
        [cv_w,cv_h,cv_angle]=[w,h,angle_mod-180]
    return ((x_center,y_center),(cv_w,cv_h),cv_angle) 

def parse_ann_info_rotatexml( lab_path,imgid):
    # lab_path = osp.join(label_base_path, img_name + '.txt')
    bboxes, masks, labels,imgids,rboxes= [], [], [], [], []

    with open(lab_path, 'r') as f:
        for i,line in enumerate (f.readlines()): 
            curLine=line.strip().split("\t")
            curLine=''.join(curLine)
            data=curLine.split( )
            try:
                points=[float(data[0]),float(data[1]),float(data[2]),float(data[3]),float(data[4]),float(data[5]),float(data[6]),float(data[7])]
            except :#如果报错就下一行
                continue
                print('error in {}'.format(lab_path))
            class_name = data[8]
            difficult = int(float(data[9]))
            point1=np.array([points[0],points[1]])
            point2=np.array([points[2],points[3]])
            point3=np.array([points[4],points[5]])
            point4=np.array([points[6],points[7]])
            l12=np.linalg.norm(point1-point2) 
            l23=np.linalg.norm(point2-point3) 
            l34=np.linalg.norm(point3-point4)
            l41=np.linalg.norm(point4-point1)
            head=(point1+point2)/2#头部坐标
            center=(point1+point2+point3+point4)/4#中心坐标
            Width=(l23+l41)/2
            Height=(l12+l34)/2
            det1=point2-point3
            det2=point1-point4
            if det1[0]==0:
                if det1[1]>0:
                    Angle1=np.pi/2
                else:
                    Angle1=-np.pi/2
            else:
                Angle1=np.arctan(det1[1]/det1[0])
            if det2[0]==0:
                if det2[1]>0:
                    Angle2=np.pi/2
                else:
                    Angle2=-np.pi/2
            else:
                Angle2=np.arctan(det2[1]/det2[0])
            #还会出现一种情况就是angle1 angle2 都比较大，但是由于在90度俯角，导致两个差异很大
            if np.abs(Angle1)>np.pi/2-np.pi/36:
                if Angle2<0:
                    Angle1=-np.pi/2
                else:
                    Angle1=np.pi/2
            if np.abs(Angle2)>np.pi/2-np.pi/36:
                if Angle1<0:
                    Angle2=-np.pi/2
                else:
                    Angle2=np.pi/2
            Angle=(Angle1+Angle2)/2
            #以上得到了HRSC格式的表示的各项数据，以下将其转为旋转xml格式的表示的数据
            #分别计算旋转矩形两个头部的坐标，和实际我们得出的头部坐标比较，距离小的我们就认为他是头部
            head_rect_right=[center[0]+Width/2*np.cos(Angle),center[1]+Width/2*np.sin(Angle)]
            head_rect_left=[center[0]-Width/2*np.cos(Angle),center[1]-Width/2*np.sin(Angle)]
            l_head_right=np.linalg.norm(head_rect_right-head) 
            l_head_left=np.linalg.norm(head_rect_left-head) 
            if l_head_right<l_head_left:#头部方向在第一四象限
                Angle=Angle+np.pi/2
            else:
                Angle=Angle+np.pi*3/2#头部方向在第二三象限，角度要在原来基础上加上PI
            NewWidth=Height
            NewHeight=Width
            Angle=np.mod(Angle,2*np.pi)
            #为了纠正游泳池的长宽和歧义这里纠正一下
            if class_name =='swimming-pool':
                if NewWidth>NewHeight:
                    Angle += np.pi/2
                    Angle=np.mod(Angle,2*np.pi)
                    NewWidth,NewHeight=NewHeight ,NewWidth

            #这里因为有些类别头部方向有歧义，所以更正一下主方向
            if class_name in NAME_up:
                if Angle>=np.pi/2 and Angle<np.pi:#如果再第四像限 则角度加pi
                    Angle=Angle+np.pi
                elif  Angle>=np.pi and Angle<3*np.pi/2:#再第三像限 则角度-pi
                    Angle=Angle-np.pi
            elif class_name in NAME_90:#draw circle
                Angle=0
                NewWidth,NewHeight=(NewWidth+NewHeight)/2,(NewWidth+NewHeight)/2

            rbox=[center[0],center[1],NewWidth,NewHeight,Angle,NAME_LABEL_MAP[class_name],difficult]#这就是最终的rotatexml格式表示
            cv_rbox=rotate_rect2cv(rbox[0:5])
            rect_box = cv2.boxPoints(cv_rbox)
            xmin,ymin,xmax,ymax=float(np.min(rect_box[:,0])),float(np.min(rect_box[:,1])),float(np.max(rect_box[:,0])),float(np.max(rect_box[:,1]))
            bbox=[xmin,ymin,xmax-xmin,ymax-ymin]#这是根据rbox得到的bbox
            bboxes.append(bbox)
            masks.append(points)
            rboxes.append(rbox)
            imgids.append(imgid)
    ann={}
    ann['bboxes']=bboxes
    ann['masks']=masks
    ann['rboxes']=rboxes
    ann['imgids']=imgids
    return ann#bboxes, labels, bboxes_ignore, labels_ignore


def generate_file_list(img_dir,output_txt,file_ext='.txt'):
    #读取原图路径  
    # img_dir=os.path.split(img_dir)[0]
    imgs_path = GetFileFromThisRootDir(img_dir, file_ext) 
    f = open(output_txt, "w",encoding='utf-8')
    for num,img_path in enumerate(imgs_path,0):
        obj='{}\n'.format(os.path.splitext(os.path.split(img_path)[1])[0])
        f.write(obj)
    f.close()
    print('Generate {} down!'.format(output_txt))
    
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

headstr = """\
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
objstr = """\
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
		<extra>{}</extra>
	</object>
    """
tailstr = '''\
      </annotation>
    '''
def write_rotate_xml(output_floder,img_name,size,gsd,imagesource,gtbox_label,CLASSES,extra=[]):#size,gsd,imagesource
    #将检测结果表示为中科星图比赛格式的程序,这里用folder字段记录gsd
    [floder,name]=os.path.split(img_name)
    # filename=name.replace('.jpg','.xml')
    filename=os.path.join(floder,os.path.splitext(name)[0]+'.xml')
    foldername=os.path.split(img_name)[0]
    head=headstr.format(gsd,name,foldername,imagesource,size[1],size[0],size[2])
    rotate_xml_name=os.path.join(output_floder,os.path.split(filename)[1])
    f = open(rotate_xml_name, "w",encoding='utf-8')
    f.write(head)
    for i,box in enumerate (gtbox_label):
        if len(extra)==0:
            obj=objstr.format(CLASSES[int(box[5])],int(box[6]),box[0],box[1],box[2],box[3],box[4],' ')
        else:
            obj=objstr.format(CLASSES[int(box[5])],int(box[6]),box[0],box[1],box[2],box[3],box[4],extra[i])
        f.write(obj)
    f.write(tailstr)
    f.close()

def make_if_not_exit(dirpath):
    if not osp.exists(dirpath):
        os.mkdir(dirpath)

def convert_dota_to_mmdet_rep(src_path, flag, out_path, trainval='train', ext='.png'):#这个函数是要将切割后的dota，txt格式转换为json格式
    """Generate .pkl format annotation that is consistent with mmdet.
    Args:
        src_path: dataset path containing images and labelTxt folders.
        out_path: output json file path
        trainval: trainval or test
    """
    img_path = os.path.join(src_path, '{}2017'.format(flag))
    label_path = os.path.join(src_path, '{}2017labelTxt'.format(flag))
    img_lists = get_file_paths_recursive(img_path,ext)#os.listdir(img_path)
    outputfolder = osp.dirname(out_path)
    make_if_not_exit(outputfolder)
    outputfolder= osp.join( osp.dirname(outputfolder), '{}2017rotatexml'.format(flag))
    make_if_not_exit(outputfolder)
    
    start = timer()
    categories = []#这里开始制作categories字段
    for iind, cat in enumerate(NAME_LABEL_MAP.items()):
        cate = {}
        cate['supercategory'] = cat
        cate['name'] = cat
        cate['id'] = iind + 1
        cate['keypoints'] = [
                    "ship head"
                ]
        cate['skeleton'] =   [
                    [
                        1,
                        1
                    ]
                ]
        categories.append(cate)

    img_infos=[]
    bboxes, masks, rboxes,imgids =[],[],[],[]
    total_num=len(img_lists)
    for imgid, img in enumerate(img_lists):
        img_info = {}
        img_name = osp.splitext(img)[0]
        label = (img_name + '.txt').replace(img_path,label_path)
        img = Image.open(osp.join(img_path, img))#获得图像信息
        img_info['file_name'] = osp.split(img_name)[-1] + ext
        img_info['height'] = img.height
        img_info['width'] = img.width
        img_info['id'] = int(imgid)
        img_infos.append(img_info)
        # /media/zf/E/Dataset/dota_1024_s2anet2/val1027labelTxt/P0003__1.0__0___0.txt
  
        if not os.path.exists(label):
            print('Label:' + label +  ' Not Exist')
        else:
            parse_ann=parse_ann_info_rotatexml(label,imgid)#这里解析标注txt文件返回字典
            if  trainval =='train' and len(parse_ann['bboxes'])==0:
                continue
            bboxes.extend( parse_ann['bboxes'])
            masks.extend( parse_ann['masks'])
            rboxes.extend( parse_ann['rboxes'])
            imgids.extend( parse_ann['imgids'])
            
            difficult=np.array(parse_ann['rboxes'])[:,-1] if len(parse_ann['rboxes'])>0 else []
            write_rotate_xml(outputfolder, img_info['file_name'],[1024,1024,3],0.5,'imagesource', parse_ann['rboxes'],LABEl_NAME_MAP,difficult)
        
        if imgid%1000==0:
            print('{}/{}'.format(imgid,total_num))
    
    print('parse txt annotations down!')
    ann_js = {}
    annotations=[]
    total_box=len(bboxes)
    assert (len(bboxes)==len(rboxes)==len(masks)==len(imgids))
    for i, box in enumerate(bboxes):
        anno = {}
        anno['image_id'] = int(imgids[i])
        anno['category_id'] = rboxes[i][-2]  #1 HRSC
        anno['bbox'] = box[0:4]
        anno['id'] = int(i)
        anno['area'] = rboxes[i][2]*rboxes[i][3]
        anno['iscrowd'] = rboxes[i][-1]
        anno['segmentation']=[masks[i]]
        anno['rbox']=[rboxes[i]]
        [cx,cy,w,h,angle,label]=rboxes[i][0:6]
        r0,r1=np.transpose([0,-h/2]),np.transpose([w/2,0])
        RotateMatrix=np.array([
                              [np.cos(angle),-np.sin(angle)],
                              [np.sin(angle),np.cos(angle)]])
        p0=np.transpose(np.dot(RotateMatrix, r0))+[cx,cy]
        p1=np.transpose(np.dot(RotateMatrix, r1))+[cx,cy]
        anno['num_keypoints'] = 2#2
        z1=np.array([p0[0], p0[1], 2, p1[0], p1[1], 2]).reshape((1,-1))#np.ones([1,3])
        anno['keypoints']=z1.tolist()
        annotations.append(anno)
        if i%5000==0:
            print('{}/{}'.format(i,total_box))
            
    ann_js['images'] = img_infos
    ann_js['categories'] = categories
    ann_js['annotations'] = annotations
    json.dump(ann_js, open(out_path, 'w'), indent=4)  # indent=4 更加美观显示
    time_elapsed = timer() - start
    print('time:{}s'.format(time_elapsed))
    print('down!')
    # mmcv.dump(data_dict, out_path)

def convert_dota_to_mmdet(img_path, label_path, out_path, trainval='train', ext='.png'):#这个函数是要将切割后的dota，txt格式转换为json格式
    """Generate .pkl format annotation that is consistent with mmdet.
    Args:
        src_path: dataset path containing images and labelTxt folders.
        out_path: output json file path
        trainval: trainval or test
    """
    # img_path = os.path.join(src_path, '{}2017'.format(flag))
    # label_path = os.path.join(src_path, '{}2017labelTxt'.format(flag))
    img_lists = get_file_paths_recursive(img_path,ext)#os.listdir(img_path)
    outputfolder = osp.dirname(out_path)
    make_if_not_exit(outputfolder)
    outputfolder= osp.join( osp.dirname(outputfolder), '{}2017rotatexml'.format(flag))
    make_if_not_exit(outputfolder)
    
    start = timer()
    categories = []#这里开始制作categories字段
    for iind, cat in enumerate(NAME_LABEL_MAP.items()):
        cate = {}
        cate['supercategory'] = cat
        cate['name'] = cat
        cate['id'] = iind + 1
        cate['keypoints'] = [
                    "ship head"
                ]
        cate['skeleton'] =   [
                    [
                        1,
                        1
                    ]
                ]
        categories.append(cate)

    img_infos=[]
    bboxes, masks, rboxes,imgids =[],[],[],[]
    total_num=len(img_lists)
    for imgid, img in enumerate(img_lists):
        img_info = {}
        img_name = osp.splitext(img)[0]
        label = (img_name + '.txt').replace(img_path,label_path)
        img = Image.open(osp.join(img_path, img))#获得图像信息
        img_info['file_name'] = osp.split(img_name)[-1] + ext
        img_info['height'] = img.height
        img_info['width'] = img.width
        img_info['id'] = int(imgid)
        img_infos.append(img_info)
        # /media/zf/E/Dataset/dota_1024_s2anet2/val1027labelTxt/P0003__1.0__0___0.txt
  
        if not os.path.exists(label):
            print('Label:' + label +  ' Not Exist')
        else:
            parse_ann=parse_ann_info_rotatexml(label,imgid)#这里解析标注txt文件返回字典
            if  trainval =='train' and len(parse_ann['bboxes'])==0:
                continue
            bboxes.extend( parse_ann['bboxes'])
            masks.extend( parse_ann['masks'])
            rboxes.extend( parse_ann['rboxes'])
            imgids.extend( parse_ann['imgids'])
            
            difficult=np.array(parse_ann['rboxes'])[:,-1] if len(parse_ann['rboxes'])>0 else []
            write_rotate_xml(outputfolder, img_info['file_name'],[1024,1024,3],0.5,'imagesource', parse_ann['rboxes'],LABEl_NAME_MAP,difficult)
        
        if imgid%1000==0:
            print('{}/{}'.format(imgid,total_num))
    
    print('parse txt annotations down!')
    ann_js = {}
    annotations=[]
    total_box=len(bboxes)
    assert (len(bboxes)==len(rboxes)==len(masks)==len(imgids))
    for i, box in enumerate(bboxes):
        anno = {}
        anno['image_id'] = int(imgids[i])
        anno['category_id'] = rboxes[i][-2]  #1 HRSC
        anno['bbox'] = box[0:4]
        anno['id'] = int(i)
        anno['area'] = rboxes[i][2]*rboxes[i][3]
        anno['iscrowd'] = rboxes[i][-1]
        anno['segmentation']=[masks[i]]
        anno['rbox']=[rboxes[i]]
        [cx,cy,w,h,angle,label]=rboxes[i][0:6]
        r0,r1=np.transpose([0,-h/2]),np.transpose([w/2,0])
        RotateMatrix=np.array([
                              [np.cos(angle),-np.sin(angle)],
                              [np.sin(angle),np.cos(angle)]])
        p0=np.transpose(np.dot(RotateMatrix, r0))+[cx,cy]
        p1=np.transpose(np.dot(RotateMatrix, r1))+[cx,cy]
        anno['num_keypoints'] = 2#2
        z1=np.array([p0[0], p0[1], 2, p1[0], p1[1], 2]).reshape((1,-1))#np.ones([1,3])
        anno['keypoints']=z1.tolist()
        annotations.append(anno)
        if i%5000==0:
            print('{}/{}'.format(i,total_box))
            
    ann_js['images'] = img_infos
    ann_js['categories'] = categories
    ann_js['annotations'] = annotations
    json.dump(ann_js, open(out_path, 'w'), indent=4)  # indent=4 更加美观显示
    time_elapsed = timer() - start
    print('time:{}s'.format(time_elapsed))
    print('down!')
    
if __name__ == '__main__':
    # convert_dota_to_mmdet_rep('/media/zf/E/Dataset/plane_train_aug2',
    #                           'train',
    #                       '/media/zf/E/Dataset/plane_train_aug2/annotations/reppoint_keypoints_train2017.json',ext='.jpg')
    # convert_dota_to_mmdet_rep('/media/zf/E/Dataset/plane_train_aug2',
    #                           'val',
    #                       '/media/zf/E/Dataset/plane_train_aug2/annotations/reppoint_keypoints_val2017.json',ext='.jpg')
      # convert_dota_to_mmdet_rep(dst_trainval_path,
    #                       osp.join(dst_trainval_path, 'trainval1024.pkl'))
    convert_dota_to_mmdet_rep('/media/zf/E/Dataset/dota_1024_s2anet2',
                              'train',
                          '/media/zf/E/Dataset/dota_1024_s2anet2/annotations/reppoint_fliter_keypoints_val2017.json')
    # convert_dota_to_mmdet_rep('/media/zf/E/Dataset/plane_aug',
    #                           'val',
    #                       '/media/zf/E/Dataset/plane_aug/annotations/val_train2017.json',ext='.jpg')

    # convert_dota_to_mmdet('/media/zf/E/Dataset/dota_1024_s2anet2/trainval_split/images',
    #                       '/media/zf/E/Dataset/dota1-split-1024/trainval1024/labelTxt',
    #                      '/media/zf/E/Dataset/dota_1024_s2anet2/trainval_split/trainval_split/reppoint_keypoints_train2017.json')
    # convert_dota_to_mmdet('/home/zf/Dataset/dota_1024_s2anet2/test_split/',
    #                      '/home/zf/Dataset/dota_1024_s2anet2/test_split/test_s2anet.pkl', trainval=False)
    # generate_file_list(img_dir,output_txt,file_ext='.txt')
    # convert_dota_to_mmdet('data/dota_1024/trainval_split',
    #                      'data/dota_1024/trainval_split/trainval_s2anet.pkl')
    # convert_dota_to_mmdet('data/dota_1024/test_split',
    #                      'data/dota_1024/test_split/test_s2anet.pkl', trainval=False)
    print('done!')
