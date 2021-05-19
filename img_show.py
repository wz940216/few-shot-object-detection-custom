# from __future__ import print_function
# from pycocotools.coco import COCO
# import os, sys, zipfile
# import urllib.request
# import shutil
# import numpy as np
# import skimage.io as io
# import matplotlib.pyplot as plt
# import pylab
# import json
# json_file='dataset.json' #
# # Object Instance 类型的标注
# # person_keypoints_val2017.json
# # Object Keypoint 类型的标注格式
# # captions_val2017.json
# # Image Caption的标注格式
# data=json.load(open(json_file,'r'))
# data_2={}
# data_2['info']=data['info']
# data_2['licenses']=data['licenses']
# data_2['images']=[data['images'][0]] # 只提取第一张图片
# data_2['categories']=data['categories']
# annotation=[] # 通过imgID 找到其所有对象
# imgID=data_2['images'][0]['id']
# for ann in data['annotations']:
#     if ann['image_id']==imgID:
#         annotation.append(ann)
# data_2['annotations']=annotation # 保存到新的JSON文件，便于查看数据特点
# json.dump(data_2,open('dataset_mini.json','w'),indent=4) # indent=4 更加美观显示

#from __future__ import print_function
from pycocotools.coco import COCO
import os, sys, zipfile
import urllib.request
import shutil
import numpy as np
import skimage.io as io
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pylab
dataDir = 'datasets/jiaonang/base/test'
annFile='datasets/jiaonang/base/test.json'
pylab.rcParams['figure.figsize'] = (8.0, 10.0)


for i in range(10):
    coco=COCO(annFile) # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    nms=[cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))
    # nms = set([cat['supercategory'] for cat in cats])
    # print('COCO supercategories: \n{}'.format(' '.join(nms)))
    # imgIds = coco.getImgIds(imgIds = [324158])
    imgIds = coco.getImgIds()
    img = coco.loadImgs(imgIds[i])[0]
    I = io.imread('%s/%s'%(dataDir,img['file_name']))
    catIds=[]
    for ann in coco.dataset['annotations']:
        if ann['image_id']==imgIds[i]:
            catIds.append(ann['category_id'])
    plt.imshow(I);
    plt.axis('off')
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    print(anns)
    coco.showAnns(anns)
    plt.imshow(I); plt.axis('off'); plt.show()
    plt.close()

#加载instances mask：

# coco = COCO("dataset.json")
#
# img_ids = coco.getImgIds()
# print(len(img_ids))
# cat_ids = []
# for ann in coco.dataset["annotations"]:
#     if ann["image_id"] == img_ids[0]:
#         cat_ids.append(ann["category_id"])
# ann_ids = coco.getAnnIds(imgIds=img_ids[0], catIds = cat_ids)
# ann_ids2 = coco.getAnnIds(imgIds=img_ids[0], catIds = cat_ids)
# plt.imshow(I)
# print(ann_ids)
# print(ann_ids2)
# anns = coco.loadAnns(ann_ids)
# coco.showAnns(anns)
# plt.imshow(I)
# plt.show()