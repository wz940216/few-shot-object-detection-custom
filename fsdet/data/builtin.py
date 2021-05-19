"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.
We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations
We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Here we only register the few-shot datasets and complete COCO, PascalVOC and 
LVIS have been handled by the builtin datasets in detectron2. 
"""

import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.lvis import (
    get_lvis_instances_meta,
    register_lvis_instances,
)
from detectron2.data.datasets.pascal_voc import register_pascal_voc
from detectron2.data.datasets.register_coco import register_coco_instances

from .builtin_meta import _get_builtin_metadata
from .meta_coco import register_meta_coco
from .meta_lvis import register_meta_lvis
from .meta_pascal_voc import register_meta_pascal_voc

# ==== Predefined datasets and splits for COCO ==========
root_pth = "F:/workspace/Daheng/Deep-learning-library/few-shot-object-detection-master/datasets"
_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    # jiaonang
    "jiaonang_train": (
        "coco/jiaonang/train",
        "coco/jiaonang/train.json",
    ),
    # "coco_2014_train": (
    #     "coco/train2014",
    #     "coco/annotations/instances_train2014.json",
    # ),
    # "coco_2014_val": (
    #     "coco/val2014",
    #     "coco/annotations/instances_val2014.json",
    # ),
    # "coco_2014_minival": (
    #     "coco/val2014",
    #     "coco/annotations/instances_minival2014.json",
    # ),
    # "coco_2014_minival_100": (
    #     "coco/val2014",
    #     "coco/annotations/instances_minival2014_100.json",
    # ),
    # "coco_2014_valminusminival": (
    #     "coco/val2014",
    #     "coco/annotations/instances_valminusminival2014.json",
    # ),
    # "coco_2017_train": (
    #     "coco/train2017",
    #     "coco/annotations/instances_train2017.json",
    # ),
    # "coco_2017_val": (
    #     "coco/val2017",
    #     "coco/annotations/instances_val2017.json",
    # ),
    # "coco_2017_test": (
    #     "coco/test2017",
    #     "coco/annotations/image_info_test2017.json",
    # ),
    # "coco_2017_test-dev": (
    #     "coco/test2017",
    #     "coco/annotations/image_info_test-dev2017.json",
    # ),
    # "coco_2017_val_100": (
    #     "coco/val2017",
    #     "coco/annotations/instances_val2017_100.json",
    # ),
}


def register_all_coco(root=root_pth):
    # for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
    #     for key, (image_root, json_file) in splits_per_dataset.items():
    #         # Assume pre-defined datasets live in `./datasets`.
    #         register_coco_instances(
    #             key,
    #             _get_builtin_metadata(dataset_name),
    #             os.path.join(root, json_file)
    #             if "://" not in json_file
    #             else json_file,
    #             os.path.join(root, image_root),
    #         )

    # register meta datasets
    METASPLITS = [
        (
            "jiaonang_train_all",
            "coco/jiaonang/train",
            "coco/jiaonang/train.json",
        ),
        (
            "jiaonang_train_base",
            "coco/jiaonang/train",
            "coco/jiaonang/train.json",
        ),
        ("test_all", "coco/jiaonang/test", "coco/jiaonang/test.json"),
        ("test_base", "coco/jiaonang/test", "coco/jiaonang/test.json"),
        ("test_novel", "coco/jiaonang/test", "coco/jiaonang/test.json"),
    ]

    # register small meta datasets for fine-tuning stage
    for prefix in ["all", "novel"]:
        for shot in [1, 2, 3]:#shot改为自己的
            for seed in range(10):
                seed = "" if seed == 0 else "_seed{}".format(seed)
                name = "coco_trainval_{}_{}shot{}".format(prefix, shot, seed)
                METASPLITS.append((name, "coco/jiaonang/train", ""))

    for name, imgdir, annofile in METASPLITS:
        register_meta_coco(
            name,
            _get_builtin_metadata("coco_fewshot"),
            os.path.join(root, imgdir),
            os.path.join(root, annofile),
        )

# ==== register custom dataset for coco format ==========

_PREDEFINED_BASE_DATA = {
    "jiaonang_base_data":{
        "base_train":("jiaonang/base/train","jiaonang/base/train.json"),
        "base_test":("jiaonang/base/test","jiaonang/base/test.json"),
        "base_val":("jiaonang/base/val","jiaonang/base/val.json")
    }
}
def register_base_data(root=root_pth):
    for dataset_name, splits_per_dataset in _PREDEFINED_BASE_DATA.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                {},
                #_get_builtin_metadata(dataset_name),
                os.path.join(root, json_file)
                if "://" not in json_file
                else json_file,
                os.path.join(root, image_root),
            )

_PREDEFINED_BALANCE_DATA = {
    "jiaonang_balance_data":{
        "balance_train":("jiaonang/balance/train","jiaonang/balance/train.json"),
        "balance_test":("jiaonang/balance/test","jiaonang/balance/test.json"),
        "balance_val":("jiaonang/balance/val","jiaonang/balance/val.json")
    }
}
def register_balance_data(root=root_pth):
    for dataset_name, splits_per_dataset in _PREDEFINED_BALANCE_DATA.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                {},
                # _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file)
                if "://" not in json_file
                else json_file,
                os.path.join(root, image_root),
            )

_PREDEFINED_FEW_DATA = {
    "jiaonang_few_data":{
        "few_train":("jiaonang/few/train","jiaonang/few/train.json"),
        "few_test":("jiaonang/few/few","jiaonang/few/test.json"),
        "few_val":("jiaonang/few/val","jiaonang/few/val.json")
    }
}
def register_few_data(root=root_pth):
    for dataset_name, splits_per_dataset in _PREDEFINED_FEW_DATA.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                {},
                # _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file)
                if "://" not in json_file
                else json_file,
                os.path.join(root, image_root),
            )

_PREDEFINED_YINXIAN_DATA = {
    "yinxian_data":{
        "yx_train":("yinxian/images","yinxian/dataset.json"),
        "yx_test": ("yinxian/test", "yinxian/test.json")
    }
}
def register_yx_data(root=root_pth):
    for dataset_name, splits_per_dataset in _PREDEFINED_YINXIAN_DATA.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                {},
                # _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file)
                if "://" not in json_file
                else json_file,
                os.path.join(root, image_root),
            )
            
# ==== Predefined datasets and splits for LVIS ==========

_PREDEFINED_SPLITS_LVIS = {
    "lvis_v0.5": {
        # "lvis_v0.5_train": ("coco/train2017", "lvis/lvis_v0.5_train.json"),
        "lvis_v0.5_train_freq": (
            "coco/train2017",
            "lvis/lvis_v0.5_train_freq.json",
        ),
        "lvis_v0.5_train_common": (
            "coco/train2017",
            "lvis/lvis_v0.5_train_common.json",
        ),
        "lvis_v0.5_train_rare": (
            "coco/train2017",
            "lvis/lvis_v0.5_train_rare.json",
        ),
        # "lvis_v0.5_val": ("coco/val2017", "lvis/lvis_v0.5_val.json"),
        # "lvis_v0.5_val_rand_100": (
        #     "coco/val2017",
        #     "lvis/lvis_v0.5_val_rand_100.json",
        # ),
        # "lvis_v0.5_test": (
        #     "coco/test2017",
        #     "lvis/lvis_v0.5_image_info_test.json",
        # ),
    },
}


def register_all_lvis(root="datasets"):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LVIS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_lvis_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file)
                if "://" not in json_file
                else json_file,
                os.path.join(root, image_root),
            )

    # register meta datasets
    METASPLITS = [
        (
            "lvis_v0.5_train_shots",
            "coco/train2017",
            "lvissplit/lvis_shots.json",
        ),
        (
            "lvis_v0.5_train_rare_novel",
            "coco/train2017",
            "lvis/lvis_v0.5_train_rare.json",
        ),
        ("lvis_v0.5_val_novel", "coco/val2017", "lvis/lvis_v0.5_val.json"),
    ]

    for name, image_root, json_file in METASPLITS:
        dataset_name = "lvis_v0.5_fewshot" if "novel" in name else "lvis_v0.5"
        register_meta_lvis(
            name,
            _get_builtin_metadata(dataset_name),
            os.path.join(root, json_file)
            if "://" not in json_file
            else json_file,
            os.path.join(root, image_root),
        )


# ==== Predefined splits for PASCAL VOC ===========
def register_all_pascal_voc(root=root_pth):
    # SPLITS = [
    #     ("voc_2007_trainval", "VOC2007", "trainval"),
    #     ("voc_2007_train", "VOC2007", "train"),
    #     ("voc_2007_val", "VOC2007", "val"),
    #     ("voc_2007_test", "VOC2007", "test"),
    #     ("voc_2012_trainval", "VOC2012", "trainval"),
    #     ("voc_2012_train", "VOC2012", "train"),
    #     ("voc_2012_val", "VOC2012", "val"),
    # ]
    # for name, dirname, split in SPLITS:
    #     year = 2007 if "2007" in name else 2012
    #     register_pascal_voc(name, os.path.join(root, dirname), split, year)
    #     MetadataCatalog.get(name).evaluator_type = "pascal_voc"

    # register meta datasets
    METASPLITS = [
        ("voc_2007_trainval_base1", "VOC2007", "trainval", "base1", 1),
        ("voc_2007_trainval_base2", "VOC2007", "trainval", "base2", 2),
        ("voc_2007_trainval_base3", "VOC2007", "trainval", "base3", 3),
        ("voc_2012_trainval_base1", "VOC2012", "trainval", "base1", 1),
        ("voc_2012_trainval_base2", "VOC2012", "trainval", "base2", 2),
        ("voc_2012_trainval_base3", "VOC2012", "trainval", "base3", 3),
        ("voc_2007_trainval_all1", "VOC2007", "trainval", "base_novel_1", 1),
        ("voc_2007_trainval_all2", "VOC2007", "trainval", "base_novel_2", 2),
        ("voc_2007_trainval_all3", "VOC2007", "trainval", "base_novel_3", 3),
        ("voc_2012_trainval_all1", "VOC2012", "trainval", "base_novel_1", 1),
        ("voc_2012_trainval_all2", "VOC2012", "trainval", "base_novel_2", 2),
        ("voc_2012_trainval_all3", "VOC2012", "trainval", "base_novel_3", 3),
        ("voc_2007_test_base1", "VOC2007", "test", "base1", 1),
        ("voc_2007_test_base2", "VOC2007", "test", "base2", 2),
        ("voc_2007_test_base3", "VOC2007", "test", "base3", 3),
        ("voc_2007_test_novel1", "VOC2007", "test", "novel1", 1),
        ("voc_2007_test_novel2", "VOC2007", "test", "novel2", 2),
        ("voc_2007_test_novel3", "VOC2007", "test", "novel3", 3),
        ("voc_2007_test_all1", "VOC2007", "test", "base_novel_1", 1),
        ("voc_2007_test_all2", "VOC2007", "test", "base_novel_2", 2),
        ("voc_2007_test_all3", "VOC2007", "test", "base_novel_3", 3),
    ]

    # register small meta datasets for fine-tuning stage
    for prefix in ["all", "novel"]:
        for sid in range(1, 4):
            for shot in [1, 2, 3, 5, 10]:
                for year in [2007, 2012]:
                    for seed in range(100):
                        seed = "" if seed == 0 else "_seed{}".format(seed)
                        name = "voc_{}_trainval_{}{}_{}shot{}".format(
                            year, prefix, sid, shot, seed
                        )
                        dirname = "VOC{}".format(year)
                        img_file = "{}_{}shot_split_{}_trainval".format(
                            prefix, shot, sid
                        )
                        keepclasses = (
                            "base_novel_{}".format(sid)
                            if prefix == "all"
                            else "novel{}".format(sid)
                        )
                        METASPLITS.append(
                            (name, dirname, img_file, keepclasses, sid)
                        )

    for name, dirname, split, keepclasses, sid in METASPLITS:
        year = 2007 if "2007" in name else 2012
        register_meta_pascal_voc(
            name,
            _get_builtin_metadata("pascal_voc_fewshot"),
            os.path.join(root, dirname),
            split,
            year,
            keepclasses,
            sid,
        )
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


# Register them all under "./datasets"
register_all_coco()
register_all_lvis()
register_all_pascal_voc()
# Register custom data
register_base_data()
register_balance_data()
register_few_data()
register_yx_data()

#引入以下注释
# import cv2
# from detectron2.data import DatasetCatalog, MetadataCatalog
# from detectron2.data.datasets.coco import load_coco_json
# from detectron2.utils.visualizer import  Visualizer
# import pycocotools
# #声明类别，尽量保持
# CLASS_NAMES =["0","1","2"]
# # 数据集路径
# DATASET_ROOT = 'F:/workspace/Daheng/Deep-learning-library/few-shot-object-detection-master/datasets/jiaonang'
# ANN_ROOT = os.path.join(DATASET_ROOT, 'base')
#
# TRAIN_PATH = os.path.join(ANN_ROOT, 'train')
# VAL_PATH = os.path.join(ANN_ROOT, 'val')
#
# TRAIN_JSON = os.path.join(ANN_ROOT, 'train.json')
# #VAL_JSON = os.path.join(ANN_ROOT, 'val.json')
# VAL_JSON = os.path.join(ANN_ROOT, 'val.json')
#
# # 声明数据集的子集
# PREDEFINED_SPLITS_DATASET = {
#     "coco_my_train": (TRAIN_PATH, TRAIN_JSON),
#     "coco_my_val": (VAL_PATH, VAL_JSON),
# }
#
# #注册数据集（这一步就是将自定义数据集注册进Detectron2）
# def register_dataset():
#     """
#     purpose: register all splits of dataset with PREDEFINED_SPLITS_DATASET
#     """
#     for key, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
#         register_dataset_instances(name=key,
#                                    json_file=json_file,
#                                    image_root=image_root)
#
#
# #注册数据集实例，加载数据集中的对象实例
# def register_dataset_instances(name, json_file, image_root):
#     """
#     purpose: register dataset to DatasetCatalog,
#              register metadata to MetadataCatalog and set attribute
#     """
#     DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
#     MetadataCatalog.get(name).set(json_file=json_file,
#                                   image_root=image_root,
#                                   evaluator_type="coco")
#
#
# # 注册数据集和元数据
# def plain_register_dataset():
#     #训练集
#     DatasetCatalog.register("coco_my_train", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH))
#     MetadataCatalog.get("coco_my_train").set(thing_classes=CLASS_NAMES,  # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
#                                                     evaluator_type='coco', # 指定评估方式
#                                                     json_file=TRAIN_JSON,
#                                                     image_root=TRAIN_PATH)
#
#     #DatasetCatalog.register("coco_my_val", lambda: load_coco_json(VAL_JSON, VAL_PATH, "coco_2017_val"))
#     #验证/测试集
#     DatasetCatalog.register("coco_my_val", lambda: load_coco_json(VAL_JSON, VAL_PATH))
#     MetadataCatalog.get("coco_my_val").set(thing_classes=CLASS_NAMES, # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
#                                                 evaluator_type='coco', # 指定评估方式
#                                                 json_file=VAL_JSON,
#                                                 image_root=VAL_PATH)
# # 查看数据集标注，可视化检查数据集标注是否正确，
# #这个也可以自己写脚本判断，其实就是判断标注框是否超越图像边界
# #可选择使用此方法
# def checkout_dataset_annotation(name="coco_my_val"):
#     #dataset_dicts = load_coco_json(TRAIN_JSON, TRAIN_PATH, name)
#     dataset_dicts = load_coco_json(TRAIN_JSON, TRAIN_PATH)
#     #print(len(dataset_dicts))
#     for i, d in enumerate(dataset_dicts,0):
#         #print(d)
#         img = cv2.imread(d["file_name"])
#         visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(name), scale=1.5)
#         vis = visualizer.draw_dataset_dict(d)
#         cv2.imshow('show', vis.get_image()[:, :, ::-1])
#         cv2.imwrite('out/'+str(i) + '.jpg',vis.get_image()[:, :, ::-1])
#         cv2.waitKey(0)
#         if i == 200:
#             break