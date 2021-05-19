import os
os.system("python -m tools.train_net --num-gpus 1 --config-file configs/COCO-detection/faster_rcnn_R_50_FPN_ft_all_jiaonang_1shot.yaml")
os.system("python -m tools.train_net --num-gpus 1 --config-file configs/COCO-detection/faster_rcnn_R_50_FPN_ft_all_jiaonang_2shot.yaml")
os.system("python -m tools.train_net --num-gpus 1 --config-file configs/COCO-detection/faster_rcnn_R_50_FPN_ft_all_jiaonang_3shot.yaml")