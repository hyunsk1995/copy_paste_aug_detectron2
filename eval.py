# import some common libraries
import numpy as np
import cv2
import torch
import random

# import some common detectron2 utilities
from detectron2 import model_zoo

from collections import OrderedDict
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog,DatasetCatalog
from detectron2.data.datasets import register_coco_instances,load_coco_json
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils

from detectron2.engine import DefaultTrainer, hooks
from detectron2.engine import HookBase
from detectron2.engine import launch
from detectron2.engine import DefaultPredictor
from detectron2.engine import default_argument_parser

from detectron2.checkpoint import Checkpointer
from detectron2.modeling import build_model

import detectron2.utils.comm as comm

from pycocotools import mask
from skimage import measure

import copy
import os
from matplotlib import pyplot as plt

from copy_paste import CopyPaste
from coco import CocoDetectionCP

import albumentations as A
from visualize import display_instances

from copy_paste import MAX_ITER

coco_path = "../cpaug"

# Data Preprocess
DatasetCatalog.clear()
for d in ['train','val']:
    DatasetCatalog.register("coco_"+d, lambda d=d: load_coco_json(coco_path + "/coco/annotations/instances_{}2017.json".format(d,d),
    image_root= coco_path+"/coco/{}2017".format(d),\
    dataset_name="coco_"+d))

dataset_dicts_train = DatasetCatalog.get("coco_train")
dataset_dicts_test = DatasetCatalog.get("coco_val")

train_metadata = MetadataCatalog.get("coco_train")
test_metadata = MetadataCatalog.get("coco_val")

from detectron2.evaluation import COCOEvaluator, inference_on_dataset, DatasetEvaluators
from detectron2.data import build_detection_test_loader
    
# Augmentation list

# aug_list = [
#             # A.Resize(256,256),
#             A.RandomScale(scale_limit=(-0.9, 1), p=1), #LargeScaleJitter from scale of 0.1 to 2
#             # A.RandomScale(scale_limit=(-0.2, 0.25), p=1), #SmallScaleJitter from scale of 0.8 to 1.25
#             A.PadIfNeeded(256, 256, border_mode=0), #pads with image in the center, not the top left like the paper
#             A.RandomCrop(256, 256),
#         CopyPaste(blend=True, sigma=1, pct_objects_paste=1.0, p=1.0) #pct_objects_paste is a guess
#     ]
        
# transform = A.Compose(
#             aug_list, bbox_params=A.BboxParams(format="coco", min_visibility=0.05)
#         )

# data = CocoDetectionCP(
#     coco_path+'/coco/train2017',
#     coco_path+'/coco/annotations/instances_train2017.json',
#     transform
# )

# data_id_to_num = {i:q for q,i in enumerate(data.ids)}
# ALL_IDS = list(data_id_to_num.keys())

# dataset_dicts_train = [i for i in dataset_dicts_train if i['image_id'] in ALL_IDS]
# BOX_MODE = dataset_dicts_train[0]['annotations'][0]['bbox_mode']

class MyMapper:
    """Mapper which uses `detectron2.data.transforms` augmentations"""

    def __init__(self, cfg, is_train: bool = True):

        self.is_train = is_train

        mode = "training" if is_train else "inference"
        #print(f"[MyDatasetMapper] Augmentations used in {mode}: {self.augmentations}")

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        img_id = dataset_dict['image_id']
        
        
        aug_sample = data[data_id_to_num[img_id]]
        
        image = aug_sample['image']

        image =  cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        
        
        bboxes = aug_sample['bboxes']
        box_classes = np.array([b[-2] for b in bboxes])
        # boxes = np.stack([b[:4] for b in bboxes], axis=0)
        mask_indices = np.array([b[-1] for b in bboxes])
        
        masks = aug_sample['masks']
        
        annos = []
        
        for enum,index in enumerate(mask_indices):
            curr_mask = masks[index]
            
            fortran_ground_truth_binary_mask = np.asfortranarray(curr_mask)
            encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
            ground_truth_area = mask.area(encoded_ground_truth)
            ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
            contours = measure.find_contours(curr_mask, 0.5)
            
            annotation = {
        "segmentation": [],
        "iscrowd": 0,
        "bbox": ground_truth_bounding_box.tolist(), 
        "category_id": train_metadata.thing_dataset_id_to_contiguous_id[box_classes[enum]]  ,
        "bbox_mode":BOX_MODE
                
                
    }
            valid = True
            for contour in contours:
                contour = np.flip(contour, axis=1)
                segmentation = contour.ravel().tolist()
                annotation["segmentation"].append(segmentation)
                if len(segmentation) <= 4:
                    valid = False

            if valid:
                annos.append(annotation)
        
        image_shape = image.shape[:2]  # h, w
       
        instances = utils.annotations_to_instances(annos, image_shape)
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg, sampler=None):
        return build_detection_train_loader(
            cfg, mapper=MyMapper(cfg, True), sampler=sampler
        )
    def _test(cls, cfg, model):
        evaluator = [COCOEvaluator("coco_val", ("bbox", "segm"), False, output_dir="./output_val/")]
        res = cls.test(cfg, model, evaluator)
        res = OrderedDict({k:v for k,v in res.items()})
        return res
   

def main(args):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))

    cfg.DATASETS.TRAIN = ("coco_train",)
    cfg.DATASETS.VAL = ("coco_val",)
    cfg.DATASETS.TEST = ("coco_val",)

    cfg.INPUT.FORMAT = 'BGR'

    model = build_model(cfg)
    Checkpointer(model).load("./output_aug_var/1/model_0001999.pth")
    
    evaluator = [COCOEvaluator("coco_val", ("bbox", "segm"), False, output_dir="./output_val/")]

    val_loader = build_detection_test_loader(cfg, "coco_val")
    print(inference_on_dataset(model, val_loader, evaluator))


    return

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )