# run from detectron2/detectron2 directory
#from detectron2.data.datasets.coco import convert_to_coco_json

import torch
assert torch.__version__.startswith("1.7")
import argparse

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation.coco_evaluation import COCOEvaluator

# GET FOLDER TO RUN DETECTRON ON WITHIN annotation_tools/image_datasets
parser = argparse.ArgumentParser()
parser.add_argument('image_folder', type=str, help="Path to Folder? Folder must be in image_datasets")
parser.add_argument('output_json', type=str, help="Path to json output file? File will be output in json_datasets")
parser.add_argument('model_yaml', type=str, help="Model Yaml from Model Zoo - https://detectron2.readthedocs.io/en/latest/_modules/detectron2/model_zoo/model_zoo.html")
args = parser.parse_args()
model_yaml = args.model_yaml

# keypoint model = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml"

# CONFIGURE MODEL
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file(model_yaml))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.DEVICE = "cpu"
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_yaml)
predictor = DefaultPredictor(cfg)

# DATASET FORMAT FOR VISIPEDIA
images = []
licenses = []
annotations = []
categories = [{"keypoints_style": ["#e6194b", "#3cb44b", "#ffe119", "#0082c8", "#f58231", "#911eb4", "#46f0f0", "#f032e6",
"#d2f53c", "#fabebe", "#008080", "#e6beff", "#aa6e28", "#fffac8", "#800000", "#aaffc3", "#808000"], "skeleton": [[16, 14],
[14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
[2, 4], [3, 5], [4, 6], [5, 7]], "supercategory": "person", "keypoints": ["nose", "left_eye", "right_eye", "left_ear",
"right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip",
"right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"], "id": "1", "name": "person"}]

# INITIALIZE IMAGE AND ANNOTATION ID COUNTER
image_ctr = 0
annotation_ctr = 0
kp_indices = {"nose": 0, "left_eye": 1, "right_eye": 3, "left_ear": 4,
"right_ear": 5, "left_shoulder": 6, "right_shoulder": 7, "left_elbow": 8,
"right_elbow": 9, "left_wrist": 10, "right_wrist": 11, "left_hip": 12,
"right_hip": 13, "left_knee": 14, "right_knee": 15, "left_ankle": 16, "right_ankle": 17}
kp_threshes = [0.5, 0.4, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1]

# LOOP THROUGH FOLDER, RUN DETECTRON
directory = args.image_folder #directory in ../../annotation_tools/
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        # INITIALIZE IMAGE COUNTER
        image_ctr += 1

        # READ IMAGE AND PREDICT
        im = cv2.imread(os.path.join(directory, filename))
        height, width, channels = im.shape
        outputs = predictor(im)
        print("Images complete: {}".format(image_ctr))

        #ADD ANNOTATIONS TO ANNOTATIONS LIST
        boxes_tensor = list(outputs["instances"].pred_boxes)
        keypoints_tensor = list(outputs["instances"].get_fields()["pred_keypoints"])

        for i, bbox in enumerate(boxes_tensor):
            ann = {}
            annotation_ctr += 1
            ann["id"] = str(annotation_ctr)
            ann["image_id"] = str(image_ctr)
            ann["category_id"] = "1"
            d2_bbox = boxes_tensor[i].tolist()
            coco_bbox = [d2_bbox[0], d2_bbox[1], d2_bbox[2] - d2_bbox[0], d2_bbox[3] - d2_bbox[1]]
            ann["bbox"] = coco_bbox
            list_kp_list = keypoints_tensor[i].tolist()

            for n, kp in enumerate(list_kp_list):
                if kp[2] < kp_threshes[n]:
                    list_kp_list[n][0] = 0
                    list_kp_list[n][1] = 0
                    list_kp_list[n][2] = 0
                else:
                    list_kp_list[n][2] = 2
            list_kp_list = [item for sublist in list_kp_list for item in sublist]
            ann["keypoints"] = list_kp_list
            annotations.append(ann)

        # print(keypoints_tensor)
        # print(ann["keypoints"])


        # APPEND IMAGES TO IMAGE DICT
        image_dict = {}
        image_dict["url"] = "http://localhost:6008/image_datasets/" + str(args.image_folder).split("/")[-1] + "/" + filename
        image_dict["file_name"] = filename
        image_dict["rights_holder"] = ""
        image_dict["id"] = str(image_ctr)
        image_dict["license"] = "1"
        image_dict["width"] = width
        image_dict["height"] = height
        images.append(image_dict)

        # # DISPLAY ANNOTATIONS ON IMAGE
        # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2.imshow("", out.get_image()[:, :, ::-1])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


# APPEND FIELDS TO DATASET DICT
silver_dataset = {}
silver_dataset["images"] = images
silver_dataset["licenses"] = licenses
silver_dataset["annotations"] = annotations
silver_dataset["categories"] = categories

# WRITE DATASET DICT TO JSON
silver_dataset_path = args.output_json
with open(silver_dataset_path, 'w') as f:
    json.dump(silver_dataset, f, indent=4)


#convert_to_coco_json(cfg.DATASETS.TRAIN[0], 'test_d2_coco', allow_cached=False)
