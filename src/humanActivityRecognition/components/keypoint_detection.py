from tqdm import tqdm
from ultralytics import YOLO
import numpy as np
from glob import glob
import os
from pathlib import Path
from humanActivityRecognition import logger
from humanActivityRecognition.entity.config_entity import KeypointDetectionConfig


class KeypointDetection:
    def __init__(self, config: KeypointDetectionConfig):
        self.config = config
        self.yolo = YOLO(self.config.yolo_model_path)


    def detect_keypoints(self, images):
        all_keypoints = []
        all_boxes = []
        # Detect the person in the image
        results = self.yolo(images, verbose=False)
        for result in results:
            # Get the boxes
            boxes = result.boxes.data.cpu().numpy().astype(np.int16)
            if boxes.shape[0] !=0 and len(boxes.shape)==2 and boxes.shape[1]==6:
                boxes = result.boxes.data.cpu()[:2,:4].numpy().astype(np.int16)
            else:
                boxes = np.array([], dtype=np.int16)
            all_boxes.append(boxes)
            # Get the keypoints
            keypoints = result.keypoints.data.cpu().numpy()
            if keypoints.shape[1:] != (17, 3):
                keypoints = np.zeros((2,17,3), dtype=np.int16)
            elif keypoints.shape[0] == 1:
                keypoints = np.concatenate((keypoints, np.zeros((1,17,3), dtype=np.int16)))
            else:
                keypoints = keypoints[:2]
            all_keypoints.append(keypoints)

        return all_keypoints, all_boxes

    def detect_keypoints_from_images(self, image_dir):
        # Get the image paths
        image_paths = sorted(glob(os.path.join(image_dir, f"*.{self.config.image_format}")))
        all_keypoints_paths = [
            x.replace(f".{self.config.image_format}", f".{self.config.keypoint_format}").replace(
                self.config.image_dir, self.config.keypoint_dir) for x in image_paths
            ]
        all_boxes_paths = [
            x.replace(f".{self.config.image_format}", f".{self.config.keypoint_format}").replace(
                self.config.image_dir, self.config.box_dir) for x in image_paths
            ]
        if os.path.exists(all_keypoints_paths[-1]) and os.path.exists(all_boxes_paths[-1]):
            return
        all_keypoints, all_boxes = self.detect_keypoints(image_paths)
        os.makedirs(os.path.dirname(all_keypoints_paths[0]), exist_ok=True)
        os.makedirs(os.path.dirname(all_boxes_paths[0]), exist_ok=True)
        for i in range(len(image_paths)):
            np.save(all_keypoints_paths[i], all_keypoints[i])
            np.save(all_boxes_paths[i], all_boxes[i])

    def run(self):
        image_dirs = glob(os.path.join(self.config.image_dir, "*","*"))
        for image_dir in tqdm(image_dirs, desc="Detecting keypoints"):
            self.detect_keypoints_from_images(image_dir)