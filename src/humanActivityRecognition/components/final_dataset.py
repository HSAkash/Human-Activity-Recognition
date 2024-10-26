import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path
from humanActivityRecognition import logger
from humanActivityRecognition.entity.config_entity import FinalDatasetConfig
from concurrent.futures import ThreadPoolExecutor, as_completed



class FinalDataset:
    def __init__(self, config: FinalDatasetConfig):
        self.config = config

    
    def save_data(self, data: np.ndarray, data_path: Path, root_dir: Path, dest_dir: Path):
        """
        Save the data to the disk.
        Args:
            data: The data to be saved.
            data_path: The path of the data.
            root_dir: The root directory of the data.
            dest_dir: The destination directory of the data.
        """
        os.makedirs(
            os.path.dirname(
                data_path.replace(root_dir, dest_dir)
            ),
            exist_ok=True
        )

        np.save(data_path.replace(root_dir, dest_dir), data)

    def merge_normalize_keypoints(self, keypoint_dir: Path ):
        """
        Merge and normalize the keypoints.
        Args:
            keypoint_paths: The paths of the keypoints.
        """
        keypoints = []

        keypoint_paths = sorted(glob(os.path.join(keypoint_dir, f"*.{self.config.data_format}")))

        for keypoint_path in keypoint_paths:
            keypoint = np.load(keypoint_path)
            keypoint[:,:,0] = keypoint[:,:,0] / self.config.IMAGE_WIDTH
            keypoint[:,:,1] = keypoint[:,:,1] / self.config.IMAGE_HEIGHT
            keypoints.append(keypoint.flatten())

        keypoints = np.array(keypoints, dtype=np.float32)

        self.save_data(keypoints, f"{keypoint_dir}.{self.config.data_format}", self.config.keypoint_aug_dir, self.config.keypoint_final_dir)

    def create_keypoint_final_dataset(self):
        """Create keypoint final dataset.
            Sequential keypoints merge together.
        """

        logger.info("Creating keypoint final dataset...")

        keypoint_dirs = glob(
            os.path.join(self.config.keypoint_aug_dir, "*","*","*")
        )

        tasks = []

        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            for keypoint_dir in keypoint_dirs:
                if os.path.exists(
                    f"{keypoint_dir}.{self.config.data_format}".replace(self.config.keypoint_aug_dir, self.config.keypoint_final_dir)
                ):
                    continue

                tasks.append(
                    executor.submit(self.merge_normalize_keypoints, keypoint_dir)
                )

            for future in tqdm(as_completed(tasks), total=len(tasks), desc="Merging keypoints"):
                future.result()

        logger.info("Keypoint final dataset created successfully.")


    
    def merge_image_features(self, feature_dir: Path):
        """
        Merge the image features.
        Args:
            feature_dir: The directory of the features.
        """
        features = []

        feature_paths = sorted(glob(os.path.join(feature_dir, f"*.{self.config.data_format}")))

        for feature_path in feature_paths:
            feature = np.load(feature_path)
            features.append(feature)

        features = np.array(features, dtype=np.float32)

        self.save_data(features, f"{feature_dir}.{self.config.data_format}", self.config.blured_feature_dir, self.config.blured_final_dir)


    def create_blured_final_dataset(self):
        """Create blured final dataset.
            Sequential features merge together.
        """

        logger.info("Creating blured final dataset...")

        feature_dirs = glob(
            os.path.join(self.config.blured_feature_dir, "*","*","*")
        )

        tasks = []

        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            for feature_dir in feature_dirs:
                if os.path.exists(
                    f"{feature_dir}.{self.config.data_format}".replace(self.config.blured_feature_dir, self.config.blured_final_dir)
                ):
                    continue

                tasks.append(
                    executor.submit(self.merge_image_features, feature_dir)
                )

            for future in tqdm(as_completed(tasks), total=len(tasks), desc="Merging features"):
                future.result()

        logger.info("Blured final dataset creation completed.")
        



    def run(self):
        """Run the final dataset creation pipeline."""
        self.create_keypoint_final_dataset()
        self.create_blured_final_dataset()


