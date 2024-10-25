import os
import cv2
import shutil
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path
from humanActivityRecognition import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from humanActivityRecognition.entity.config_entity import DataAugmentationConfig


class KeypointAugmentation:
    def __init__(self, config: DataAugmentationConfig):
        self.config = config

    def rotate_keypoints(self, keypoints, angle):
        """
        Rotate keypoints by the specified angle.

        Args:
            keypoints: The input keypoints as a NumPy array.
            angle: The angle to rotate the keypoints.

        Returns:
            The rotated keypoints.
        """
        # Convert angle to radians for math operations
        angle_rad = np.radians(angle)

        # Rotation matrix
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        # Calculate image center
        center_x = self.config.IMAGE_WIDTH / 2
        center_y = self.config.IMAGE_HEIGHT / 2

        # Translate keypoints to the origin (center of the image)
        translated_keypoints = keypoints[:, :, :2] - np.array([center_x, center_y])

        # Apply rotation using the rotation matrix
        rotated_x = translated_keypoints[:, :, 0] * cos_angle - translated_keypoints[:, :, 1] * sin_angle
        rotated_y = translated_keypoints[:, :, 0] * sin_angle + translated_keypoints[:, :, 1] * cos_angle

        # Combine rotated coordinates
        rotated_keypoints = np.stack([rotated_x, rotated_y], axis=-1)

        # Translate back to the original coordinates
        rotated_keypoints += np.array([center_x, center_y])

        # Clip the rotated coordinates to stay within the image boundaries
        rotated_keypoints[:, :, 0] = np.clip(rotated_keypoints[:, :, 0], 0, self.config.IMAGE_WIDTH - 1)
        rotated_keypoints[:, :, 1] = np.clip(rotated_keypoints[:, :, 1], 0, self.config.IMAGE_HEIGHT - 1)

        # Update the keypoints array
        keypoints[:, :, :2] = rotated_keypoints
                
        return keypoints
    
    def flip_keypoints(self, keypoints):
        """
        Flip keypoints horizontally.

        Args:
            keypoints: The input keypoints as a NumPy array.

        Returns:
            The flipped keypoints.
        """
        keypoints[:, :, 0] = self.config.IMAGE_WIDTH - keypoints[:, :, 0]
        return keypoints
    
    def scale_keypoints(self, keypoints, scale_factor):
        """
        Scale keypoints by the specified factor.

        Args:
            keypoints: The input keypoints as a NumPy array.
            scale_factor: The factor to scale the keypoints.

        Returns:
            The scaled keypoints.
        """
        # Calculate image center
        center_x = self.config.IMAGE_WIDTH / 2
        center_y = self.config.IMAGE_HEIGHT / 2

        # Translate keypoints to the origin (center of the image)
        translated_keypoints = keypoints[:, :, :2] - np.array([center_x, center_y])

        # Scale keypoints around the center
        scaled_keypoints = translated_keypoints * scale_factor

        # Translate back to original coordinates
        scaled_keypoints += np.array([center_x, center_y])

        # Clip the scaled coordinates to stay within the image boundaries
        scaled_keypoints[:, :, 0] = np.clip(scaled_keypoints[:, :, 0], 0, self.config.IMAGE_WIDTH - 1)
        scaled_keypoints[:, :, 1] = np.clip(scaled_keypoints[:, :, 1], 0, self.config.IMAGE_HEIGHT - 1)

        # Update the keypoints array
        keypoints[:, :, :2] = scaled_keypoints

        return keypoints
    
    def get_dest_path(self, keypoints_path: Path, source_dir: Path, dist_dir: Path, aug_type: str, extra_arg: str=''):
        """
        Get the destination path for the augmented keypoints.

        Args:
            keypoints_path: The path to the keypoints file.
            source_dir: The source directory.
            dist_dir: The destination directory.
            aug_type: The augmentation type.
            extra_arg: The extra argument for the destination path.

        Returns:
            The destination path for the augmented keypoints.
        """
        dist_path = keypoints_path.replace(source_dir, dist_dir)
        dist_path = dist_path.split('/')
        dist_path[-2] = f"{dist_path[-2]}_{aug_type}_{extra_arg}"
        return '/'.join(dist_path)
    
    def augment_keypoints(self, keypoints_path: Path, source_dir: Path, dist_dir: Path):
        """
        Augment the keypoints.

        Args:
            keypoints_path: The path to the keypoints file.
            source_dir: The source directory.
            dist_dir: The destination directory.
        """
        # check if the image is already augmented
        dest_path = self.get_dest_path(
                keypoints_path=keypoints_path,
                source_dir=self.config.keypoint_split_dir,
                dist_dir=self.config.keypoint_aug_dir,
                aug_type='flip')
        if os.path.exists(dest_path):
            return


        # Load the keypoints
        keypoints = np.load(keypoints_path)
        # Rotate the keypoints
        for angle in self.config.ROTATE_FACTORS:
            temp_keypoints = keypoints.copy()
            rotated_keypoints = self.rotate_keypoints(temp_keypoints, angle)
            dest_path = self.get_dest_path(keypoints_path, source_dir, dist_dir, 'rotate', str(angle))
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            np.save(dest_path, rotated_keypoints)

        # Scale the keypoints
        for scale_factor in self.config.SCALE_FACTORS:
            temp_keypoints = keypoints.copy()
            scaled_keypoints = self.scale_keypoints(temp_keypoints, scale_factor)
            dest_path = self.get_dest_path(keypoints_path, source_dir, dist_dir, 'scale', str(scale_factor))
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            np.save(dest_path, scaled_keypoints)

        # Flip the keypoints
        if self.config.FLIP_FACTOR:
            temp_keypoints = keypoints.copy()
            flipped_keypoints = self.flip_keypoints(temp_keypoints)
            dest_path = self.get_dest_path(keypoints_path, source_dir, dist_dir, 'flip')
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            np.save(dest_path, flipped_keypoints)

    def run(self):
        """
        Run the keypoint augmentation.
        """
        # Get the list of keypoints files
        keypoints_files = glob(os.path.join(self.config.keypoint_split_dir, 'train','*','*', '*'))

        # Augment the keypoints
        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            futures = [executor.submit(self.augment_keypoints, keypoints_path, self.config.keypoint_split_dir, self.config.keypoint_aug_dir) for keypoints_path in keypoints_files]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Augmenting keypoints"):
                future.result()

        shutil.copytree(self.config.keypoint_split_dir, self.config.keypoint_aug_dir, dirs_exist_ok=True)

        logger.info("Keypoint augmentation completed.")