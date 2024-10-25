from humanActivityRecognition.constants import (
    CONFIG_FILE_PATH,
    PARAMS_FILE_PATH
)
import os
from pathlib import Path
from humanActivityRecognition.utils.common import read_yaml, create_directories
from humanActivityRecognition.entity.config_entity import (
    ImageExtractionConfig,
    KeypointDetectionConfig,
    BluringImageConfig,
    SplitingDatasetConfig,
)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> ImageExtractionConfig:
        config = self.config.image_extraction

        create_directories([config.root_dir])

        image_extraction_config = ImageExtractionConfig(
            root_dir = config.root_dir,
            source_dir = config.source_dir,
            destination_dir = config.destination_dir,
            image_format = config.image_format,
            IMAGE_HEIGHT = self.params.IMAGE_HEIGHT,
            IMAGE_WIDTH = self.params.IMAGE_WIDTH,
            SEQUENCE_LENGTH = self.params.SEQUENCE_LENGTH,
            MAX_WORKERS = self.params.MAX_WORKERS
        )

        return image_extraction_config
    
    def get_keypoint_detection_config(self) -> KeypointDetectionConfig:
        config = self.config.keypoint_detection

        create_directories([config.keypoint_dir, config.box_dir])

        keypoint_detection_config = KeypointDetectionConfig(
            root_dir = config.root_dir,
            image_dir = config.image_dir,
            keypoint_dir = config.keypoint_dir,
            box_dir = config.box_dir,
            yolo_model_path = config.yolo_model_path,
            image_format = config.image_format,
            keypoint_format = config.keypoint_format
        )

        return keypoint_detection_config
    
    def get_bluring_image_config(self) -> BluringImageConfig:
        config = self.config.bluring_image

        create_directories([config.blured_image_dir])

        bluring_image_config = BluringImageConfig(
            root_dir = config.root_dir,
            image_dir = config.image_dir,
            box_dir = config.box_dir,
            blured_image_dir = config.blured_image_dir,
            image_format = config.image_format,
            box_format = config.box_format,
            BLUR_STRENGTH = self.params.BLUR_STRENGTH,
            MAX_WORKERS = self.params.MAX_WORKERS
        )

        return bluring_image_config
    
    def get_spliting_dataset_config(self) -> SplitingDatasetConfig:
        config = self.config.spliting_dataset

        create_directories([config.keypoint_split_dir, config.blured_split_dir])

        spliting_dataset_config = SplitingDatasetConfig(
            blured_image_dir = config.blured_image_dir,
            keypoint_dir = config.keypoint_dir,
            keypoint_split_dir = config.keypoint_split_dir,
            blured_split_dir = config.blured_split_dir,
            split_dir_dict_path = config.split_dir_dict_path,
            TRAIN_RATION = self.params.TRAIN_RATION,
            SEED = self.params.SEED
        )

        return spliting_dataset_config
