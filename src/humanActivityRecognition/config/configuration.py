from humanActivityRecognition.constants import (
    CONFIG_FILE_PATH,
    PARAMS_FILE_PATH
)
import os
from pathlib import Path
from humanActivityRecognition.utils.common import read_yaml, create_directories
from humanActivityRecognition.entity.config_entity import (
    ImageExtractionConfig,
    KeypointDetectionConfig
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