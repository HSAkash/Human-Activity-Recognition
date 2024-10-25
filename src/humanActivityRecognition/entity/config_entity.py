from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class ImageExtractionConfig:
    root_dir                : Path
    source_dir              : Path
    destination_dir         : Path
    image_format            : str
    IMAGE_HEIGHT            : int
    IMAGE_WIDTH             : int
    SEQUENCE_LENGTH         : int
    MAX_WORKERS             : int

@dataclass(frozen=True)
class KeypointDetectionConfig:
    root_dir                : Path
    image_dir               : Path
    keypoint_dir            : Path
    box_dir                 : Path
    yolo_model_path         : Path
    image_format            : str
    keypoint_format         : str


@dataclass(frozen=True)
class BluringImageConfig:
    root_dir                : Path
    image_dir               : Path
    box_dir                 : Path
    blured_image_dir        : Path
    image_format            : str
    box_format              : str
    BLUR_STRENGTH           : int
    MAX_WORKERS             : int

@dataclass(frozen=True)
class SplitingDatasetConfig:
    blured_image_dir        : Path
    keypoint_dir            : Path
    keypoint_split_dir      : Path
    blured_split_dir        : Path
    split_dir_dict_path     : Path
    TRAIN_RATION            : float
    SEED                    : int

@dataclass(frozen=True)
class DataAugmentationConfig:
    keypoint_split_dir      : Path
    keypoint_aug_dir        : Path
    blured_split_dir        : Path
    blured_aug_dir          : Path
    ROTATE_FACTORS          : list[int] # [0, 90, 180, 270]
    SCALE_FACTORS           : list[float] # [0.5, 1.0, 1.5]
    FLIP_FACTOR             : bool
    MAX_WORKERS             : int
    IMAGE_HEIGHT            : int
    IMAGE_WIDTH             : int


@dataclass(frozen=True)
class HistoryModelConfig:
    epoch                   : np.array
    loss                    : np.array
    val_loss                : np.array
    accuracy                : np.array
    val_accuracy            : np.array
