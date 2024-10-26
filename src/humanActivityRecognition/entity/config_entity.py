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
class FeatureExtractionConfig:
    model_weights           : str
    blured_aug_dir          : Path
    blured_feature_dir      : Path
    feature_format          : str
    image_format            : str
    IMAGE_HEIGHT            : int
    IMAGE_WIDTH             : int

@dataclass(frozen=True)
class FinalDatasetConfig:
    blured_feature_dir      : Path
    keypoint_aug_dir        : Path
    blured_final_dir        : Path
    keypoint_final_dir      : Path
    data_format             : str
    IMAGE_HEIGHT            : int
    IMAGE_WIDTH             : int
    MAX_WORKERS             : int

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir                            : Path
    base_model_path                     : Path
    model_architecture_plot_path        : Path
    blured_final_dir                    : Path
    keypoint_final_dir                  : Path
    SEED                                : int

@dataclass(frozen=True)
class PrepareCallbacksConfig:
    root_dir                : Path
    best_checkpoint_path    : Path
    checkpoint_path         : Path
    history_path            : Path
    VERBOSE                 : int

@dataclass(frozen=True)
class PrepareDatasetConfig:
    blured_final_dir        : Path
    keypoint_final_dir      : Path
    SEED                    : int
    BATCH_SIZE              : int

@dataclass(frozen=True)
class TrainingConfig:
    base_model_path         : Path
    history_path            : Path
    checkpoint_path         : Path
    loss_curve_path         : Path
    accuracy_curve_path     : Path
    EPOCHS                  : int
    BATCH_SIZE              : int
    SEED                    : int
    VERBOSE                 : int
    SAVE_PLOTS              : bool
