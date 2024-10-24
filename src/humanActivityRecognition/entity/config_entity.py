from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class ImageExtractionConfig:
    root_dir : Path
    source_dir : Path
    destination_dir : Path
    image_format : str
    IMAGE_HEIGHT : int
    IMAGE_WIDTH : int
    SEQUENCE_LENGTH : int
    MAX_WORKERS : int


@dataclass(frozen=True)
class HistoryModelConfig:
    epoch: np.array
    loss: np.array
    val_loss: np.array
    accuracy: np.array
    val_accuracy: np.array
