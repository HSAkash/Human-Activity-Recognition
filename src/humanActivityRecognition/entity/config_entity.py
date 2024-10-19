from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class HistoryModelConfig:
    epoch: np.array
    loss: np.array
    val_loss: np.array
    accuracy: np.array
    val_accuracy: np.array
