import os
from glob import glob
from pathlib import Path
from humanActivityRecognition import logger
from humanActivityRecognition.entity.config_entity import PrepareCallbacksConfig
import tensorflow as tf

class PrepareCallbacks:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config

    @property
    def _create_best_model_check(self):
        return tf.keras.callbacks.ModelCheckpoint(
            self.config.best_checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True
        )
    
    @property
    def _create_checkpoint_model(self):
        return tf.keras.callbacks.ModelCheckpoint(
            os.path.join(self.config.checkpoint_path, 'model_{epoch:010d}.keras'),
            save_freq='epoch'
        )
    
    @property
    def _create_reduce(self):
        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.1,
            patience=5,
            verbose=self.config.VERBOSE,
            min_lr=1e-7
        )
    
    @property
    def _create_history(self):
        os.makedirs(os.path.dirname(self.config.history_path), exist_ok=True)
        return tf.keras.callbacks.CSVLogger(self.config.history_path, append=True)

    def create_callbacks(self) -> list:
        os.makedirs
        return [
            self._create_best_model_check,
            self._create_checkpoint_model,
            self._create_reduce,
            self._create_history
        ]

