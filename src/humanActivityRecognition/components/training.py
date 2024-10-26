import os
import pandas as pd
from glob import glob
from pathlib import Path
from humanActivityRecognition import logger
from humanActivityRecognition.entity.config_entity import TrainingConfig
from humanActivityRecognition.utils.helperFunction import plot_loss_curves_history
import tensorflow as tf

class Training:
    def __init__(self, config: TrainingConfig, callbacks, train_ds, test_ds):
        self.config = config
        self.callbacks = callbacks
        self.train_ds = train_ds
        self.test_ds = test_ds

    def _get_model(self):
        save_model_paths = glob(os.path.join(self.config.checkpoint_path, '*.keras'))
        save_model_paths = sorted(save_model_paths)
        if len(save_model_paths) == 0:
            logger.info("No saved model found. Initializing a new model.")
            return tf.keras.models.load_model(self.config.base_model_path)
        return tf.keras.models.load_model(save_model_paths[-1])
    

    def _get_initial_epoch(self):
        if not os.path.exists(self.config.history_path):
            return 0
        try:
            history = pd.read_csv(self.config.history_path)
        except Exception as e:
            logger.error("Error reading history file: ", str(e))
            return 0
        return max(history['epoch'])+1 if 'epoch' in history.columns else 0


    def _train_model(self):
        history = self.model.fit(
            self.train_ds,
            epochs=self.config.EPOCHS,
            initial_epoch=self.initial_epoch,
            validation_data=self.test_ds,
            callbacks=self.callbacks,
            verbose=self.config.VERBOSE
        )

    def run(self):
        
        # Get initial epoch & save last model state
        self.initial_epoch = self._get_initial_epoch()
        self.model = self._get_model()
        logger.info("Model loaded successfully")
        logger.info(f"Traing resume from epoch: {self.initial_epoch}")


        # Train the model
        logger.info("Training started...")
        self._train_model()
        logger.info("Training completed")

        # Save the loss and accuracy curves
        logger.info("Plotting loss and accuracy curves...")
        plot_loss_curves_history(
            history_path=self.config.history_path,
            save_loss_curves=self.config.loss_curve_path,
            save_accuracy_curves=self.config.accuracy_curve_path,
            save=self.config.SAVE_PLOTS
        )