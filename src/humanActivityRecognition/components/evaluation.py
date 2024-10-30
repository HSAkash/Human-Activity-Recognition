import os
import numpy as np
from humanActivityRecognition import logger
from humanActivityRecognition.entity.config_entity import EvaluationConfig
from humanActivityRecognition.utils.helperFunction import make_confusion_matrix, getPrescisionRecallF1
import tensorflow as tf


class Evaluation:
    def __init__(self, config: EvaluationConfig, test_ds):
        self.config = config
        self.model = self._load_model()
        self.test_ds = test_ds
        self._get_class_name()

    def _get_class_name(self):
        """
        Get the class names and the number of classes in the dataset.
        """
        self.class_names = sorted([d for d in os.listdir(self.config.blured_final_dir+"/train") if os.path.isdir(os.path.join(self.config.blured_final_dir+"/train", d))])


    def _load_model(self):
        """
        Load the saved best performance model from the specified path.
        """
        if not os.path.exists(self.config.best_model_path):
            logger.info("No saved model found. Initializing a new model.")
            raise FileNotFoundError(f"Model not found: {self.config.best_model_path}")
        return tf.keras.models.load_model(self.config.best_model_path)
    
    def _evaluate_model(self):
        """
        Evaluate the model on the test dataset.
        """
        logger.info(f"Evaluating model...")
        results = self.model.evaluate(self.test_ds, verbose=self.config.VERBOSE)
        logger.info(f"Model evaluation completed: {results}")

    def run(self):
        """
        Run the evaluation pipeline.
        """
        self._evaluate_model()

        logger.info("Generating confusion matrix...")
        y_pred = self.model.predict(self.test_ds, verbose=self.config.VERBOSE)
        y_true = np.concatenate([y for x, y in self.test_ds], axis=0)
        y_true = np.argmax(y_true, axis=-1)
        y_pred = np.argmax(y_pred, axis=-1)


        make_confusion_matrix(
            y_true, y_pred,
            classes=self.class_names,
            savefig=True,
            save_path=self.config.confusion_matrix_path)
        

        # Generate classification report
        result = getPrescisionRecallF1(y_true, y_pred, class_names=["class_01 class_01 class_01","class_02 class_02 class_02"])
        logger.info("Confusion matrix and classification report generated.")
        logger.info(f"Classification report: {result}")
        with open(self.config.classification_report_path, 'w') as f:
            f.write(result)
        logger.info(f"Classification report saved at: {self.config.classification_report_path}")