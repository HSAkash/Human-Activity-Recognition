import os
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from pathlib import Path
from humanActivityRecognition import logger
from sklearn.preprocessing import LabelBinarizer
from humanActivityRecognition.entity.config_entity import FeatureExtractionConfig


class FeatureExtraction:
    def __init__(self, config: FeatureExtractionConfig):
        self.config = config
        self.get_model()

    def get_model(self):
        """
        Load the InceptionV3 model with imagenet weights.
        """
        model = tf.keras.applications.InceptionV3(
            input_shape=(self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH, 3),
            include_top=False,
            weights=self.config.model_weights,
            pooling='avg'
        )

        model.trainable = False

        preprocess_input = tf.keras.applications.inception_v3.preprocess_input
        inputs = tf.keras.Input((self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH, 3))
        preprocessed = preprocess_input(inputs)
        outputs = model(preprocessed)
        self.model =  tf.keras.Model(inputs, outputs, name="feature_extractor_imgnet")
    

    def save_features(self, features: np.array, image_paths: list[Path]):
        """
        Save the extracted features to the disk.
        Args:
            features: The extracted features.
            image_paths: List of image paths.
        """
        os.makedirs(os.path.dirname(
            image_paths[0].replace(
                self.config.blured_aug_dir, self.config.blured_feature_dir)), exist_ok=True)
        for image_path, feature in zip(image_paths, features):
            feature_path = image_path.replace(
                self.config.blured_aug_dir, self.config.blured_feature_dir).replace(
                    self.config.image_format, self.config.feature_format)
            np.save(feature_path, feature)

    def load_images(self, image_paths: list[Path]):
        """
        Load images from the given paths.
        Args:
            image_paths: List of image paths.
        """
        images = []
        for image_path in image_paths:
            image = Image.open(image_path)
            image = np.asarray(image, dtype=np.float32)
            images.append(image)
        return np.array(images)
    

    def extract_features(self, image_paths: list[Path]) -> np.array:
        """
        Extract features from the given images.
        Args:
            image_paths: List of image paths.
        Returns:
            features: The extracted features.
        """
        images = self.load_images(image_paths)
        features = self.model.predict(images, verbose=0)
        return features
    
    def run(self):
        """
        Run the feature extraction process.
        """
        all_dirs = glob(
            os.path.join(self.config.blured_aug_dir, '*', '*', '*')
        )
        for dir in tqdm(all_dirs):
            image_paths = glob(os.path.join(dir, f'*{self.config.image_format}'))
            image_paths = sorted(image_paths)
            feature_path = image_paths[-1].replace(
                self.config.blured_aug_dir, self.config.blured_feature_dir).replace(
                    self.config.image_format, self.config.feature_format)
            if os.path.exists(feature_path):
                continue
            features = self.extract_features(image_paths)
            self.save_features(features, image_paths)
    

