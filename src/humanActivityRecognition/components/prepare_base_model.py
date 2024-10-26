import os
import numpy as np
from glob import glob
from humanActivityRecognition import logger
from humanActivityRecognition.entity.config_entity import PrepareBaseModelConfig


import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dense, GRU,Dropout, concatenate



class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_image_feature_shape(self) -> tuple:
        """
        Returns the shape of the image feature data.
        """
        feature_file_path = glob(
            os.path.join(self.config.blured_final_dir, '*', '*', '*.npy')
        )[0]
        feature = np.load(feature_file_path)
        return feature.shape
    
    def get_keypoints_shape(self) -> tuple:
        """
        Returns the shape of the keypoint data.
        """
        keypoint_file_path = glob(
            os.path.join(self.config.keypoint_final_dir, '*', '*', '*.npy')
        )[0]
        keypoints = np.load(keypoint_file_path)
        return keypoints.shape
    
    def get_num_classes(self) -> int:
        """
        Returns the number of classes in the dataset.
        """
        return len(os.listdir(os.path.join(self.config.blured_final_dir, 'train')))
    

    def get_model(self, keypoints_shape, img_feature_shape, num_classes):
        """
        Returns the compiled combined model.
        args:
            keypoints_shape: Shape of the keypoints input data.
            img_feature_shape: Shape of the image feature input data.
            num_classes: Number of classes in the dataset.
        """

        if os.path.exists(self.config.base_model_path):
            return tf.keras.models.load_model(self.config.base_model_path)
        tf.random.set_seed(self.config.SEED)

        key_input = Input(shape=keypoints_shape)
        x = LSTM(64, return_sequences=True, activation='relu')(key_input)
        x = LSTM(128, return_sequences=True, activation='relu')(x)
        x = LSTM(64, return_sequences=False, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        key_output = x
        model_keypoint = Model(inputs=key_input, outputs=key_output)

        # Define the image model
        img_input = Input(shape=img_feature_shape)
        x = LSTM(64, return_sequences=True)(img_input)
        x = GRU(8)(x)
        x = Dropout(0.4)(x)
        img_output = Dense(8, activation="relu")(x)
        model_img = Model(inputs=img_input, outputs=img_output)

        # Concatenate the outputs of both models
        combined_output = concatenate([model_img.output, model_keypoint.output])

        # Add a Dense layer for the final prediction
        output = Dense(num_classes, activation="softmax")(combined_output)

        # Create the combined model
        model = Model(
            inputs=[
                model_img.input,
                model_keypoint.input
            ],
            outputs=output
        )
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def run(self):

        keypoints_shape = self.get_keypoints_shape()
        img_feature_shape = self.get_image_feature_shape()
        num_classes = self.get_num_classes()

        # Create the combined model using the specified shapes and number of classes
        model = self.get_model(keypoints_shape, img_feature_shape, num_classes)
        logger.info(f"""Model summary {model.summary()}""")

        os.makedirs(os.path.dirname(self.config.model_architecture_plot_path), exist_ok=True)

        # Save the model architecture as a plot
        if not os.path.exists(self.config.model_architecture_plot_path):
            tf.keras.utils.plot_model(
                model,
                to_file=self.config.model_architecture_plot_path,
                show_shapes=True,
                show_dtype=False,
                show_layer_names=True,
                rankdir='TB',
                expand_nested=False,
                dpi=96,
                layer_range=None,
                show_layer_activations=False,
                show_trainable=False
            )

        # Save the model
        if os.path.exists(self.config.base_model_path):
            logger.info("Base model already exists. Skipping the model creation step.")
            return
        model.save(self.config.base_model_path)

        logger.info(f"Model architecture saved to {self.config.model_architecture_plot_path}")
        logger.info(f"Model saved to {self.config.base_model_path}")

