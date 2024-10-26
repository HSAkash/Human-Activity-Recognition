import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, GRU, Dropout, Dense, concatenate
from tensorflow.keras.models import Model

class CombinedModel(tf.keras.Model):
    def __init__(self, keypoints_shape:tuple, img_feature_shape:tuple, num_classes:int):
        super(CombinedModel, self).__init__()
        """
        Initialize the combined model with the given keypoints and image feature shapes and number of classes.

        Args:
            keypoints_shape: Shape of the keypoints input data.
            img_feature_shape: Shape of the image feature input data.
            num_classes: Number of classes in the dataset.
        """


        # Keypoint model layers
        self.keypoint_input = Input(shape=keypoints_shape)
        self.keypoint_lstm1 = LSTM(64, return_sequences=True, activation='relu')
        self.keypoint_lstm2 = LSTM(128, return_sequences=True, activation='relu')
        self.keypoint_lstm3 = LSTM(64, return_sequences=False, activation='relu')
        self.keypoint_dense1 = Dense(64, activation='relu')
        self.keypoint_dense2 = Dense(32, activation='relu')
        
        # Image model layers
        self.img_input = Input(shape=img_feature_shape)
        self.img_lstm = LSTM(64, return_sequences=True)
        self.img_gru = GRU(8)
        self.img_dropout = Dropout(0.4)
        self.img_dense = Dense(8, activation='relu')
        
        # Final dense layer for combined output
        self.final_dense = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        # Separate the inputs for the two sub-models
        keypoint_data, img_data = inputs
        
        # Keypoint model forward pass
        x_key = self.keypoint_lstm1(keypoint_data)
        x_key = self.keypoint_lstm2(x_key)
        x_key = self.keypoint_lstm3(x_key)
        x_key = self.keypoint_dense1(x_key)
        key_output = self.keypoint_dense2(x_key)
        
        # Image model forward pass
        x_img = self.img_lstm(img_data)
        x_img = self.img_gru(x_img)
        x_img = self.img_dropout(x_img)
        img_output = self.img_dense(x_img)
        
        # Concatenate outputs and pass through the final dense layer
        combined_output = concatenate([img_output, key_output])
        output = self.final_dense(combined_output)
        
        return output





