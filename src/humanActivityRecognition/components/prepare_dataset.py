import os
import numpy as np
from glob import glob
from pathlib import Path
from humanActivityRecognition import logger
from humanActivityRecognition.entity.config_entity import PrepareDatasetConfig
import tensorflow as tf

class PrepareDataset:
    def __init__(self, config: PrepareDatasetConfig):
        self.config = config
        self._get_class_name()

    def _get_class_name(self):
        """
        Get the class names and the number of classes in the dataset.
        """
        self.class_names = sorted([d for d in os.listdir(self.config.blured_final_dir+"/train") if os.path.isdir(os.path.join(self.config.blured_final_dir+"/train", d))])
        self.class_dict = {x:i for i,x in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

    def _get_labels(self, dirs: list[Path]) -> np.array:
        """
        get the labels of the dataset
        args:
            dirs: list of directories containing the dataset files
        returns:
            labels: numpy array containing the labels of the dataset
        """
        labels = np.array([self.class_dict[d.split("/")[-2]] for d in dirs])
        return labels



    def _prepare_data(self):

        def load_npy_files(filePath):
            # Load .npy file and return tensor
            data = np.load(filePath.decode("utf-8"))  # decode the file path to string
            return tf.convert_to_tensor(data, dtype=tf.float32)

        def create_dataset(mesh_files, keypoints_files, labels, batch_size):
            def load_data(mesh_file, keypoints_file, label):
                # Load mesh and keypoints data from files
                mesh_data = tf.numpy_function(load_npy_files, [mesh_file], tf.float32)  
                keypoints_data = tf.numpy_function(load_npy_files, [keypoints_file], tf.float32)    
                # Set the shape explicitly after loading the data
                mesh_data.set_shape((20, 2048))  # Assuming your mesh data has this shape
                keypoints_data.set_shape((20, 102)) # Assuming your keypoints data has this shape
                return (mesh_data, keypoints_data), label

            # Create a dataset from file paths and labels
            dataset = tf.data.Dataset.from_tensor_slices((mesh_files, keypoints_files, labels))
            dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            return dataset
        
        X_train_paths_1 = sorted(glob(self.config.blured_final_dir+"/train/*/*.npy"))
        X_train_paths_2 = sorted(glob(self.config.keypoint_final_dir+"/train/*/*.npy"))
        train_labels = self._get_labels(X_train_paths_1)


        X_test_paths_1 = sorted(glob(self.config.blured_final_dir+"/test/*/*.npy"))
        X_test_paths_2 = sorted(glob(self.config.keypoint_final_dir+"/test/*/*.npy"))
        test_labels = self._get_labels(X_test_paths_1)

        # one hot encode the labels
        y_train = tf.one_hot(train_labels, depth=self.num_classes)
        y_test = tf.one_hot(test_labels, depth=self.num_classes)

        self.train_ds = create_dataset(X_train_paths_1, X_train_paths_2, y_train, self.config.BATCH_SIZE)
        self.test_ds = create_dataset(X_test_paths_1, X_test_paths_2, y_test, self.config.BATCH_SIZE)

    def _show_dataset_shape(self):
        for (feature_data, keypoints_data), labels in self.test_ds.take(1):
            print("feature data shape:", feature_data.shape)
            print("Keypoints data shape:", keypoints_data.shape)
            print("Labels shape:", labels.shape)

    def get_dataset(self):
        self._prepare_data()
        self._show_dataset_shape()
        return self.train_ds, self.test_ds
    
    