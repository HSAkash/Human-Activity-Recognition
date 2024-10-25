import os
import cv2
import shutil
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from humanActivityRecognition import logger
from sklearn.model_selection import train_test_split
from humanActivityRecognition.entity.config_entity import SplitingDatasetConfig

class SplitingDataset:
    def __init__(self, config: SplitingDatasetConfig):
        self.config = config

    
    def load_split_dir_dict(self, root_dir: Path, split_dir_dict_path: Path, for_dest: bool = False):
        split_dir_dict = np.load(split_dir_dict_path, allow_pickle=True).item()
        train_dirs = np.array(split_dir_dict["train"], dtype=object)
        test_dirs = np.array(split_dir_dict["test"], dtype=object)
        
        if for_dest:
            train_dirs = np.array([os.path.join(root_dir, 'train', x) for x in train_dirs], dtype=object)
            test_dirs = np.array([os.path.join(root_dir, 'test', x) for x in test_dirs], dtype=object)
            return train_dirs, test_dirs
        
        train_dirs = np.array([os.path.join(root_dir, x) for x in train_dirs], dtype=object)
        test_dirs = np.array([os.path.join(root_dir, x) for x in test_dirs], dtype=object)
        
        return train_dirs, test_dirs

    def create_train_test_split(self, root_dir:Path, class_names:list[str], split_dir_dict_path:Path):
        """
        This function will split the dataset into training and testing sets.
        Args:
            root_dir: The root directory of the dataset.
            class_names: The names of the classes in the dataset.
            split_dir_dict_path: The path where the split dictionary will be stored.
        Returns:
            split_dir_dict: A dictionary containing the split of the dataset.
        """

        # Check if the split dictionary already exists.
        if os.path.exists(split_dir_dict_path):
            logger.info(f"Split dictionary already exists at {split_dir_dict_path}")
            return

        # If the split dictionary does not exist, create it.
        split_dir_dict = {

            "train": [],

            "test": [],

        }

        for class_name in class_names:
            try:
                temp_dirs = os.listdir(f"{root_dir}/{class_name}")
                temp_dirs = [ os.path.join(class_name, x) for x in temp_dirs]
                np.random.seed(self.config.SEED)
                np.random.shuffle(temp_dirs)
                temp_train_dirs, temp_test_dirs = train_test_split(
                    temp_dirs,
                    train_size=self.config.TRAIN_RATION,
                    random_state=self.config.SEED)
            except Exception as e:
                logger.exception(e)
                logger.error(f"Error splitting {class_name}")
                logger.info("Dataset is not large enough to split")
                exit(1)
            split_dir_dict["train"].extend(temp_train_dirs)
            split_dir_dict["test"].extend(temp_test_dirs)

        np.random.seed(self.config.SEED)

        np.random.shuffle(split_dir_dict["train"])

        np.save(split_dir_dict_path, split_dir_dict, allow_pickle=True)

        logger.info(f"Split dictionary saved at {split_dir_dict_path}")
        return 


    def split_dataset(self, source_dir:Path, destination_dir:Path, split_dir_dict_path:Path):
        """
        This function will split the dataset into training and testing sets.
        Args:
            source_dir: The directory containing the dataset.
            destination_dir: The directory where the dataset will be split.
            split_dir_dict_path: The path where the split dictionary will be stored.
        """
    
        # create the split dictionary
        self.create_train_test_split(
            source_dir,
            os.listdir(source_dir),
            split_dir_dict_path
        )
        # load the split dictionary
        source_train_dir, source_test_dir = self.load_split_dir_dict(
            source_dir,
            split_dir_dict_path
        )

        destination_train_dir, destination_test_dir = self.load_split_dir_dict(
            destination_dir,
            split_dir_dict_path,
            for_dest=True
        )
        # copy the images to the respective directories
        logger.info(f"Splitting {source_dir} dataset into training and testing sets...")
        for src, dest in tqdm(zip(source_train_dir, destination_train_dir), total=len(source_train_dir)):
            shutil.copytree(src, dest, dirs_exist_ok=True)

        for src, dest in tqdm(zip(source_test_dir, destination_test_dir), total=len(source_test_dir)):
            shutil.copytree(src, dest, dirs_exist_ok=True)

    def run(self):
        '''
        This function will split the dataset into training and testing sets.
        '''
        self.split_dataset(self.config.blured_image_dir, self.config.blured_split_dir, self.config.split_dir_dict_path)
        self.split_dataset(self.config.keypoint_dir, self.config.keypoint_split_dir, self.config.split_dir_dict_path)