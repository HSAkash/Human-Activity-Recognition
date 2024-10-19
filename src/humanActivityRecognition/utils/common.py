import os
from box.exceptions import BoxValueError
import yaml
from humanActivityRecognition import logger
from humanActivityRecognition.entity.config_entity import HistoryModelConfig
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64
import pandas as pd



@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def save_json(path: Path, data: dict, verbose=True):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    if verbose:
        logger.info(f"json file saved at: {path}")




@ensure_annotations
def load_json(path: Path, verbose=True) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)
    if verbose:
        logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def load_history(history_path: Path) -> HistoryModelConfig:
    """load history model

    Args:
        path (Path): path to history model

    Returns:
        ConfigBox: model history data
    """
    history = pd.read_csv(history_path)
    loss = history['loss'].values
    val_loss = history['val_loss'].values

    accuracy = history['accuracy'].values
    val_accuracy = history['val_accuracy'].values

    epochs = history['epoch'].values

    return HistoryModelConfig(
        epoch=epochs,
        loss=loss,
        val_loss=val_loss,
        accuracy=accuracy,
        val_accuracy=val_accuracy
    )








