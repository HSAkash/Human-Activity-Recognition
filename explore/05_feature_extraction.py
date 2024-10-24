# import tensorflow as tf
# import numpy as np
# from glob import glob
# from tqdm.auto import tqdm
# from sklearn.preprocessing import LabelBinarizer
# from PIL import Image
# import os
import gdown
import zipfile

aug_seg_bin_url = "https://drive.google.com/file/d/1-1DpWRCnXLjVmek41ufd4FWbjdnhJVgy/view?usp=sharing"
aug_sen_rgb_url = "https://drive.google.com/file/d/1erUQxsy35emFLkKUKYtwsbvm9xj7yfeF/view?usp=sharing"

def download_and_extract(url, output):
    # resume able
    gdown.download(url, output, quiet=False, fuzzy=True)
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(output.parent)


download_and_extract(aug_seg_bin_url, "aug_seg_bin.zip")
download_and_extract(aug_sen_rgb_url, "aug_sen_rgb.zip")