import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path
from humanActivityRecognition import logger
from humanActivityRecognition.entity.config_entity import BluringImageConfig
from concurrent.futures import ThreadPoolExecutor, as_completed

class BluringImage:
    def __init__(self, config: BluringImageConfig):
        self.config = config

    def blur_outside_boxes(self, image, boxes, blur_strength=60):
        """
        Blurs the area outside the specified bounding boxes.
        
        Args:
            image: The input image as a NumPy array.
            boxes: A list of bounding boxes, each specified as [x1, y1, x2, y2].
            blur_strength: The strength of the blur, must be an odd number (default is 51).
            
        Returns:
            The image with blurred areas outside the bounding boxes.
        """
        height, width = image.shape[:2]

        # Ensure blur_strength is odd (required for GaussianBlur)
        if blur_strength % 2 == 0:
            blur_strength += 1

        # Create a mask with the same size as the image, initialized to zeros (black)
        mask = np.zeros((height, width), dtype=np.uint8)

        # Fill the bounding boxes in the mask with white (255)
        for box in boxes:
            x1, y1, x2, y2 = box
            mask[y1:y2, x1:x2] = 255

        # Blur the entire image with the specified blur strength
        blurred_image = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)

        # Create the inverse of the mask
        inverse_mask = cv2.bitwise_not(mask)

        # Use the mask to keep the regions inside the bounding boxes clear
        result = cv2.bitwise_and(image, image, mask=mask)

        # Use the inverse mask to keep the blurred areas outside the boxes
        blurred_outside = cv2.bitwise_and(blurred_image, blurred_image, mask=inverse_mask)

        # Combine the result: clear inside boxes + blurred outside
        final_image = cv2.add(result, blurred_outside)

        return final_image
    

    def blur_process(self, img_path, box_path):
        '''
        This function will blur the images outside the bounding boxes.
        Args:
            img_path: The path of the image.
            box_path: The path of the bounding box file.
        '''
        save_path = img_path.replace(self.config.image_dir, self.config.blured_image_dir)

        if os.path.exists(save_path):
            return
        # Read the image using OpenCV.
        image = cv2.imread(img_path)
        
        # Read the bounding box file (.npy).
        boxes = np.load(box_path)

        # Blur the image outside the bounding boxes.
        blured_image = self.blur_outside_boxes(image, boxes, self.config.BLUR_STRENGTH)

        # Save the blured image to the disk.
        cv2.imwrite(save_path, blured_image)


    def run(self):
        '''
        This function will blur the images outside the bounding boxes.
        '''
        # Get the list of image files.
        image_files = glob(
            os.path.join(
                self.config.image_dir,
                '*','*',
                f'*.{self.config.image_format}'))
        # Get the list of bounding box files.
        box_files = glob(
            os.path.join(
                self.config.box_dir,
                '*','*',
                f'*.{self.config.box_format}'))

        # Get the list of image directories.
        image_dirs = glob(
            os.path.join(self.config.image_dir, '*', '*'))
        # Create the destination directory.
        for image_dir in image_dirs:
            blured_image_dir = image_dir.replace(self.config.image_dir, self.config.blured_image_dir)
            os.makedirs(blured_image_dir, exist_ok=True)

        # Create a list to store the tasks.
        tasks = []
        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            # Iterate over each image file.
            for image_file, box_file in zip(image_files, box_files):
                tasks.append(
                    executor.submit(self.blur_process, image_file, box_file)
                )

            # Display progress with tqdm
            for future in tqdm(as_completed(tasks), total=len(tasks), desc="Bluring Images"):
                # Wait for task completion
                future.result()