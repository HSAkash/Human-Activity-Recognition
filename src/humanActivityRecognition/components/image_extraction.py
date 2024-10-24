import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path
from humanActivityRecognition import logger
from humanActivityRecognition.entity.config_entity import ImageExtractionConfig
from concurrent.futures import ThreadPoolExecutor, as_completed

class ImageExtraction:
    def __init__(self, config: ImageExtractionConfig):
        self.config = config


    def resize_with_padding(self, image):
        # Get the original dimensions
        original_height, original_width = image.shape[:2]

        # Calculate the scaling factor while maintaining aspect ratio
        scale = min(self.config.IMAGE_WIDTH / original_width, self.config.IMAGE_HEIGHT / original_height)
        
        # Compute the new size while maintaining the aspect ratio
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize the image
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Create a black canvas of the target size
        canvas = np.zeros((self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH, 3), dtype=np.uint8)

        # Calculate the top-left corner position to center the image on the canvas
        x_offset = (self.config.IMAGE_WIDTH - new_width) // 2
        y_offset = (self.config.IMAGE_HEIGHT - new_height) // 2

        # Place the resized image onto the black canvas
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

        return canvas

    def frames_extraction(self, video_path):
        '''
        This function will extract the required frames from a video after resizing and normalizing them.
        Args:
            video_path: The path of the video in the disk, whose frames are to be extracted.
        Returns:
            frames_list: A list containing the resized.
        '''
        # Declare a list to store video frames.
        frames_list = []

        # Read the Video File using the VideoCapture object.
        video_reader = cv2.VideoCapture(video_path)

        # Get the total number of frames in the video.
        video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate the interval after which frames will be added to the list.
        skip_frames_window = max(int(video_frames_count / self.config.SEQUENCE_LENGTH), 1)

        # Iterate through the Video Frames.
        for frame_counter in range(self.config.SEQUENCE_LENGTH):

            # Set the current frame position of the video.
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

            # Reading the frame from the video.
            success, frame = video_reader.read()

            # Check if Video frame is not successfully read then break the loop
            if not success:
                break

            # Resize the Frame to fixed height and width.
            resized_frame = self.resize_with_padding(frame)

            # # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
            # normalized_frame = resized_frame / 255.0

            # Append the normalized frame into the frames list
            frames_list.append(resized_frame)

        # Release the VideoCapture object.
        video_reader.release()

        # Return the frames list.
        return frames_list

    def process_video(self, video_path, class_name, video_index, imageDataset_dest_dir):
        '''
        This function processes a single video by extracting frames and saving them to the destination directory.
        '''
        dest_dir = os.path.join(imageDataset_dest_dir, class_name, f"{video_index:0>5}")
        os.makedirs(dest_dir, exist_ok=True)

        # Skip processing if frames already exist
        if os.path.exists(os.path.join(dest_dir, f"{self.config.SEQUENCE_LENGTH-1:0>3}.{self.config.image_format}")):
            return

        frames = self.frames_extraction(video_path)
        for i, frame in enumerate(frames):
            cv2.imwrite(os.path.join(dest_dir, f"{i:0>3}.{self.config.image_format}"), frame.astype('uint8'))

    def run(self):
        class_dirs = [d for d in os.listdir(self.config.source_dir) if os.path.isdir(os.path.join(self.config.source_dir, d))]
        os.makedirs(self.config.destination_dir, exist_ok=True)
        logger.info(f"""
total_class = {len(class_dirs)}
Class Name: {class_dirs}
""")

        tasks = []
        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            # Collect video processing tasks
            for class_dir in class_dirs:
                class_name = os.path.basename(class_dir)
                video_paths = glob(f"{os.path.join(self.config.source_dir,class_name)}/*")
                for video_index, video_path in enumerate(video_paths):
                    tasks.append(
                        executor.submit(self.process_video, video_path, class_name, video_index, self.config.destination_dir)
                    )

            # Display progress with tqdm
            for future in tqdm(as_completed(tasks), total=len(tasks), desc="Processing videos"):
                # Wait for task completion
                future.result()
    
