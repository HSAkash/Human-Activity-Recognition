from humanActivityRecognition.config.configuration import ConfigurationManager
from humanActivityRecognition.components.image_augmentation import ImageAugmentation
from humanActivityRecognition.components.keypoint_augmentation import KeypointAugmentation
from humanActivityRecognition import logger


STAGE_NAME = "Data Augmentation"

class DataAugmentationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_augmentation_config = config.get_data_augmentation_config()

        logger.info("Image Augmentation Started")
        image_augmentation = ImageAugmentation(config=data_augmentation_config)
        image_augmentation.run()

        logger.info("Keypoint Augmentation Started")
        keypoint_augmentation = KeypointAugmentation(config=data_augmentation_config)
        keypoint_augmentation.run()




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataAugmentationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e