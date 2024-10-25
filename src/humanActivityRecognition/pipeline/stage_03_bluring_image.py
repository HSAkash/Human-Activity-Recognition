from humanActivityRecognition.config.configuration import ConfigurationManager
from humanActivityRecognition.components.bluring_image import BluringImage
from humanActivityRecognition import logger


STAGE_NAME = "Bluring The Images"

class BluringImagePipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        bluring_image_config = config.get_bluring_image_config()
        bluring_image = BluringImage(config=bluring_image_config)
        bluring_image.run()




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = BluringImagePipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e