from humanActivityRecognition.config.configuration import ConfigurationManager
from humanActivityRecognition.components.keypoint_detection import KeypointDetection
from humanActivityRecognition import logger


STAGE_NAME = "Keypoint Detection"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        image_extraction_config = config.get_keypoint_detection_config()
        image_extraction = KeypointDetection(config=image_extraction_config)
        image_extraction.run()




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e