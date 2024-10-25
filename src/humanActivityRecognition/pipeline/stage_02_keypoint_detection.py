from humanActivityRecognition.config.configuration import ConfigurationManager
from humanActivityRecognition.components.keypoint_detection import KeypointDetection
from humanActivityRecognition import logger


STAGE_NAME = "Keypoint Detection"

class KeypointDetectionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        keypoint_detection_config = config.get_keypoint_detection_config()
        keypoint_detection = KeypointDetection(config=keypoint_detection_config)
        keypoint_detection.run()




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = KeypointDetectionPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e