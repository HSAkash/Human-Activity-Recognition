from humanActivityRecognition.config.configuration import ConfigurationManager
from humanActivityRecognition.components.feature_extraction import FeatureExtraction
from humanActivityRecognition import logger


STAGE_NAME = "Feature Extraction"

class FeatureExtractionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        feature_extraction_config = config.get_feature_extraction_config()
        feature_extraction = FeatureExtraction(config=feature_extraction_config)
        feature_extraction.run()




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = FeatureExtractionPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e