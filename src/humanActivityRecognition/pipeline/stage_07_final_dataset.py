from humanActivityRecognition.config.configuration import ConfigurationManager
from humanActivityRecognition.components.final_dataset import FinalDataset
from humanActivityRecognition import logger


STAGE_NAME = "Create a new stage for final dataset creation"

class FinalDatasetPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        final_dataset_config = config.get_final_dataset_config()
        final_dataset = FinalDataset(config=final_dataset_config)
        final_dataset.run()




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = FinalDatasetPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e