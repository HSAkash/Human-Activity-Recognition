from humanActivityRecognition.config.configuration import ConfigurationManager
from humanActivityRecognition.components.spliting_dataset import SplitingDataset
from humanActivityRecognition import logger


STAGE_NAME = "Spliting Dataset"

class SplitingDatasetPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        spliting_dataset_config = config.get_spliting_dataset_config()
        spliting_dataset = SplitingDataset(config=spliting_dataset_config)
        spliting_dataset.run()




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = SplitingDatasetPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e