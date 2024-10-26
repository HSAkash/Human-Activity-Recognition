from humanActivityRecognition.config.configuration import ConfigurationManager
from humanActivityRecognition.components.prepare_base_model import PrepareBaseModel
from humanActivityRecognition import logger


STAGE_NAME = "Create a new stage for final dataset creation"

class PrepareBaseModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.run()




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e