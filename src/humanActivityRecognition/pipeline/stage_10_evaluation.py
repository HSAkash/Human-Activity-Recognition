from humanActivityRecognition.config.configuration import ConfigurationManager
from humanActivityRecognition.components.prepare_dataset import PrepareDataset
from humanActivityRecognition.components.evaluation import Evaluation
from humanActivityRecognition import logger


STAGE_NAME = "Model evaluation Pipeline"

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()

        logger.info("Preparing dataset")
        prepare_dataset_config = config.get_prepare_dataset_config()
        prepare_dataset = PrepareDataset(config=prepare_dataset_config)
        train_ds, test_ds = prepare_dataset.get_dataset()
        logger.info("Training dataset prepared")

        logger.info("Training model")
        training_config = config.get_evaluation_config()
        training = Evaluation(config=training_config, test_ds=test_ds)
        training.run()




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e