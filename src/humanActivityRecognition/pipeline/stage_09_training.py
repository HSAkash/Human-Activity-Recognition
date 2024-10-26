from humanActivityRecognition.config.configuration import ConfigurationManager
from humanActivityRecognition.components.prepare_callbacks import PrepareCallbacks
from humanActivityRecognition.components.prepare_dataset import PrepareDataset
from humanActivityRecognition.components.training import Training
from humanActivityRecognition import logger


STAGE_NAME = "Training Pipeline"

class TrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()

        logger.info("Preparing callbacks")
        prepare_callbacks_config = config.get_prepare_callbacks_config()
        prepare_callbacks = PrepareCallbacks(config=prepare_callbacks_config)
        callbacks = prepare_callbacks.get_callbacks()

        logger.info("Preparing dataset")
        prepare_dataset_config = config.get_prepare_dataset_config()
        prepare_dataset = PrepareDataset(config=prepare_dataset_config)
        train_ds, test_ds = prepare_dataset.get_dataset()
        logger.info("Training dataset prepared")

        logger.info("Training model")
        training_config = config.get_training_config()
        training = Training(config=training_config, callbacks=callbacks, train_ds=train_ds, test_ds=test_ds)
        training.run()




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = TrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e