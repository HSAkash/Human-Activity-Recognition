from humanActivityRecognition import logger
from humanActivityRecognition.pipeline.stage_01_image_extraction import DataIngestionTrainingPipeline
from humanActivityRecognition.pipeline.stage_02_keypoint_detection import KeypointDetectionPipeline
from humanActivityRecognition.pipeline.stage_03_bluring_image import BluringImagePipeline
from humanActivityRecognition.pipeline.stage_04_spliting_dataset import SplitingDatasetPipeline
from humanActivityRecognition.pipeline.stage_05_data_augmentation import DataAugmentationPipeline
from humanActivityRecognition.pipeline.stage_06_feature_extraction import FeatureExtractionPipeline
from humanActivityRecognition.pipeline.stage_07_final_dataset import FinalDatasetPipeline


if __name__ == '__main__':

    STAGE_NAME = "Image Extraction from videos"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


    STAGE_NAME = "Keypoint Detection"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = KeypointDetectionPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


    STAGE_NAME = "Bluring Image"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = BluringImagePipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


    STAGE_NAME = "Splitting Dataset"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = SplitingDatasetPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


    STAGE_NAME = "Data Augmentation"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataAugmentationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


    STAGE_NAME = "Feature Extraction"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = FeatureExtractionPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


    STAGE_NAME = "Create a new stage for final dataset creation"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = FinalDatasetPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
