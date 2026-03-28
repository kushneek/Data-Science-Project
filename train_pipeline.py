import sys
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.mlproject.components.data_transformation import DataTransformationConfig, DataTransformation
from src.mlproject.components.model_training import ModelTrainerConfig, ModelTrainer
 
 
if __name__ == "__main__":
    logging.info("The training pipeline has started !!!")
 
    try:
        # Step 1 - Data Ingestion
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
 
        # Step 2 - Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
 
        # Step 3 - Model Training
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr, test_arr))
 
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys)