from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.model_training import ModelTrainer

if __name__ == "__main__":
    # Step 1: Data Ingestion
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # Step 2: Data Transformation
    transformation = DataTransformation()
    train_array, test_array, _ = transformation.initiate_data_transformation(train_path, test_path)

    # Step 3: Model Training
    trainer = ModelTrainer()
    r2_score = trainer.initiate_model_trainer(train_array, test_array)
    print(f"Model R2 Score: {r2_score}")