import sys
import numpy as np
import pandas as pd

from flask import Flask, request, render_template
from src.mlproject.logger import logging 
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_ingestion import DataIngestionConfig
from src.mlproject.components.data_transformation import DataTransformationConfig, DataTransformation
from src.mlproject.components.model_training import ModelTrainerConfig, ModelTrainer
from src.mlproject.pipeline.prediction_pipeline import CustomData, PredictPipeline

if __name__=="__main__":
    logging.info("The execution has started !!!")

    try:
        #data_ingestion_config=DataIngestionConfig()
        data_ingestion=DataIngestion()
        train_data_path, test_data_path= data_ingestion.initiate_data_ingestion()

        # data_transformation_config=DataTransformationConfig()
        data_transformation=DataTransformation()
        train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)

        ## Model Training
        model_trainer=ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))
        
    except Exception as e:
        logging.info("Custom Exception ")
        raise CustomException(e, sys)
    

# Flask Prediction App
# ─────────────────────────────────────────────
application = Flask(__name__)
app = application
 
## Route for home page
@app.route('/')
def index():
    return render_template('index.html')
 
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
                Gender=request.form.get('Gender'),
                Section=request.form.get('Section'),
                Science=int(request.form.get('Science')),
                English=int(request.form.get('English')),
                History=int(request.form.get('History'))
        )
 
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")
 
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
 
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")
 
        return render_template('home.html', results=results[0])
 
 
if __name__ == "__main__":       
    app.run(host="0.0.0.0")      