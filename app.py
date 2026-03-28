import sys
import os
import numpy as np
import pandas as pd

from flask import Flask, request, render_template
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.pipeline.prediction_pipeline import CustomData, PredictPipeline

# ─────────────────────────────────────────────
# Flask Prediction App ONLY
# Training pipeline removed - not needed on server
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
            Age=int(request.form.get('Age')),
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
    port = int(os.environ.get("PORT", 8080))  # ✅ reads Render's port
    app.run(host="0.0.0.0", port=port)