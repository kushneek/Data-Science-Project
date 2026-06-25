# Student Performance Predictor

A Flask web app that predicts a student's Math score based on gender, 
race/ethnicity, parental level of education, lunch type, test preparation 
course, reading score and writing score. Trained on a labeled student 
dataset using 9 regression models with the best performer selected 
automatically via GridSearchCV and tracked using MLflow on DagsHub.

**Live:** https://data-science-project-1-36kz.onrender.com/predictdata

<img width="1536" height="726" alt="Screenshot" src="https://github.com/user-attachments/assets/ba91c6dd-d073-4a2e-8aed-09e5171b3a10" />

---

## Stack

- **Backend:** Python, Flask
- **ML:** Scikit-learn, XGBoost, CatBoost
- **Tracking:** MLflow, DagsHub
- **Deployment:** Render with GitHub CI/CD

---

## How It Works

User inputs gender, race/ethnicity, parental level of education, lunch 
type, test preparation course, reading score and writing score through 
a web form. The request hits a Flask REST API prediction endpoint that 
loads a preprocessor and trained model from serialized `.pkl` files 
and returns the predicted Math score in real time.

Training and prediction pipelines are fully separated — only the 
prediction pipeline runs on the server.

---

## Model Training

9 regression algorithms trained simultaneously:
- Linear Regression, Ridge, Lasso
- Decision Tree, Random Forest
- Gradient Boosting, XGBoost, CatBoost, AdaBoost

Automated hyperparameter tuning using GridSearchCV. Best model 
selected based on R² score and saved to `artifacts/model.pkl`.

**Best Model: Lasso Regression**
**R² Score: 88.06%**

All runs logged via MLflow on DagsHub.

Experiments: https://dagshub.com/kushneek/Data-Science-Project/experiments

---

## Data Pipeline

1. **Data Ingestion** — Reads raw CSV, splits into 80/20 train/test
2. **Data Transformation** — StandardScaler for numerical features, 
   OneHotEncoding for categorical features, saved as `preprocessor.pkl`
3. **Model Training** — GridSearchCV across 9 algorithms, best model saved
4. **Prediction** — Loads saved artifacts, transforms input, returns prediction

---

## Local Setup

```bash
1) git clone https://github.com/kushneek/Data-Science-Project.git
cd Data-Science-Project

2) python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

3) pip install -r Requirements.txt
pip install -e .

4) python -m src.mlproject.components.data_ingestion
   python -m src.mlproject.components.data_transformation
   python -m src.mlproject.components.model_training

5) python app.py
```

App runs at `http://localhost:8080/predictdata`

---

## Project Structure

```
├── app.py                        # Flask entry point
├── Requirements.txt
├── setup.py
├── src/mlproject/
│   ├── pipeline/
│   │   ├── prediction_pipeline.py
│   │   └── training_pipeline.py
│   ├── components/               # Ingestion, transformation, training
│   ├── logger.py
│   ├── exception.py
│   └── utils.py
└── artifacts/
    ├── model.pkl
    ├── preprocessor.pkl
    └── data/
```

---

## Author

Kushagra Neekhra — [GitHub](https://github.com/kushneek) | 
[LinkedIn](https://www.linkedin.com/in/kush-neek)