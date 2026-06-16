# Student Maths Score Predictor

A Flask web app that predicts a student's Maths score using scores from Science, English, and History alongside basic student attributes. Trained on a labeled student dataset using multiple regression models with the best performer selected via MLflow experiment tracking.

**Live:** https://data-science-project-1-36kz.onrender.com/predictdata

![Prediction Form](Screenshot.png)

<img width="1536" height="726" alt="Screenshot" src="https://github.com/user-attachments/assets/ba91c6dd-d073-4a2e-8aed-09e5171b3a10" />

---

## Stack

- **Backend:** Python, Flask, Gunicorn
- **ML:** Scikit-learn, XGBoost, CatBoost
- **Tracking:** MLflow, DagsHub
- **Data versioning:** DVC
- **Deployment:** Render

---

## How It Works

User inputs Gender, Age, Section, Science score, English score, and History score through a web form. The request hits a prediction pipeline that loads a preprocessor and trained model from serialized `.pkl` files and returns the predicted Maths score.

Training and prediction pipelines are fully separated — only the prediction pipeline runs on the server.

---

## Local Setup

```bash
1) git clone https://github.com/kushneek/Data-Science-Project.git
cd Data-Science-Project

2) python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

3) pip install -r Requirements.txt
pip install -e .

4) python app.py
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

## Model Training

Multiple regression models evaluated — Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, XGBoost, CatBoost. Best model selected based on R² score and serialized to `artifacts/model.pkl`. All runs logged via MLflow on DagsHub.

---

## Author

Kushagra Neekhra — [GitHub](https://github.com/kushneek) 
Email - kush.neek18@gmail.com
