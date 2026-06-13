# Hospital Readmission Predictor

An end-to-end AI system to predict 30-day hospital readmission 
risk with Explainable AI and clinical intervention recommendations.

## Problem Statement
Hospitals face 15-20% readmission rates within 30 days of discharge.
Each readmission costs ₹40,000–₹80,000. This system predicts which 
patients are high risk at discharge — so doctors can intervene before 
they leave.

## Features
- Predicts readmission risk as a 0–100% score
- SHAP explanations — overall feature importance
- LIME explanations — per patient reasoning
- What-If simulator — quantifies impact of interventions

## Tech Stack
Python, Scikit-learn, Gradient Boosting, SHAP, LIME, Streamlit, Pandas

## Dataset
UCI Diabetes 130-US Hospitals dataset — 100,000+ real patient records

## Project Structure
readmission_project/
├── data/                  # Dataset (not included)
├── backend/
│   ├── preprocess.py      # Data cleaning + feature engineering
│   ├── train.py           # Model training + GridSearchCV
│   ├── explain.py         # SHAP + LIME logic
│   └── predict.py         # Prediction + What-If logic
├── models/                # Saved model files
├── frontend/
│   └── app.py             # Streamlit web application
└── notebooks/
    └── exploration.ipynb  # EDA notebook

## How To Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Add dataset
Download from Kaggle and place in data/diabetic_data.csv

### 3. Train model
python backend/train.py

### 4. Run app
python -m streamlit run frontend/app.py

## Results
- Best Model: Gradient Boosting
- ROC-AUC Score: 0.67+
- Key risk factors: prior inpatient visits, medication change rate

## Author
YOUR NAME | B.Tech 2026 | YOUR COLLEGE NAME