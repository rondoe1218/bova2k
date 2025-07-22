import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Paths — no "models/" folder prefix now:
MODEL_PATH_CLASS = "classifier_model.pkl"
MODEL_PATH_REG = "regressor_model.pkl"

# Constants
minutes_per_quarter = 5

# Load models at startup
clf_model = joblib.load(MODEL_PATH_CLASS)
reg_model = joblib.load(MODEL_PATH_REG)

def predict_live(input_data):
    df = pd.DataFrame([input_data])

    pred_class = clf_model.predict(df)[0]
    conf = clf_model.predict_proba(df).max()

    pred_total = reg_model.predict(df)[0]

    return pred_class, conf, pred_total

def classify_prediction_stage(q, mins):
    if q == 1 and mins == 0:
        return "END_OF_Q1"
    elif q == 2 and mins == 0:
        return "END_OF_Q2"
    elif q == 3 and mins == 0:
        return "END_OF_Q3"
    elif q == 4 and mins == 0:
        return "END_OF_Q4"
    elif q == 1 and mins == minutes_per_quarter:
        return "START_OF_Q1"
    elif q == 2 and mins == minutes_per_quarter:
        return "START_OF_Q2"
    elif q == 3 and mins == minutes_per_quarter:
        return "START_OF_Q3"
    elif q == 4 and mins == minutes_per_quarter:
        return "START_OF_Q4"
    else:
        return "IN_GAME"

def load_all_logs():
    try:
        df = pd.read_csv("prediction_history.csv")
    except FileNotFoundError:
        df = pd.DataFrame()
    return df
