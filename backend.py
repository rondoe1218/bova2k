import joblib
import pandas as pd
import os
import datetime

minutes_per_quarter = 5

MODEL_PATH_CLASS = 'models/classifier_model.pkl'
MODEL_PATH_REG = 'models/regressor_model.pkl'
LOGS_DIR = 'logs'

os.makedirs(LOGS_DIR, exist_ok=True)

clf_model = joblib.load(MODEL_PATH_CLASS)
reg_model = joblib.load(MODEL_PATH_REG)

def predict_live(input_data):
    df_live = pd.DataFrame([input_data])
    pred_class = clf_model.predict(df_live)[0]
    pred_confidence = clf_model.predict_proba(df_live).max()
    pred_total_score = reg_model.predict(df_live)[0]
    return pred_class, pred_confidence, pred_total_score

def classify_prediction_stage(quarter, minutes_remaining):
    if minutes_remaining == minutes_per_quarter:
        return f"START_OF_Q{quarter}"
    elif minutes_remaining == 0:
        return f"END_OF_Q{quarter}"
    else:
        return "IN_GAME"

def load_all_logs():
    logs = []
    for f in os.listdir(LOGS_DIR):
        if f.startswith('prediction_history_') and f.endswith('.csv'):
            logs.append(pd.read_csv(f'{LOGS_DIR}/{f}'))
    return pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
