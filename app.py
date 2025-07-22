import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from backend import predict_live, classify_prediction_stage, load_all_logs, minutes_per_quarter

st.set_page_config(page_title="🏀 Basketball Predictor", layout="wide")
st.title("🏀 Basketball Live Prediction Dashboard")

menu = st.sidebar.selectbox("Select Action", ["Live Prediction", "View Trends", "History"])

if menu == "Live Prediction":
    st.header("📡 Make a Prediction")

    home_team = st.text_input("Home Team")
    away_team = st.text_input("Away Team")
    home_tag = st.text_input("Home Tag")
    away_tag = st.text_input("Away Tag")
    quarter = st.number_input("Quarter (1-4)", min_value=1, max_value=4, value=1, step=1)
    
    # 🔧 FIX: ensure all values are float type
    mins_remaining = st.number_input(
        f"Minutes Remaining (0-{minutes_per_quarter})",
        min_value=0.0,
        max_value=float(minutes_per_quarter),
        value=float(minutes_per_quarter),
        step=0.1
    )

    current_total = st.number_input("Current Total Score", min_value=0, value=0)
    line = st.number_input("Over/Under Line", min_value=0.0, value=100.0, step=0.1)

    if st.button("Predict"):
        elapsed = (quarter - 1) * minutes_per_quarter + (minutes_per_quarter - mins_remaining)
        elapsed = max(elapsed, 1)
        score_rate = current_total / elapsed

        avg_home_score = 100  # Placeholder, could pull from history later
        avg_away_score = 100  # Same here

        input_data = {
            'team_home': home_team,
            'team_away': away_team,
            'home_tag': home_tag,
            'away_tag': away_tag,
            'score_rate': score_rate,
            'line': line,
            'avg_home_score': avg_home_score,
            'avg_away_score': avg_away_score
        }

        pred_class, conf, pred_total = predict_live(input_data)
        prediction_stage = classify_prediction_stage(quarter, mins_remaining)

        st.success(f"Prediction Stage: {prediction_stage}")
        st.info(f"Over/Under Prediction: {'OVER' if pred_class else 'UNDER'} (confidence: {conf:.2f})")
        st.info(f"Predicted Final Total Score: {pred_total:.2f}")

elif menu == "View Trends":
    st.header("📈 Combined Trends")
    logs = load_all_logs()
    if logs.empty:
        st.warning("No predictions logged yet.")
    else:
        over_count = (logs['prediction'] == 'OVER').sum()
        under_count = (logs['prediction'] == 'UNDER').sum()
        st.metric("Total Predictions", len(logs))
        st.metric("OVER Predictions", over_count)
        st.metric("UNDER Predictions", under_count)

        st.subheader("OVER vs UNDER")
        fig, ax = plt.subplots()
        ax.bar(['OVER', 'UNDER'], [over_count, under_count])
        st.pyplot(fig)

elif menu == "History":
    st.header("🕘 Prediction History")
    logs = load_all_logs()
    if logs.empty:
        st.warning("No predictions logged yet.")
    else:
        st.dataframe(logs.tail(10))
        if {'predicted_total_score', 'actual_total_score'}.issubset(logs.columns):
            st.line_chart(logs[['predicted_total_score', 'actual_total_score']])
