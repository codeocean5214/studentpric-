import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("random_forest_math_score_model.pkl")

# Streamlit App UI
st.set_page_config(page_title="Math Score Predictor", layout="centered")
st.title("📊 Math Score Predictor using Random Forest")

st.write("Enter the student's reading and writing scores to predict their math score.")

# Input sliders
reading = st.slider("📘 Reading Score", 0, 100, 50)
writing = st.slider("✍️ Writing Score", 0, 100, 50)

# Predict button
if st.button("🔮 Predict Math Score"):
    input_features = np.array([[reading, writing]])
    prediction = model.predict(input_features)[0]
    st.success(f"✅ Predicted Math Score: **{round(prediction, 2)}**")
