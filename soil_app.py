import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load model
model = pickle.load(open("soil_model.pkl", "rb"))

st.title("Soil Classification Prediction (IS System)")
st.write("Machine Learning Based Soil Classification Model")

# -----------------------------
# User Inputs
# -----------------------------
LL = st.number_input("Liquid Limit (LL)", min_value=0.0)
PL = st.number_input("Plastic Limit (PL)", min_value=0.0)
Finess = st.number_input("Fines (%)", min_value=0.0)
Sand = st.number_input("Sand (%)", min_value=0.0)
Gravel = st.number_input("Gravel (%)", min_value=0.0)

if st.button("Predict Soil Type"):

    PI = LL - PL

    input_data = pd.DataFrame([[LL, PL, PI, Fines, Sand, Gravel]],
        columns=["LL", "PL", "PI", "Fines", "Sand", "Gravel"])

    prediction = model.predict(input_data)

    st.success(f"Plasticity Index (PI): {PI}")
    st.success(f"Predicted Soil Classification: {prediction[0]}")

    # -----------------------------
    # Plasticity Chart
    # -----------------------------
    st.subheader("Plasticity Chart (LL vs PI)")

    fig1, ax1 = plt.subplots()
    ax1.scatter(LL, PI)
    ax1.set_xlabel("Liquid Limit (LL)")
    ax1.set_ylabel("Plasticity Index (PI)")
    ax1.set_title("Plasticity Chart")

    st.pyplot(fig1)

    # -----------------------------
    # Feature Importance Chart
    # -----------------------------
    st.subheader("Feature Importance")

    importance = model.feature_importances_
    features = ["LL", "PL", "PI", "Fines", "Sand", "Gravel"]

    fig2, ax2 = plt.subplots()
    ax2.bar(features, importance)
    ax2.set_title("Feature Importance")
    ax2.set_xticklabels(features, rotation=45)

    st.pyplot(fig2)

