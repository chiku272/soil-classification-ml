import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("soil_model.pkl", "rb"))

st.title("Soil Classification System ")
st.write("Sieve Analysis  ")

# =====================================================
# SIEVE ANALYSIS INPUT
# =====================================================
st.header("Sieve Analysis Input")

total_weight = st.number_input("Total Dry Weight (g)", min_value=0.0)

ret_475 = st.number_input("Retained on 4.75 mm (g)", min_value=0.0)
ret_236 = st.number_input("Retained on 2.36 mm (g)", min_value=0.0)
ret_600 = st.number_input("Retained on 600 micron (g)", min_value=0.0)
ret_425 = st.number_input("Retained on 425 micron (g)", min_value=0.0)
ret_75 = st.number_input("Retained on 75 micron (g)", min_value=0.0)
pan = st.number_input("Pan (<75 micron) (g)", min_value=0.0)

# =====================================================
# ATTERBERG LIMITS
# =====================================================
st.header("Atterberg Limits")

LL = st.number_input("Liquid Limit (LL)", min_value=0.0)
PL = st.number_input("Plastic Limit (PL)", min_value=0.0)

# =====================================================
# WATER CONTENT
# =====================================================
st.header("Water Content")

wet_weight = st.number_input("Wet Weight (g)", min_value=0.0)
dry_weight = st.number_input("Dry Weight (g)", min_value=0.0)

# =====================================================
# MAIN BUTTON
# =====================================================
if st.button("Analyze Soil"):

    # -----------------------------
    # VALIDATIONS
    # -----------------------------
    if total_weight <= 0:
        st.error("Total weight must be greater than 0.")
        st.stop()

    total_retained = ret_475 + ret_236 + ret_600 + ret_425 + ret_75 + pan

    if abs(total_retained - total_weight) > 1:
        st.warning("Sum of retained weights does not match total weight.")

    if PL > LL:
        st.error("Plastic Limit cannot be greater than Liquid Limit.")
        st.stop()

    if dry_weight <= 0:
        st.error("Dry weight must be greater than 0.")
        st.stop()

    if wet_weight < dry_weight:
        st.error("Wet weight cannot be less than dry weight.")
        st.stop()

    # -----------------------------
    # CALCULATIONS
    # -----------------------------
    pr_475 = (ret_475 / total_weight) * 100
    pr_pan = (pan / total_weight) * 100

    gravel = pr_475
    fines = pr_pan
    sand = 100 - (gravel + fines)

    if sand < 0:
        st.error("Invalid sieve data: Sand cannot be negative.")
        st.stop()

    st.subheader("Soil Composition")
    st.success(f"Gravel: {gravel:.2f}%")
    st.success(f"Sand: {sand:.2f}%")
    st.success(f"Fines: {fines:.2f}%")

    # -----------------------------
    # WATER CONTENT
    # -----------------------------
    water_content = ((wet_weight - dry_weight) / dry_weight) * 100
    st.success(f"Water Content: {water_content:.2f}%")

    # -----------------------------
    # PI
    # -----------------------------
    PI = LL - PL

    if PI < 0:
        st.error("Plasticity Index cannot be negative.")
        st.stop()

    st.success(f"Plasticity Index (PI): {PI:.2f}")

    # -----------------------------
    # CONSISTENCY STATE
    # -----------------------------
    st.subheader("Consistency State")

    if water_content > LL:
        state = "Liquid State"
    elif water_content > PL:
        state = "Plastic State"
    elif water_content > (0.5 * PL):
        state = "Semi-Solid State"
    else:
        state = "Solid State"

    st.info(f"Soil is in: {state}")

    # -----------------------------
    # ML MODEL
    # -----------------------------
    input_data = pd.DataFrame([[LL, PL, PI, fines, sand, gravel]],
        columns=["LL", "PL", "PI", "Fines", "Sand", "Gravel"])

    prediction = model.predict(input_data)

    st.subheader("Soil Classification")
    st.success(prediction[0])

    # =====================================================
    # PLASTICITY CHART
    # =====================================================
    st.subheader("Plasticity Chart")

    LL_range = np.linspace(0, 100, 100)
    A_line = 0.73 * (LL_range - 20)

    fig1, ax1 = plt.subplots()
    ax1.plot(LL_range, A_line)
    ax1.scatter(LL, PI)

    ax1.set_xlabel("Liquid Limit (LL)")
    ax1.set_ylabel("Plasticity Index (PI)")

    st.pyplot(fig1)

    # =====================================================
    # PIE CHART
    # =====================================================
    st.subheader("Soil Composition (%)")

    fig2, ax2 = plt.subplots()
    ax2.pie([fines, sand, gravel],
            labels=["Fines", "Sand", "Gravel"],
            autopct='%1.1f%%')

    st.pyplot(fig2)

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================
    st.subheader("Feature Importance")

    importance = model.feature_importances_
    features = ["LL", "PL", "PI", "Fines", "Sand", "Gravel"]

    fig3, ax3 = plt.subplots()
    ax3.bar(features, importance)

    st.pyplot(fig3)
