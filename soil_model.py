# ---------------------------------------------
# Soil Classification Model (IS System)
# Machine Learning - Decision Tree Classifier
# ---------------------------------------------

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# For reproducibility
np.random.seed(42)

# ---------------------------------------------
# Function to Generate Synthetic Soil Data
# ---------------------------------------------

def generate_soil_data(label, n, LL_range, PL_range, fines_range, sand_range, gravel_range):
    LL = np.random.uniform(*LL_range, n)
    PL = np.random.uniform(*PL_range, n)
    fines = np.random.uniform(*fines_range, n)
    sand = np.random.uniform(*sand_range, n)
    gravel = np.random.uniform(*gravel_range, n)

    return pd.DataFrame({
        "LL": LL,
        "PL": PL,
        "Fines": fines,
        "Sand": sand,
        "Gravel": gravel,
        "Soil_Class": [label]*n
    })

# ---------------------------------------------
# Create Dataset (70 Samples, 10 Soil Types)
# ---------------------------------------------

data = pd.concat([

    generate_soil_data("CL (Clay of Low Plasticity)", 7, (35,50), (18,30), (50,70), (20,40), (5,15)),
    generate_soil_data("CH (Clay of High Plasticity)", 7, (60,90), (25,45), (65,90), (5,25), (0,10)),
    generate_soil_data("ML (Silt of Low Plasticity)", 7, (30,45), (15,25), (40,60), (30,50), (5,15)),
    generate_soil_data("MH (Silt of High Plasticity)", 7, (55,75), (25,40), (60,85), (10,30), (0,10)),
    generate_soil_data("SC (Clayey Sand)", 7, (20,35), (10,20), (20,40), (50,70), (5,20)),
    generate_soil_data("SM (Silty Sand)", 7, (25,40), (12,22), (30,50), (40,65), (5,15)),
    generate_soil_data("SW (Well Graded Sand)", 7, (15,30), (5,15), (5,15), (70,85), (5,15)),
    generate_soil_data("SP (Poorly Graded Sand)", 7, (15,30), (5,12), (5,20), (75,90), (5,10)),
    generate_soil_data("GW (Well Graded Gravel)", 7, (10,25), (3,10), (0,10), (20,40), (55,75)),
    generate_soil_data("GP (Poorly Graded Gravel)", 7, (10,25), (3,8), (0,10), (15,35), (60,85))

])

# ---------------------------------------------
# Calculate Plasticity Index (PI)
# PI = LL - PL
# ---------------------------------------------

data["PI"] = data["LL"] - data["PL"]

# Features and Target
X = data[["LL", "PL", "PI", "Fines", "Sand", "Gravel"]]
y = data["Soil_Class"]

# ---------------------------------------------
# Train-Test Split
# ---------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------
# Train Model
# ---------------------------------------------

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------------
# Evaluate Model
# ---------------------------------------------

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

# ---------------------------------------------
# Save Model
# ---------------------------------------------

pickle.dump(model, open("soil_model.pkl", "wb"))

print("Model saved successfully as soil_model.pkl")
