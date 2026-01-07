import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Bank Note Authentication - Random Forest")

st.title("💵 Bank Note Authentication")
st.write("Random Forest Classifier")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("BankNote_Authentication.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Features & Target
# -----------------------------
X = df.drop("class", axis=1)
y = df["class"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -----------------------------
# Sidebar Parameters
# -----------------------------
st.sidebar.header("Model Parameters")

n_estimators = st.sidebar.slider(
    "Number of Trees", min_value=10, max_value=300, value=100
)

max_depth = st.sidebar.slider(
    "Max Depth", min_value=2, max_value=20, value=5
)

# -----------------------------
# Train Random Forest
# -----------------------------
model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Model Evaluation
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.sidebar.write(f"### Accuracy: **{accuracy:.2f}**")

# -----------------------------
# User Input
# -----------------------------
st.subheader("Enter Bank Note Details")

variance = st.number_input("Variance", -10.0, 10.0, 0.0)
skewness = st.number_input("Skewness", -10.0, 10.0, 0.0)
curtosis = st.number_input("Curtosis", -10.0, 10.0, 0.0)
entropy = st.number_input("Entropy", -10.0, 10.0, 0.0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    input_data = np.array([[variance, skewness, curtosis, entropy]])
    prediction = model.predict(input_data)

    if prediction[0] == 0:
        st.success("✅ Bank Note is **AUTHENTIC**")
    else:
        st.error("❌ Bank Note is **FAKE**")

# -----------------------------
# Confusion Matrix
# -----------------------------
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
st.write(cm)
