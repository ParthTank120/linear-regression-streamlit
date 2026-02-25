import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")

st.title("🩺 Diabetes Progression Prediction")
st.write("This app uses Linear Regression to predict diabetes progression.")

# -----------------------------
# Load Dataset
# -----------------------------
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

st.subheader("📊 Dataset Information")
st.write(f"Number of samples: {X.shape[0]}")
st.write(f"Number of features: {X.shape[1]}")

# -----------------------------
# Train/Test Split
# -----------------------------
test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20)
test_size = test_size / 100

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# -----------------------------
# Metrics
# -----------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📈 Model Performance")
st.success(f"Mean Squared Error: {mse:.2f}")
st.info(f"R-squared (R² Score): {r2:.2f}")

# -----------------------------
# Visualization
# -----------------------------
st.subheader("📊 Visualizations")

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# True vs Predicted
axs[0].scatter(y_test, y_pred, color="blue", alpha=0.5)
axs[0].plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "k--",
    lw=2
)
axs[0].set_title("True vs Predicted Values")
axs[0].set_xlabel("True Values")
axs[0].set_ylabel("Predicted Values")
axs[0].grid(True)

# BMI (Feature 2) vs Predicted
axs[1].scatter(X_test[:, 2], y_pred, color="green", alpha=0.7)
axs[1].set_title("BMI (Feature 2) vs Predicted Values")
axs[1].set_xlabel("BMI (Feature 2)")
axs[1].set_ylabel("Predicted Diabetes Progression")
axs[1].grid(True)

plt.tight_layout()

st.pyplot(fig)
