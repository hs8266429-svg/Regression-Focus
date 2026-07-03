# ==========================================
# California Housing Price Prediction
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# Load Dataset
# ==========================================

housing = fetch_california_housing()

# Create DataFrame
df = pd.DataFrame(housing.data, columns=housing.feature_names)

# Add Target Column
df["HouseValue"] = housing.target

# Display Dataset
print("First 5 Rows")
print(df.head())

# Dataset Information
print("\nDataset Information")
print(df.info())

# Statistical Summary
print("\nStatistical Summary")
print(df.describe())

# ==========================================
# Features and Target
# ==========================================

X = df.drop("HouseValue", axis=1)
y = df["HouseValue"]

# ==========================================
# Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# Feature Scaling
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# Train Linear Regression Model
# ==========================================

model = LinearRegression()

model.fit(X_train_scaled, y_train)

# ==========================================
# Prediction
# ==========================================

y_pred = model.predict(X_test_scaled)

# ==========================================
# Model Evaluation
# ==========================================

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("----------------------")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# ==========================================
# Model Coefficients
# ==========================================

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nFeature Coefficients")
print(coefficients)

# ==========================================
# Actual vs Predicted Plot
# ==========================================

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")
plt.show()

# ==========================================
# Residual Plot
# ==========================================

residuals = y_test - y_pred

plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# ==========================================
# Predict New House Price
# ==========================================

sample = [[8.3, 40, 6.5, 1.1, 900, 2.8, 37.8, -122.4]]

sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)

print("\nPredicted House Value:")
print(prediction[0], "(in $100,000 units)")