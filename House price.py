# House Price Prediction using Linear Regression

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# 2. Load Dataset
boston = load_boston()

# Convert dataset into DataFrame
df = pd.DataFrame(boston.data, columns=boston.feature_names)

# Add target variable (house price)
df["PRICE"] = boston.target


# 3. Display First 5 Rows
print("Dataset Preview:\n")
print(df.head())


# 4. Check Dataset Info
print("\nDataset Info:\n")
print(df.info())


# 5. Check Missing Values
print("\nMissing Values:\n")
print(df.isnull().sum())


# 6. Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()


# 7. Split Features and Target
X = df.drop("PRICE", axis=1)
y = df["PRICE"]


# 8. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 9. Create Linear Regression Model
model = LinearRegression()


# 10. Train Model
model.fit(X_train, y_train)


# 11. Make Predictions
predictions = model.predict(X_test)


# 12. Model Evaluation
r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print("\nModel Performance:")
print("R2 Score:", r2)
print("Mean Squared Error:", mse)


# 13. Plot Actual vs Predicted Prices
plt.scatter(y_test, predictions)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()
