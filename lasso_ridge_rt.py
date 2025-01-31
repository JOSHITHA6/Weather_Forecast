# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import requests  # For downloading the dataset programmatically
import io       # To handle in-memory data

# Download the dataset from GitHub
url = "https://raw.githubusercontent.com/JOSHITHA6/Weather_Forecast/main/Project1WeatherDataset.csv"
response = requests.get(url)
if response.status_code == 200:
    data = pd.read_csv(io.StringIO(response.text))
else:
    print("Error: Unable to fetch the dataset. Please check the URL.")
    exit()

# Inspecting the data
print(data.head())
print(data.info())

# Preprocessing: Selecting relevant features and handling missing values
data = data[['Temp_C', 'Rel Hum_%', 'Wind Speed_km/h', 'Press_kPa']]
data.dropna(inplace=True)

# Defining features and target
X = data.drop('Temp_C', axis=1)
y = data['Temp_C']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Helper function to evaluate model performance
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R-squared: {r2:.2f}\n")
    return y_pred

# 1. Lasso Regression
lasso = Lasso(alpha=0.1, random_state=42)
lasso.fit(X_train, y_train)
y_pred_lasso = evaluate_model(lasso, X_test, y_test, "Lasso Regression")

# 2. Decision Tree Regression
decision_tree = DecisionTreeRegressor(random_state=42)
decision_tree.fit(X_train, y_train)
y_pred_tree = evaluate_model(decision_tree, X_test, y_test, "Decision Tree Regression")

# 3. Random Forest Regression
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)
y_pred_forest = evaluate_model(random_forest, X_test, y_test, "Random Forest Regression")

# Plotting actual vs predicted for each model
plt.figure(figsize=(18, 5))

# Lasso Regression
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_lasso, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Lasso Regression: Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')

# Decision Tree Regression
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_tree, alpha=0.6, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Decision Tree: Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')

# Random Forest Regression
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_forest, alpha=0.6, color='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Random Forest: Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')

plt.tight_layout()
plt.show()
