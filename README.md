# AI-Powered Energy Usage Forecasting

Welcome to the **electricity-forecasting** project!  
This repository provides tools and models to predict future electricity consumption using machine learning and artificial intelligence. Our goal is to enable better energy management for utilities, businesses, and individuals by providing accurate usage forecasts.

## Project Description

This project analyzes historical electricity usage data and applies AI and machine learning models to forecast future consumption. The result is a system that identifies patterns in energy usage and helps optimize costs, reduce waste, and support sustainability efforts. The code includes data preprocessing, model training, evaluation, and visualization, all accessible via Jupyter notebooks.

## Features

- Data cleaning and preprocessing
- Multiple machine learning models (e.g., Linear Regression, Decision Trees, etc.)
- Model evaluation and accuracy reporting
- Interactive visualization of forecasts
- Jupyter notebook for step-by-step demonstration

## Quick Start Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('data/electricity_usage.csv')

# Prepare features and target
X = data[['temperature', 'hour', 'day_of_week']]
y = data['usage']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Visualize
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()
