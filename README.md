
# Uber Fare Prediction

A Streamlit application that loads the Uber Fares dataset, performs data cleaning and feature engineering, trains regression models, and saves the best model as a pipeline for future predictions.

## Dataset

The dataset is available on Kaggle:  
[Uber Fares Dataset](https://www.kaggle.com/datasets/yasserh/uber-fares-dataset)

Columns:  
- key: unique identifier  
- fare_amount: target variable (fare in dollars)  
- pickup_datetime: date and time of pickup  
- pickup_longitude, pickup_latitude: pickup coordinates  
- dropoff_longitude, dropoff_latitude: dropoff coordinates  
- passenger_count: number of passengers

## Features

- Cleans data: removes duplicates, handles missing values, caps outliers  
- Extracts datetime features: hour, day of week, month, year  
- Calculates trip distance using the Haversine formula  
- Compares three regression models: Linear Regression, Random Forest, Gradient Boosting  
- Performs hyperparameter tuning with GridSearchCV  
- Saves the final model as a scikit-learn pipeline (scaler + model)

## Installation

1. Clone this repository or download the files.  
2. Install the required packages:

```bash
pip install streamlit pandas numpy scikit-learn joblib
