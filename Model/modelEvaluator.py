import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib 

def load_models(model_directory):
    models = {}
    for filename in os.listdir(model_directory):
        if filename.endswith('.pkl'):
            model_path = os.path.join(model_directory, filename)
            model_name = filename[:-4]
            models[model_name] = joblib.load(model_path)
    return models

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

def main():
    data_path = input("Enter the path to your dataset: ")
    df = pd.read_csv(data_path)
    
    # Assuming the last column is the target variable
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load models
    model_directory = input("Enter the path to your models directory: ")
    models = load_models(model_directory)
    
    # Evaluate and print results for each model
    for model_name, model in models.items():
        print(f"\\nEvaluating model: {model_name}")
        results = evaluate_model(model, X_test, y_test)
        for metric, value in results.items():
            print(f"{metric}: {value}")