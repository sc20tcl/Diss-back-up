import json
import numpy as np
import os
from tensorflow.keras.models import load_model
from joblib import load
from azureml.core.model import Model

def init():
    global model
    global scaler
    model_path = Model.get_model_path('lstm_model')
    scaler_path = Model.get_model_path('scaler')
    model = load_model(model_path)
    scaler = load(scaler_path)

def run(raw_data):
    try:
        data = np.array(json.loads(raw_data)['data'])
        predicted_rpm_scaled = model.predict(data)
        predicted_rpm = scaler.inverse_transform(predicted_rpm_scaled)
        return {"result": predicted_rpm}
    
    except Exception as e:
        error = str(e)
        return {"error": error}