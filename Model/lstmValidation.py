import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import numpy as np
from joblib import load

# Load the dataset
validation_data = pd.read_csv('../ScaledVD.csv', parse_dates=['period'], index_col='period')
validation_data = validation_data.asfreq('T')  # Ensure it's at a minute frequency

# Initialize list to keep track of predictions
predictions = []
start_time = pd.Timestamp("1998-06-24 00:15:00")


model = load_model('optimisedLSTM.keras')
scaler = load('scaler.joblib')

look_back = 15
    
for index in range(len(validation_data) - look_back):
    # Prepare the sequence for prediction
    current_sequence = validation_data['count'].iloc[index:index+look_back].values.reshape(-1, 1)
    current_sequence_scaled = scaler.transform(current_sequence)
    # Make prediction
    input_sequence = current_sequence_scaled.reshape((1, look_back, 1))
    predicted_rpm_scaled = model.predict(input_sequence)
    predicted_rpm = scaler.inverse_transform(predicted_rpm_scaled)
    predictions.append(predicted_rpm[0][0])
    print(f"Predicted RPM at {validation_data.index[index + look_back]}: {predicted_rpm[0][0]}")

prediction_index = pd.date_range(start="1998-06-24 00:14:00", periods=len(predictions), freq='T')
predictions_df = pd.DataFrame(data=predictions, index=prediction_index, columns=["period"])
predictions_df.to_csv('lstm_predictions.csv')  
