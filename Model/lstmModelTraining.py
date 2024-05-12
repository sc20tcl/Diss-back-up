import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from joblib import dump

def load_and_preprocess_data(filepath, train_start_date, train_end_date, test_start_date, test_end_date):
    data = pd.read_csv(filepath, parse_dates=['period'], index_col='period')
    train_data = data[train_start_date:train_end_date]
    test_data = data[test_start_date:test_end_date]
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data['count'].values.reshape(-1, 1))
    test_scaled = scaler.transform(test_data['count'].values.reshape(-1, 1))
    dump(scaler, 'scaler.joblib')
    return train_scaled, test_scaled, scaler

def create_dataset(data, look_back, steps_ahead=1):
    X, Y = [], []
    for i in range(len(data)-look_back-steps_ahead):
        a = data[i:(i+look_back), 0]
        X.append(a)
        Y.append(data[i + look_back + steps_ahead - 1, 0])
    return np.array(X), np.array(Y)

def build_model(look_back, lstm_units=50, learning_rate=0.001):
    model = Sequential()
    model.add(LSTM(lstm_units, input_shape=(look_back, 1)))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def train_and_test_model(model, X_train, y_train, X_test, y_test, batch_size, epochs):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    y_pred = model.predict(X_test).flatten()  
    test_rmse = np.sqrt(np.mean(np.square(y_test - y_pred))) 
    return test_rmse

def grid_search(filepath, train_start_date, train_end_date, test_start_date, test_end_date):
    # Define grid search parameters
    lstm_units_options = [10, 20, 30]
    batch_size_options = [8, 16, 24]
    epochs_options = [17, 20, 25]
    learning_rate_options = [0.007, 0.01, 0.05]
    look_back_options = [15]  # Different look-back windows to test

    best_loss = float('inf')
    best_config = {}

    for look_back in look_back_options:
        train_scaled, test_scaled, _ = load_and_preprocess_data(filepath, train_start_date, train_end_date, test_start_date, test_end_date)
        X_train, y_train = create_dataset(train_scaled, look_back)
        X_test, y_test = create_dataset(test_scaled, look_back)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        for units in lstm_units_options:
            for batch_size in batch_size_options:
                for epochs in epochs_options:
                    for lr in learning_rate_options:
                        print(f"test config look_back: {look_back}, lstm_unit: {units}, learning_rate: {lr}, batch_size: {batch_size}, epochs: {epochs}")
                        model = build_model(look_back, lstm_units=units, learning_rate=lr)
                        test_loss = train_and_test_model(model, X_train, y_train, X_test, y_test, batch_size, epochs)
                        if test_loss < best_loss:
                            best_loss = test_loss
                            best_config = {'look_back': look_back, 'units': units, 'batch_size': batch_size, 'epochs': epochs, 'learning_rate': lr}
                            print(f'New best model with loss {best_loss}: {best_config}')
                            model.save('optimisedLSTM.keras')

    return best_config, best_loss


# Set your parameters
train_start_date = '1998-06-15 00:00:00'
train_end_date = '1998-06-19 23:59:59'  
test_start_date = '1998-06-24 00:00:00' 
test_end_date = '1998-06-24 23:59:59'    
filepath = '../ScaledFRPM.csv' 

# best_config, best_loss = grid_search(filepath, train_start_date, train_end_date, test_start_date, test_end_date)
# print(f'Best configuration: {best_config}, Best loss: {best_loss}')

look_back = 15
units = 30
batch_size = 16
epochs = 25
lr = 0.007

train_scaled, test_scaled, _ = load_and_preprocess_data(filepath, train_start_date, train_end_date, test_start_date, test_end_date)
X_train, y_train = create_dataset(train_scaled, look_back)
X_test, y_test = create_dataset(test_scaled, look_back)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

config = {'look_back': look_back, 'units': units, 'batch_size': batch_size, 'epochs': epochs, 'learning_rate': lr}
model = build_model(look_back, lstm_units=units, learning_rate=lr)
test_loss = train_and_test_model(model, X_train, y_train, X_test, y_test, batch_size, epochs)
model.save('optimisedLSTM.keras')
print(f'Best configuration: {config}, Best loss: {test_loss}')