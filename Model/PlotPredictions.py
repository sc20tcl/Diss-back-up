import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.dates as mdates

original_data = pd.read_csv('../ScaledVD.csv', parse_dates=True, index_col='period')
predicted_data = pd.read_csv('lstm_predictions.csv', parse_dates=True, index_col='period')

def create_graph(start_date, end_date, graph_name):
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date)  

    original_filtered = original_data[start_date:end_date]
    predicted_filtered = predicted_data[start_date:end_date]
    rmse = sqrt(mean_squared_error(original_filtered, predicted_filtered))/len(predicted_data)
    print("rsme: ", rmse)

    plt.figure(figsize=(10, 7))
    plt.plot(original_filtered, label='Original QPS', linestyle='-', color='blue')
    plt.plot(predicted_filtered, label='Predicted QPS', linestyle='--', color='orange')
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=5))  # Locator to show ticks every hour; adjust as necessary
    plt.xticks(rotation=45)

    time_format = '%H:%M'
    title = f"Comparison of Original and Predicted QPS between {start_datetime.strftime(time_format)} and {end_datetime.strftime(time_format)}"
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Queries Per Second (QPS)')

    plt.legend()

    plt.savefig(f'Model Graphs/{graph_name}.png')


highlight_ranges = [
    ('1998-06-24 16:02:00', '1998-06-24 17:02:00', 'Largest Decrease'), # highest decrease
    ('1998-06-24 13:49:00', '1998-06-24 14:49:00', 'Largest Increase'), # highest increase
    ('1998-06-24 10:24:00', '1998-06-24 11:24:00', "Lowest Variance"), # lowest variance
    ('1998-06-24 16:52:00', '1998-06-24 17:52:00', "Highest Variance"), # highest variance
    ('1998-06-24 00:14:00', '1998-06-24 23:58:00', 'Whole Day') 
]

for test in highlight_ranges:
    create_graph(test[0], test[1], test[2])
