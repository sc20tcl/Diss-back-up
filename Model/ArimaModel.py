import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from itertools import product
from sklearn.metrics import mean_squared_error
from math import sqrt

data = pd.read_csv('../TestData.csv', parse_dates=True, index_col='period')
data = data.asfreq('T')  

train = data[data.index.dayofweek < 4] 
test = data[data.index.dayofweek == 4]  

p = d = q = range(0, 4)

pdq = [(3, 0, 2)]

best_rmse = float("inf")
best_order = None
best_model = None

for param in pdq:
    # try:
    temp_model = ARIMA(train['count'], order=param)
    results = temp_model.fit()
    predictions = results.predict(start=test.index[0], end=test.index[-1], dynamic=False)
    print(predictions)
    rmse = sqrt(mean_squared_error(test['count'], predictions))
    if rmse < best_rmse:
        best_rmse = rmse
        best_order = param
        best_model = results
    print('ARIMA%s RMSE=%.3f' % (param, rmse))
    # except Exception as e:
    #     print(f'Error with model {param}: {str(e)}')
    #     continue

print(f'Best ARIMA Model: {best_order} with RMSE: {best_rmse}')

best_model.save('arima_model.pkl')

# Plot the results
plt.figure(figsize=(10, 5))  # Set a figure size for better visibility
# train['count'].plot(label='Train', linewidth=1)
# test['count'].plot(label='Test', linewidth=1, alpha=0.7)
plt.plot(test.index, predictions, label='Predicted', linewidth=1, linestyle='--', color='green')
plt.plot(train.index, train['count'], label='Predicted', linewidth=1, linestyle='--', color='blue')
plt.plot(test.index, test['count'], label='Predicted', linewidth=1, linestyle='--', color='orange')

plt.legend(loc='best')
plt.title('RPM Forecasting')
plt.xlabel('Date')
plt.ylabel('RPM')
plt.tight_layout() 
plt.show()



# p = range(0, 3)  # example: 0, 1, 2
# d = range(0, 2)  # example: 0, 1
# q = range(0, 3)  # example: 0, 1, 2

# best_rmse = float("inf")
# best_cfg = None

# for i in p:
#     for j in d:
#         for k in q:
#             order = (i, j, k)
#             try:
#                 model = ARIMA(train['count'], order=order)
#                 model_fit = model.fit()
#                 predictions = model_fit.forecast(steps=len(test))
#                 rmse = np.sqrt(mean_squared_error(test['count'], predictions))
#                 if rmse < best_rmse:
#                     best_rmse = rmse
#                     best_cfg = order
#                 print('ARIMA%s RMSE=%.3f' % (order, rmse))
#             except Exception as e:
#                 print(f'Failed to fit ARIMA{order}: {str(e)}')

# if best_cfg is None:
#     print("Failed to find a suitable ARIMA model.")
# else:
#     print(f'Best ARIMA{best_cfg} RMSE={best_rmse:.3f}')
#     # Fit the model with the best configuration on the full dataset
#     model = ARIMA(data['count'], order=best_cfg)
#     model_fit = model.fit()
#     model.save('arima_model.pkl')