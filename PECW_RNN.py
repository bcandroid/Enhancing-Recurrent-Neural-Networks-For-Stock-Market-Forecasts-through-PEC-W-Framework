import time
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import matplotlib.dates as mdates
import pywt
import random
import tensorflow as tf
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
start_time = time.time()
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Download historical data for stocks from Yahoo Finance
spy = yf.download('AAPL', start='2000-01-01')
last_date = spy.index[-1]
print("En son verinin tarihi:", last_date)
t = spy['Adj Close'][last_date]
print(t)
t_data = spy['Adj Close']['2000-01-01':'2017-12-31']
v_data = spy['Adj Close']['2018-01-01':'2023-12-31']
tt_data=spy['Adj Close']['2024-01-01':last_date]

vt_data = spy['Open']['2000-01-01':'2017-12-31']
vv_data = spy['Open']['2018-01-01':'2023-12-31']
vtt_data=spy['Open']['2024-01-01':last_date]

scaler = MinMaxScaler(feature_range=(0, 1))
mt_data = scaler.fit_transform(t_data.values.reshape(-1, 1))
mv_data = scaler.transform(v_data.values.reshape(-1, 1))
mtt_data = scaler.transform(tt_data.values.reshape(-1, 1))

mvt_data = scaler.fit_transform(vt_data.values.reshape(-1, 1))
mvv_data = scaler.transform(vv_data.values.reshape(-1, 1))
mvtt_data = scaler.transform(vtt_data.values.reshape(-1, 1))

#DWT FOR PEC-W
def apply_dwt(data, wavelet='db1', level=None):
    if level is None:
        level = int(np.log2(len(data)))

    coeffs = pywt.wavedec(data, wavelet, level=min(level, pywt.dwt_max_level(len(data), pywt.Wavelet(wavelet))))
    flattened_coeffs = [c for sublist in coeffs for c in sublist]
    return flattened_coeffs

#CREATE PEC-W PREPROCESSING
def create_groups(dataset, window_size_2, window_size_3, timeslice_2, timeslice_3, step, d, p):
    X_data, y_data = [], []
    index = 0
    while index + (timeslice_3 * window_size_3) < len(dataset):
        i = 0
        t2, t3 = [], []
        while i < timeslice_3 * window_size_3:
            current_slice = dataset[index + i:index + i + window_size_3]
            if not np.isnan(current_slice).all():
                t3.append(np.mean(current_slice))
            i += window_size_3
        t2 = []
        j = timeslice_3 * window_size_3 - timeslice_2 * window_size_2
        while j < i:
            current_slice = dataset[index + j:index + j + window_size_2]
            if not np.isnan(current_slice).all():
                t2.extend(current_slice)
            j += window_size_2
        t3 = np.array(t3).reshape(-1, 1)
        t2 = np.array(t2).reshape(1, -1)
        t3 = np.transpose(t3)
        m1 = np.mean(t3)
        m2 = np.mean(t2)
        t3 = t3 - m1
        t2 = t2 - m2
        my_ar = np.full((timeslice_1,), m1)
        my_arr = np.full((timeslice_2,), m2)
        my_array = np.concatenate([my_arr, my_ar], axis=0)
        d.append(my_array)
        p.append(m1)
        t3=apply_dwt(t3)
        t2=apply_dwt(t2)
        concatenated_slices = np.concatenate([t2, t3], axis=1)
        X_data.append(concatenated_slices)
        y_data = np.append(y_data, dataset[index + timeslice_3 * window_size_3] - m1)
        index += step
    X_data = np.array(X_data)
    a, b, c = X_data.shape
    array = X_data.reshape(a * b, c)
    return array, np.array(y_data)

#END OF PEC-W PREPROCESSING STR

window_size_1 = 1
window_size_2 = 5
timeslice_1= 4
timeslice_2 = 4
step = 1
xtrm=[]
ytrm=[]
xtem=[]
ytem=[]
l,k=[],[]
#APPLY PEC-W TO ALL ADJ CLOSE PRICE DATASET 
X_train_1, y_train =create_groups(mt_data,window_size_1, window_size_2, timeslice_1,timeslice_2,step,xtrm,ytrm)
X_validation_1, y_validation = create_groups(mv_data,window_size_1, window_size_2, timeslice_1,timeslice_2,step,xtem,ytem)
X_test_1, y_test =create_groups(mtt_data,window_size_1, window_size_2, timeslice_1,timeslice_2,step,l,k)
xtrrm=[]
ytrrm=[]
xteem=[]
yteem=[]
ll,kk=[],[]
#APPLY PEC-W TO ALL OPEN PRICE DATASET
X_train_2, y_train_2 =create_groups(mt_data,window_size_1, window_size_2, timeslice_1,timeslice_2,step,xtrrm,ytrrm)
X_validation_2, y_validation_2 = create_groups(mv_data,window_size_1, window_size_2, timeslice_1,timeslice_2,step,xteem,yteem)
X_test_2, y_test_2 =create_groups(mtt_data,window_size_1, window_size_2, timeslice_1,timeslice_2,step,ll,kk)
#STACK ADJ CLOSE AND OPEN PRICE DATASET
X_train = np.hstack((X_train_1, X_train_2))
X_validation = np.hstack((X_validation_1, X_validation_2))
X_test = np.hstack((X_test_1, X_test_2))
#print("Birleştirilmiş array'in boyutu:", X_train.shape)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_validation = np.reshape(X_validation, (X_validation.shape[0], X_validation.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
model = keras.Sequential()
model.add(keras.layers.LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(units=64, return_sequences=False))
model.add(keras.layers.Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=200, batch_size=64, validation_data=(X_validation, y_validation), callbacks=[early_stopping])
y_test = y_test.reshape(-1, 1)
predicted_stock_price = model.predict(X_test)
k=np.array(k)
k = k.reshape(-1, 1)
y_test=y_test+k
predicted_stock_price=predicted_stock_price+k
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
y_test = scaler.inverse_transform(y_test)


from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_test, predicted_stock_price)
mae = mean_absolute_error(y_test, predicted_stock_price)
r2 = r2_score(y_test,predicted_stock_price)
rmse = np.sqrt(mean_squared_error(y_test, predicted_stock_price))
mape = np.mean(np.abs((y_test -predicted_stock_price) / y_test)) * 100
print('RMSE:', rmse)
print('MAPE:', mape)
print('MSE: ', mse)
print('MAE: ', mae)
print('R-squared: ', r2)

date_range = pd.date_range(start='2024-01-01', periods=len(predicted_stock_price), freq='B')  # 'B' for business day frequency

plt.figure(figsize=(10, 6))
plt.plot(date_range,y_test, label='Actual')
plt.plot(date_range,predicted_stock_price, label='Predicted')
plt.xlabel('Index')
plt.ylabel('Values')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Show tick marks for every 3 months
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Format x-axis labels as 'Jan 2023', 'Feb 2023', etc.
plt.title('APPLE Stock Price Prediction')
plt.legend()
plt.show()
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Timestamp: {elapsed_time} sec")
print( X_train.shape)
print(spy.columns)



