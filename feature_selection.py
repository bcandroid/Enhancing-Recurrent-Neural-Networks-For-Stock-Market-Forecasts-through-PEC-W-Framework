import time
# Programın başlangıç zamanını kaydet
start_time = time.time()

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
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Download historical data for SPY from Yahoo Finance
spy = yf.download('GOOGL', start='2010-01-01', end='2023-12-31')
print(spy.head())
# Display the columns
print("Columns in SPY DataFrame:", spy.columns)

# Calculate the correlation matrix
correlation_matrix = spy.corr()

# Extract the correlation of all columns with the 'Close' column
close_correlations = correlation_matrix['Adj Close']

# Print the correlations
print("Correlation of columns with 'Close':")
print(close_correlations)

# Identify the column with the highest correlation to 'Close'
best_explanatory_column = close_correlations.drop('Adj Close').idxmax()
highest_correlation_value = close_correlations.drop('Adj Close').max()

print(f"The column that best explains the 'Close' column is '{best_explanatory_column}' with a correlation of {highest_correlation_value:.2f}")

# Programın bitiş zamanını kaydet
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import random
spy.dropna(inplace=True)

# Prepare the features (X) and target (y)
X = spy.drop(columns=['Adj Close'])
y = spy['Adj Close']

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Define the model
model = LinearRegression()

# Perform Recursive Feature Elimination (RFE)
rfe = RFE(estimator=model, n_features_to_select=1)
rfe.fit(X_scaled, y)

# Get the ranking of the features
ranking = rfe.ranking_
feature_names = X.columns

# Print the ranking of the features
feature_ranking = sorted(zip(ranking, feature_names))
print("Feature ranking:")
for rank, name in feature_ranking:
    print(f"{name}: {rank}")

# Plot the feature rankings
plt.figure(figsize=(10, 6))
plt.barh(range(len(feature_ranking)), [rank for rank, name in feature_ranking], tick_label=[name for rank, name in feature_ranking])
plt.xlabel('Ranking')
plt.ylabel('Features')
plt.title('Feature Ranking Using RFE')
plt.show()

# Programın bitiş zamanını kaydet
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")
