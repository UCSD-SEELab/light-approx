"""inspired by this paper: https://dam-prod.media.mit.edu/x/2019/07/31/EMBC2019_MIT_Terumi.pdf"""
from Auto_processing import read_data, match_dates
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from numpy import concatenate
from pandas import concat


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def model(train_X):
    model = Sequential()
    # need to add more layers to make it workable
    # check input shape
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(6))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# plot history


def plot_history(history):
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()


def main():
    gps = ''
    wearable = ''
    stationary = ''
    weather = ''
    # read data
    # in Auto_processing file
    wearable_data, weather_data, stationary_data, gps_data = read_data()
    # matching timestep
    wearable, weather, stationary, gps = match_dates(
        wearable_data, weather_data, stationary_data, gps_data)
    look_back = 5
    scaler = MinMaxScaler(feature_range=(0, 1))
    # combine all dataset together
    concat = pd.concat([weather, stationary, gps, wearable], axis=1)
    # normalize features
    values = concat.values
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    # assume the new col: 0-31 and we need 16+6
    reframed.drop(
        reframed.columns[[22, 23, 24, 25, 26, 27, 28, 29, 30, 31]], axis=1, inplace=True)
    # split into train and test sets
    values = reframed.values
    test_ratio = 0.33
    sample_size = test_ratio*len(reframed)
    train = values[:sample_size, :]
    test = values[sample_size:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-6], train[:, -6:]
    test_X, test_y = test[:, :-6], test[:, -6:]
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    model = model(train_x)
    history = model.fit(train_X, train_y, epochs=50, batch_size=16, validation_data=(
        test_X, test_y), verbose=2, shuffle=False)
    plot_history(history)
    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast
    inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
