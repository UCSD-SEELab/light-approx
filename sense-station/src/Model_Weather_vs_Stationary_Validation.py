import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import keras 
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from numpy import mean
from numpy import std

def g_function(amplitude, peak,std,f_range):
    return amplitude*np.exp(-(f_range-peak)**2/(2*std**2))

def power_cal(solar, function, step_size):
    total_power = np.sum(solar*function)*step_size
    return total_power

def get_solar():
    Weather1 = pd.read_csv('weather_2020_11_30.txt',header=None)
    Weather2 = pd.read_csv('weather_2020_12_02.txt',header=None)
    Weather = pd.read_csv('weather_2020_12_01.txt',header=None)
    
    for w in [Weather1,Weather,Weather2]:
        
        w.columns=['DTime','Bar','TempIn','HumIn','TempOut','Wind','Wind10','Wdir','HumOut','RainRate','UV','Solar Radiation']

        for i in range(w.shape[0]):
            w.set_value(i, 'New_Time', datetime.strptime(w.iloc[i,0], '%Y-%m-%d %H:%M:%S'), takeable=False)
#         w = w.resample('1T', on='New_Time').mean()
        
#     solar_radiation1 = Weather1['Solar Radiation']
#     solar_radiation = Weather['Solar Radiation']
#     solar_radiation2 = Weather['Solar Radiation']
# #     solar_radiation.fillna(0)
    three_day = pd.concat([Weather1, Weather,Weather2],axis=0)
    return three_day

def get_stationary():
    stationary = pd.read_csv('colocate2_new.txt')
    stationary = stationary.iloc[15205:19493,:]
    stationary.columns=['time','violet' ,'blue', 'green', 'yellow', 'orange', 'red']
    stationary['newtime'] = pd.to_datetime(stationary['time'])
    stationary = stationary.resample('1T',on='newtime').mean()
    stationary = stationary.fillna(0)
    return stationary

def double_layer():
    model = Sequential()
    model.add(Dense(32,activation='relu',input_dim = 6))
    model.add(Dense(32,activation='relu',input_dim = 6))
    model.add(Dense(units=6))
#     adadelta = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    return model

def plot_history(history):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.title('model loss', fontsize=16)
    plt.xlabel('epoch',fontsize=16)
    plt.ylabel('loss',fontsize=16)
    
def plot_outliers_blue(predictions, test,outliers):
    data_diff = predictions[:,1]-np.array(test.iloc[:,1])
    # data_diff
    data_mean,data_std = mean(data_diff),std(data_diff)
    cut_off = data_std*2
    lower,upper = data_mean-cut_off,data_mean+cut_off
    predict = pd.DataFrame(predictions,columns=['violet','blue','green','yellow','orange','red'])
    predict['diff'] = data_diff
    predict['outliers'] = (predict['diff']<lower)|(predict['diff']>upper)
#     test['diff'] = data_diff
#     test['outliers'] = (test['diff']<lower)|(test['diff']>upper)
#     predict
    x = np.linspace(0, 1, predict.shape[0])
    plt.figure(figsize=(12,10))
    plt.scatter(x,predict['blue'],c='b',label='prediction')
    plt.scatter(x,test['blue'],c='r',label='ground truth')
    plt.legend(fontsize=14)
    plt.title('prediction dataset matching testing data')
    if outliers:
        plt.scatter(predict.index[predict['outliers']].values,predict[predict['outliers']]['blue'], label='lower bound')
#     plt.scatter(x,upper,label='upper bound')
#     return test


if __name__ == '__main__':
    peak=[450,500,550,570,600,650]
    bandwidth = 40
    f_range = np.arange(300, 800, 5) 
    violet = g_function(1,peak[0],bandwidth,f_range)
    blue = g_function(1,peak[1],bandwidth,f_range)
    green = g_function(1,peak[2],bandwidth,f_range)
    yellow = g_function(1,peak[3],bandwidth,f_range)
    orange = g_function(1,peak[4],bandwidth,f_range)
    red = g_function(1,peak[5],bandwidth,f_range)

    stationary = get_stationary()
    weather = get_solar()
    weather = weather.resample('1T', on='New_Time').mean()
    solar = weather['Solar Radiation']

    training = pd.DataFrame(columns = ['violet','blue','green','yellow','orange','red'])
    for i in range(solar.shape[0]):
        violet_power = power_cal(solar[i],violet,5)
        blue_power = power_cal(solar[i],blue,5)
        green_power = power_cal(solar[i],green,5)
        yellow_power = power_cal(solar[i], yellow,5)
        orange_power = power_cal(solar[i],orange,5)
        red_power = power_cal(solar[i],red,5)
        training = training.append({'violet':violet_power,'blue':blue_power,'green':green_power,
                        'yellow':yellow_power,'orange':orange_power,'red':red_power},ignore_index=True)
    
    X_train, X_test, y_train, y_test = train_test_split(training, stationary, test_size = 0.3, random_state = 0)
    model = double_layer()
    history = model.fit(X_train,y_train,batch_size=10,epochs=100)
    plot_history(history)

    y_predict = model.predict(training)
    plot_outliers_blue(y_predict,stationary,False) # exclude outliers version 

main()