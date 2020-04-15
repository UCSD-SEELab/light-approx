import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import keras 
from keras.layers import Dense, Activation, Input
from keras.models import Sequential,Model
from sklearn.model_selection import train_test_split,TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from numpy import mean,std
from keras.layers.merge import concatenate
from scipy.interpolate import interp1d

# return color filter
def color_filter(peak, bandwidth, f_range):
    violet = g_function(peak[0],bandwidth,f_range)
    blue = g_function(peak[1],bandwidth,f_range)
    green = g_function(peak[2],bandwidth,f_range)
    yellow = g_function(peak[3],bandwidth,f_range)
    orange = g_function(peak[4],bandwidth,f_range)
    red = g_function(peak[5],bandwidth,f_range)
    return violet, blue, green, yellow, orange, red

# construct six gaussian filters
def g_function(peak,std,f_range):
    return np.exp(-(f_range-peak)**2/(2*std**2))

# calculate the power under the curve
def power_cal(solar, function, step_size):
    total_power = np.sum(solar*function)*step_size
    return total_power

# process and concatenate the sdscweather data
def get_solar():
    Weather1 = pd.read_csv('weather_2020_11_30.txt',header=None)
    Weather2 = pd.read_csv('weather_2020_12_02.txt',header=None)
    Weather = pd.read_csv('weather_2020_12_01.txt',header=None)
    Weather_11_29 = pd.read_csv('weather_2020_11_29.txt',header=None)
    Weather_11_28 = pd.read_csv('weather_2020_11_28.txt',header=None)
    
    for w in [Weather_11_28, Weather_11_29, Weather1,Weather,Weather2]:
        
        w.columns=['DTime','Bar','TempIn','HumIn','TempOut','Wind','Wind10','Wdir','HumOut','RainRate','UV','Solar Radiation']

        for i in range(w.shape[0]):
            w.set_value(i, 'New_Time', datetime.strptime(w.iloc[i,0], '%Y-%m-%d %H:%M:%S'), takeable=False)
    three_day = pd.concat([Weather_11_28,Weather_11_29,Weather1, Weather,Weather2],axis=0)
    return three_day

# get stationary data and match the timestamp with the weather 
def get_stationary():
    stationary = pd.read_csv('colocate2_new.txt')
    stationary = stationary.iloc[12346:19492,:]
    stationary.columns=['time','violet' ,'blue', 'green', 'yellow', 'orange', 'red']
    stationary['newtime'] = pd.to_datetime(stationary['time'])
    stationary = stationary.resample('1T',on='newtime').mean()
    stationary = stationary.fillna(0)
    return stationary

# construct a integrated model 
def double_layer():
    model = Sequential()
    # spectral responses and time are processed together
    model.add(Dense(32,activation='relu',input_dim = 7))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(units=6))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    return model

# construct an end to end model
# time and spectral responses processed separately
def end_to_end():
    # time processor
    inputs_time = Input(shape=(1,),name='time')
    x1 = Dense(32, activation='relu')(inputs_time)
    out_x1 = Dense(1, )(x1)
    
    # spectral readings processor
    inputs_gaussian = Input(shape=(6,),name='gaussian')
    x2 = Dense(32,activation='relu')(inputs_gaussian)
    out_x2 = Dense(6, activation='relu')(x2)
    
    middle_layer = concatenate([out_x1,out_x2],axis=1)
    x = Dense(32, activation='relu', name='weighted')(middle_layer)
    main_out = Dense(6, activation='relu',name='main_output')(x)
    merged_model = Model(inputs=[inputs_time,inputs_gaussian], outputs=[main_out])
    merged_model.compile(optimizer='adam',loss='mean_squared_error')
    return merged_model

# plot model training history 
def plot_history(history):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.title('model loss', fontsize=16)
    plt.xlabel('epoch',fontsize=16)
    plt.ylabel('loss',fontsize=16)

# plot prediction and ground truth in the same figure
# excluding outliers  
def plot_outliers_blue(predictions, test,outliers):
    data_diff = predictions[:,1]-np.array(test.iloc[:,1])
    # data_diff
    data_mean,data_std = mean(data_diff),std(data_diff)
    cut_off = data_std*2
    lower,upper = data_mean-cut_off,data_mean+cut_off
    predict = pd.DataFrame(predictions,columns=['violet','blue','green','yellow','orange','red'])
    predict['diff'] = data_diff
    predict['outliers'] = (predict['diff']<lower)|(predict['diff']>upper)

    x = np.linspace(0, 1, predict.shape[0])
    plt.figure(figsize=(12,10))
    plt.scatter(x,predict['blue'],c='b',label='prediction')
    plt.scatter(x,test['blue'],c='r',label='ground truth')
    plt.legend(fontsize=14)
    plt.title('prediction dataset matching testing data')
    if outliers:
        plt.scatter(predict.index[predict['outliers']].values,predict[predict['outliers']]['blue'], label='lower bound')



if __name__ == '__main__':
    peak=[450,500,550,570,600,650]
    bandwidth = 40
    f_range = np.arange(300, 800, 5) 
    # construct spectral filters
    violet, blue, green, yellow, orange, red = color_filter(peak,bandwidth,f_range)
    
    stationary = get_stationary()
    weather = get_solar()
    weather = weather.resample('1T', on='New_Time').mean()
    solar = weather['Solar Radiation']

    # use Corneal as another input to solve mathematical relationships
    spectralRes = "spectralRes.xlsx"
    sR = pd.read_excel(spectralRes)
    sR.index = sR['nm']
    sR.drop(columns=['nm'],inplace=True)
    corneal = sR['Corneal']
    x = np.array(sR.index)
    y = np.array(corneal).reshape(1,-1)[0]
    # interpolate corneal to the same size of gaussian filter
    f = interp1d(x, y, kind='cubic')
    xnew = np.linspace(380, 780,num=100, endpoint=True)

    # prepare training samples
    training = pd.DataFrame(columns = ['violet','blue','green','yellow','orange','red'])
    for i in range(solar.shape[0]):
        violet_power = power_cal(solar[i],violet*f(xnew),5)
        blue_power = power_cal(solar[i],blue*f(xnew),5)
        green_power = power_cal(solar[i],green*f(xnew),5)
        yellow_power = power_cal(solar[i], yellow*f(xnew),5)
        orange_power = power_cal(solar[i],orange*f(xnew),5)
        red_power = power_cal(solar[i],red*f(xnew),5)
        training = training.append({'violet':violet_power,'blue':blue_power,'green':green_power,
                        'yellow':yellow_power,'orange':orange_power,'red':red_power},ignore_index=True)
    training['time'] = solar.index.hour

    tscv = TimeSeriesSplit()
    model = double_layer()
    merged = end_to_end()
    # use time series split
    # default train-split fold is 5
    for train_index, test_index in tscv.split(training):
        print(train_index)
        X_train, X_test = training.iloc[train_index,:6], training.iloc[test_index,:6]
        solar_train,solar_test = training.iloc[train_index,6], training.iloc[test_index,6]
        y_train, y_test = stationary.iloc[train_index,:], stationary.iloc[test_index,:]
        history = merged.fit([solar_train, X_train], y_train, validation_data=([solar_train, X_train], y_train), batch_size=16, epochs=200)
    
    y_predict = merged.predict([training.iloc[:,-1],training.iloc[:,:-1]])
    mse = mean_squared_error(y_predict,stationary)
    print(mse)
    plot_outliers_blue(y_predict,stationary,False) # exclude outliers version 

main()