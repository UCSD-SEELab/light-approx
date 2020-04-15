import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import mean, std
from sklearn.metrics import r2_score

# label dataset and adjust the sampling rate to 1 min 
def dataset_processing(filename):
    data = pd.read_csv(filename,header=None)
    data.columns = ['time','violet' ,'blue', 'green', 'yellow', 'orange', 'red']
    data['newtime'] = pd.to_datetime(data['time'])
    data = data.resample('1T',on='newtime').mean()
    data = data.interpolate()
    return data

# match the datetime object of the two dataframe
def datetime_matching(sensor1, sensor2):
    new_df1 = pd.DataFrame(columns=['time','violet' ,'blue', 'green', 'yellow', 'orange', 'red'])   
    new_df2 = pd.DataFrame(columns=['time','violet' ,'blue', 'green', 'yellow', 'orange', 'red'])

    # if the datetime index of dataframe 1 is in the index of dataframe2
    for i in range(sensor1.shape[0]):
        if sensor1.index[i] in sensor2.index:
            new_df1 = new_df1.append(sensor1.loc[sensor1.index[i]])

    for i in range(sensor2.shape[0]):
        if sensor2.index[i] in new_df1.index:
            new_df2 = new_df2.append(sensor2.loc[sensor2.index[i]])

    return new_df1, new_df2

# calculate outliers
# outliers may due to daylight saturation
def outlier_calculation(sensor1, sensor2):
    data_diff = sensor2['blue']-sensor1['blue']
    data_mean, data_std = mean(data_diff),std(data_diff)
    # outliers considered as 2 std above or below the mean
    cut_off = data_std*2
    lower, upper = data_mean - cut_off, data_mean + cut_off

    sensor1['diff'] = data_diff
    sensor1['outliers'] = (sensor1['diff']<lower)|(sensor1['diff']>upper)
    return sensor1. sensor2

# plot validation graph
# overlay the data points of two sensors on top of each other
def plot_result(sensor1,sensor2):
    fig= plt.figure(figsize=(15,10))
    plt.scatter(sensor1.index.values,sensor1['blue'], c='b', marker='x', label='sensor1')
    plt.scatter(sensor1.index[sensor1['outliers']].values,sensor1[sensor1['outliers']]['blue'], c='r', marker='.', label='outliers')
    plt.scatter(sensor2.index.values,sensor2['blue'], c='y', marker='o', label='sensor2',s=12)
    plt.legend(loc='upper left')
    plt.xlim(sensor1.index[0],sensor1.index[-1])
    #plt.title('winter colocate data', fontsize = 16)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('matching of 2 sensors values -- second trial',fontsize=16)
    plt.xlabel('datetime',fontsize=14)
    plt.ylabel('blue channel readings',fontsize=14)
    #plt.xticks('time')
    plt.show()

if __name__ == "__main__":
    file1 = "colocate1_new.txt"
    file2 = "colocate2_new.txt"
    sensor1 = dataset_processing(file1)
    sensor2 = dataset_processing(file2)
    sensor1, sensor2 = datetime_matching(sensor1,sensor2)
    plot_result(sensor1,sensor2)

