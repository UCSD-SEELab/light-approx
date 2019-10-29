"""Assume this file is in machine-learning dir"""
import pandas as pd
import numpy as np
from datetime import datetime

# weather
#from ..feature-extraction.sdsc-weather.src.weatherdata_storage.py import process_data
# gps
#from ..feature-extraction.gps-context.src.in_out_classification.py import main
# wearable --  method?
# stationary -- no method?

"""access and combine all the datasets"""
def read_data():
    wearable_arr = ['4_12_outsided_wearable.xlsx'] # populate with multiple files
    weather_arr = ['weather_4_12.txt']
    stationary_arr = ['4_20_stationary.xlsx']
    gps_arr = ['logger_oct_7.xlsx'] # sample file name

    wearable_path = '../sense-trinket/data/'
    weather_path = '../feature-extraction/sdsc-weather/data/'
    stationary_path = '../sense-station/data/'
    gps_path = '../feature-extraction/gps-context/data/'

    wearable_data = []
    weather_data = []
    stationary_data = []
    gps_data = []

    for i in range(len(wearable_arr)):
        # construct input paths
        wearable = wearable_path+wearable_arr[i]
        weather = weather_path+weather_arr[i]
        stationary = stationary_path+stationary_arr[i]
        gps = gps_path+gps_arr[i]

        # read files
        wearable = pd.read_excel(wearable)
        weather = pd.read_csv(weather) # different dates
        stationary = pd.read_excel(stationary)
        gps = pd.read_excel(gps)
        weather.columns=['DTime','Bar','TempIn','HumIn','TempOut','Wind','Wind10','Wdir','HumOut','RainRate','UV','Solar']

        # match dates for each day
        wearable,weather,stationary,gps = match_dates(wearable,weather,stationary,gps)
        for df in zip(wearable, weather, stationary):
            df.index = range(df.shape[0])

        # drop useless cols
        wearable.drop(columns=['Time(s)'],inplace=True)
        wearable.drop(wearable.columns[0],axis=1,inplace=True)
        stationary.drop(columns=['TimeStamp'],inplace=True)
        # not sure if we still save the features of gps apart from classification

        # stack files of different dates
        wearable_data.append(wearable)
        weather_data.append(weather)
        stationary_data.append(stationary)
        gps_data.append(gps)

    wearable_data = pd.concat(wearable_data)
    weather_data = pd.concat(weather_data)
    stationary_data = pd.concat(stationary_data)
    gps_data = pd.concat(gps_data)

    return wearable_data, weather_data, stationary_data,gps_data


def match_dates(wearable,weather,stationary,gps):
    for i in range(weather.shape[0]):
        weather.set_value(i, 'New_Time', datetime.strptime(weather.iloc[i,0], '%Y-%m-%d %H:%M:%S'), takeable=False)
    
    for i in range(stationary.shape[0]):
        stationary.set_value(i,'New_Time',stationary.iloc[i,0],'%Y-%m-%d %H:%M:%S'), takeable=False)

    for i in range(wearable.shape[0]):
        wearable.set_value(i,'New_Time',wearable.iloc[i,1],'%Y-%m-%d %H:%M:%S'), takeable=False)

    find_start = [wearable.iloc[0,-1],stationary.iloc[0,-1],weather.iloc[0,-1],gps.iloc[0,0]].sort()
    start_time = find_start[-1]

    find_end = [wearable.iloc[-1,-1],stationary.iloc[-1,-1],weather.iloc[-1,-1],gps.iloc[0,-1]].sort()
    end_time = find_end[0]

    for df in zip(wearable,weather,stationary,gps):
        df = df[(df['New_Time']>=start_time) & (df['New_Time']<= end_time)]
        df = df.resample('1T',on='New_Time').mean() # not sure if gps need to be resampled again

    wearable.drop(columns=['Time(s)'],inplace=True)
    wearable.drop(wearable.columns[0],axis=1,inplace=True)
    stationary = stationary.fillna(method='ffill')
    wearable = wearable.fillna(method='ffill')

    return wearable,weather,stationary,gps
    # pd.merge(left=df_with_millions, left_on='date_column',
    #      right=df_with_seven_thousand, right_on='date_column')