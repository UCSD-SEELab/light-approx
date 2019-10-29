import numpy as np
import pandas as pd
import keras
from keras.layers import Dense, Activation

from sklearn.model_selection import TimeSeriesSplit

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pickle
from keras.models import Model
from keras.layers import Input
from keras.layers.merge import concatenate


def model():
    inputs_weather = Input(shape=(10,),name='weather')
    # weather model
    x1 = Dense(32, activation='relu',name='wea_layer_1')(inputs_weather)
    x1 = Dense(32, activation='relu',name='wea_layer_2')(x1)
    x1 = Dense(32, activation='relu',name='wea_layer_3')(x1)
    out_x1 = Dense(6,activation='relu',name='weather_output')(x1)
    #model1 = Model(inputs=inputs_weather, outputs=out_x1)


    inputs_stationary = Input(shape=(6,),name='stationary')
    # stationary model
    x2 = Dense(32, activation='relu',name='stat_layer_1')(inputs_stationary)
    x2 = Dense(32, activation='relu',name='stat_layer_2')(x2)
    x2 = Dense(32, activation='relu',name='stat_layer_3')(x2)
    out_x2 = Dense(6,activation='relu',name='stationary_output')(x2)
    #model2 = Model(inputs=inputs_stationary, outputs=out_x2)

    #GPS_input = Input(shape=(3,),name='GPS_value')
    GPS_label = Input(shape=(1,),name='GPS_label')
    #middle_layer = concatenate([out_x1,out_x2,GPS_input,GPS_label],axis=1)
    middle_layer = concatenate([out_x1,out_x2,GPS_label],axis=1)

    # merge models
    x = Dense(32, activation='relu', name='weighted')(middle_layer)
    main_out = Dense(6, activation='relu',name='main_output')(x)

    merged_model = Model(inputs=[inputs_weather,inputs_stationary,GPS_label], outputs=[main_out])

    merged_model.compile(loss='mean_squared_error', optimizer='adadelta')