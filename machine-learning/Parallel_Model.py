# this model uses parallel structure 
# and process data frame one by one
import numpy as np
import pandas as pd
import keras
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.model_selection import GridSearchCV


def grid_search_mlp(X,y):
    model = Sequential()
    # Adding the input layer and the first hidden layer
    model.add(Dense(32, activation = 'relu', input_dim = 10))
    # Adding the second hidden layer
    model.add(Dense(units = 32, activation = 'relu'))
    # Adding the third hidden layer
    model.add(Dense(units = 32, activation = 'relu'))
    # Adding the output layer
    model.add(Dense(units = 6))
     
    parameter = [{'optimizer':['adam','sgd','adagrad'],'loss':['mean_squared_error'],'epoch':[50,100,200,500]}]
    search = GridSearchCV(model,parameter,cv=3)
    search.fit(X,y)
    return search.best_params_


def model(best_params_):
    model = Sequential()
    # Adding the input layer and the first hidden layer
    model.add(Dense(32, activation = 'relu', input_dim = 10))
    # Adding the second hidden layer
    model.add(Dense(units = 32, activation = 'relu'))
    # Adding the third hidden layer
    model.add(Dense(units = 32, activation = 'relu'))
    # Adding the output layer
    model.add(Dense(units = 6))

    model.compile(optimizer=best_params_['optimizer'],loss='mean_squared_error')
   
    return model

def split_data_and_train(X,y,model,best_params_):
    sc = StandardScaler()
    tscv = TimeSeriesSplit(n_splits=5)
    ave_acc = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        model.fit(X_train,y_train,batch_size = 10, epochs = best_params_['epoch'])
        prediction = model.predict(X)
        ave_acc.append(model.score(X_test,y_test))
    return ave_acc, model,prediction
    

"""def weighted_function(): 

    need to be adjusted when more data is available"""