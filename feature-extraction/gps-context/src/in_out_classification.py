import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from scipy import stats
from sklearn.cluster import KMeans
import sklearn.cluster as cluster
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import pickle

"""Data obtained through Andriod app GPS Logger on Oct 7 to Oct 11"""
""" add new feature sat_ratio to capture useful info"""

filename = ['GPS_logger_Oct_7.txt','GPS_logger_Oct_8.txt','GPS_logger_Oct_9.txt','GPS_logger_Oct_10.txt','GPS_logger_Oct_11.txt']
outputname = ['logger_oct_7.xlsx','logger_oct_8.xlsx','logger_oct_9.xlsx','logger_oct_10.xlsx','logger_oct_11.xlsx']
data_arr = []
#clustering_arr = []

def read_data(filename,outputname,data_arr):
    for i in range(len(filename)):
        logger = pd.read_csv(filename[i],delimiter=',')
        logger.drop(['type','name','desc','bearing(deg)'],axis=1,inplace=True)
        logger['new_time'] = pd.to_datetime(logger['date time']).dt.tz_localize('UTC').dt.tz_convert('US/Pacific').dt.tz_localize(None)
        logger.drop('date time',axis=1,inplace=True)
        logger = logger.resample('1T',on='new_time').mean()
        logger['sat_ratio'] = (logger['sat_used'] / logger['sat_inview']) *1.0
        logger = logger.fillna(-1) # fill empty rows
        logger['classification'] = -1
        for j in range(logger.shape[0]):
            # generally, if the accuracy is low, sat_ratio is low then the participant is indoor
            if logger.iloc[j,8] >= 0.7:
                logger.iloc[j,-1] = 1 # if sat_ratio is higher than 0.7, it is definitely outside
            elif logger.iloc[j,8] < 0: 
                logger.iloc[j,-1] = 0 # the data is not available, it is definitely inside
            elif logger.iloc[j,2] >= 2.5 and logger.iloc[j,8] <= 0.7: 
                logger.iloc[j,-1] = 0 
            elif logger.iloc[j,2] < 2.5 and logger.iloc[j,8] < 0.7:
                # accuracy helps to correct the sat_ratio
                logger.iloc[j,-1] = 1
        #logger.to_excel(outputname[i])      
        data_arr.append(logger)
    return data_arr

def read_annotated_data(filename):
    annotated = pd.read_excel(filename)

    ambuguity = annotated[annotated.classification == -1]

    annotated = annotated[annotated.classification != -1]

    X = annotated[['accuracy(m)','sat_ratio']]
    y = annotated['classification']
    return X,y,ambuguity


def grid_search_svm(X,y):
    clf = SVC(gamma='scale')
    C_list = [{'kernel':['rbf'],'C':[0.001,0.01,0.1],'gamma':[0.05,0.2,1,5]}]
    search = GridSearchCV(clf, C_list,cv=3)
    search.fit(X,y)
    #print (search.best_params_)
    return search.best_params_

def grid_search_rf(X,y):
    clf = RandomForestClassifier()
    parameter = [{'n_estimators':[20,50,100],'max_depth':[5,10,50]}]
    search = GridSearchCV(clf,parameter,cv=3)
    search.fit(X,y)
    return search.best_params_


def svm_train_test(X,y,best_params_):
    clf=SVC(kernel=best_params_['kernel'],C=best_params_['C'],gamma=best_params_['gamma'])
    classify=0
    sc = StandardScaler()
    tscv = TimeSeriesSplit(n_splits=5)
    ave_acc_svm = []
    ave_acc_svm_total = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        classify = clf.fit(X_train,y_train)
        ave_acc_svm_total.append(clf.score(X,y))
        ave_acc_svm.append(clf.score(X_test,y_test))
    return ave_acc_svm, clf

def rf_train_test(X,y,best_params_):
    clf = RandomForestClassifier(n_estimators=search.best_params_['n_estimators'],max_depth=search.best_params_['max_depth'])
    sc = StandardScaler()
    tscv = TimeSeriesSplit(n_splits=5)
    ave_acc_rf = []
    ave_acc_rf_train = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        clf.fit(X_train,y_train)
        ave_acc_rf_train.append(clf.score(X_train,y_train))
        ave_acc_rf.append(clf.score(X_test,y_test))
    return ave_acc_rf,clf

def plot_svm_results(indoor,outdoor,ambuguity):
    plt.figure(figsize=(20,10))
    plt.scatter(indoor.iloc[:,0],indoor.iloc[:,1],c='steelblue',s=20,label='indoor',marker='*')
    plt.scatter(outdoor.iloc[:,0],outdoor.iloc[:,1],c='mediumpurple',s=20,label='outdoor',marker='d')
    plt.scatter(ambuguity.iloc[:,3],ambuguity.iloc[:,9],c='slategray',s=30,label='uncertain',marker='x')
    plt.xlabel('acc',fontsize=24)
    plt.ylabel('sat_ratio', fontsize=24)
    #plt.colorbar().ax.set_ylabel('classification', rotation=-270,fontsize=18)
    plt.title('Prediction by SVM, testsize = 3600',fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    lgnd = plt.legend(loc=0, prop={'size': 18})
    for handle in lgnd.legendHandles:
        handle.set_sizes([60.0])
    plt.show()

def plot_confusion(y,Y_pred):
   
    cm = confusion_matrix(y, Y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ax= plt.subplot()
    akws = {'size':16}
    sns.heatmap(cm, annot=False, ax = ax,annot_kws=akws) #annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels',fontsize=16);
    ax.set_ylabel('True labels',fontsize=16)
    ax.set_title('Confusion Matrix',fontsize=20)
    ax.xaxis.set_ticklabels(['indoor', 'outdoor'],size=16)
    ax.yaxis.set_ticklabels(['indoor', 'outdoor'])
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.text(0.25, 0.75, cm[0][0],
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,color='purple', fontsize=12)
    ax.text(0.25, 0.65, round(cm_norm[0][0],2),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,color='purple', fontsize=12)
    
    ax.text(0.75, 0.75, cm[0][1],
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,color='pink', fontsize=12)
    ax.text(0.75, 0.65, round(cm_norm[0][1],2),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,color='pink', fontsize=12)
    
    ax.text(0.25, 0.25, cm[1][0],
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,color='pink', fontsize=12)
    ax.text(0.25, 0.15, round(cm_norm[1][0],2),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,color='pink', fontsize=12)
    
    ax.text(0.75, 0.25, cm[1][1],
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,color='white', fontsize=12)
    ax.text(0.75, 0.15, round(cm_norm[1][1],2),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,color='white', fontsize=12)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right",va='center',
             rotation_mode="anchor")
    plt.show()
    
    


def main():
    #data_arr = read_data(filename,outputname,data_arr)
    #master_data = pd.concat([data_arr[0],data_arr[1],data_arr[2],data_arr[3],data_arr[4]],axis=0)
    X,y,ambuguity = read_annotated_data('master_annotated.xlsx')
    best_params_ = grid_search_svm(X,y)
    ave_acc_svm,clf = svm_train_test(X,y,best_params_)
    # save the model to current directory
    filename = 'finalized_model.sav'
    pickle.dump(clf, open(filename, 'wb'))
 
    sc = StandardScaler()
    X_sd = sc.fit_transform(X)
    Y_pred = clf.predict(X_sd)
    plot_confusion(y,Y_pred)
    Y_pred= Y_pred.reshape(Y_pred.shape[0],1)
    prediction = np.hstack((X,Y_pred))
    df = pd.DataFrame(prediction,columns=['acc','sat_ratio','classification'])
    indoor = df[df.classification == 0]
    outdoor = df[df.classification == 1]
    plot_svm_results(indoor,outdoor,ambuguity)
    #loaded_model = pickle.load(open(filename, 'rb'))
    #result = loaded_model.score(X_test, Y_test)




#if__name__== "__main__":
main()