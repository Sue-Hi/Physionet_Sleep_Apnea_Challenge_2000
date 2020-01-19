import pandas as pd
import numpy as np
import tempfile
import subprocess
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM, Conv1D, MaxPooling1D, Bidirectional
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from sklearn import metrics


def prepare_data(file_list, n):
    
    beat_restriction_per_min = 30
    sec = 60
    X = []
    y = []
    for fig_idx, file_name in enumerate(file_list):
        
        ### Generating IHR Data from ECG Signals:
        with tempfile.TemporaryFile() as f:
            subprocess.call(['ihr', '-r', file_name , '-a', 'qrsa'], stdout=f)
            f.seek(0)
            ihr = pd.read_csv(f, header=None, delimiter='\s+')
            
        ihr.columns = ["t_sec", "ihr", "block"]
        ihr = ihr[['t_sec', 'ihr']]
    
        f2 = interp1d(ihr.t_sec , ihr.ihr, kind='cubic')
        xnew = np.arange(min(ihr.t_sec), max(ihr.t_sec),1) 
        ihr_regular_interval = pd.DataFrame({'t_sec': xnew,
                                             'ihr': f2(xnew)})
        ### Reading Annotation File:
        with tempfile.TemporaryFile() as f:
            subprocess.call(['rdann', '-r', file_name , '-a', 'apn_measurement', '-f', '0'], stdout=f)
            f.seek(0)
            apn = pd.read_csv(f, header=None, delimiter='\s+')
            
        apn.columns = ['time', 't_csec', 'response_string', 'NU1', 'NU2', 'NU3']
        apn['t_sec'] = apn['t_csec']/100
        apn['response'] = 0
        apn['response'][apn['response_string'] == 'A']= 1
        apn = apn[['t_sec','response']]
        
        ### Annotating Sequences:
        for ix,row in apn.iterrows():
            df1 = ihr[(ihr['t_sec']<=row['t_sec']+sec) & (ihr['t_sec'] >= row['t_sec'])]
        
            if (len(df1) < beat_restriction_per_min):
                continue
            
            df2 =  ihr_regular_interval[(ihr_regular_interval['t_sec']<=row['t_sec']+sec) & (ihr_regular_interval['t_sec'] >= row['t_sec'])].sort_values('t_sec')[:n]
            
            if (len(df2) < n):
                continue
        
            y.append( row['response'] )
            X.append( df2['ihr'].values )
            
    y_np = np.array(y)     
    X_np = np.array(X) 
    
    ### Train-Test Split:
    X_train, X_test, y_train, y_test1 = train_test_split(X_np, y_np, test_size = 0.2, random_state = 42, stratify = y_np)
    
    ### Generating Tensors:
    X_train = X_train.reshape(-1, n, 1)
    X_test = X_test.reshape(-1, n, 1)
    y_train = np_utils.to_categorical(y_train, num_classes = 2)
    y_test = np_utils.to_categorical(y_test1, num_classes = 2)

    return X_train, y_train, X_test, y_test, y_test1


def cnn_lstm_model(filter_num, filter_size, pool_size, LSTM_cell_num, lr):
    """Defines and returns a 2-layer CNN, 1-layer LSTM model"""

    model = Sequential()
    model.add(Conv1D(filter_num,filter_size,activation = 'relu'))
    model.add(MaxPooling1D(pool_size = pool_size))
    model.add(Bidirectional(LSTM(LSTM_cell_num)))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(optimizer=Adam(lr), loss='categorical_crossentropy', metrics= ['accuracy'])
    
    return model


def main(file_list, n, filter_num = 30, filter_size = 5, pool_size = 4, LSTM_cell_num = 100, lr = 0.0001, epochs =2000, batchsize = 50):
    
    X_train, y_train, X_test, y_test, y_test1 = prepare_data(file_list, n)
    model = cnn_lstm_model(filter_num, filter_size, pool_size, LSTM_cell_num, lr)
    np.random.seed(123)
    model.fit(X_train, y_train, epochs = epochs, batch_size = batchsize, shuffle=True, validation_data=(X_test, y_test))
    y_pred = model.predict_proba(X_test)
    fpr, tpr, _ = metrics.roc_curve(np.array(y_test1), y_pred[:,1])
    auc = metrics.auc(fpr,tpr)

    return auc, fpr, tpr

if __name__ == '__main__':
   
    os.chdir("/home/sue/www.physionet.org/physiobank/database/apnea-ecg")
    file_list = ["a01","a02", "a03", "a04", "a05", "a06", "a07", "a08", "a09", "a10", 
             "a11","a12", "a13", "a14", "a15", "a16", "a17", "a18", "a19", "a20",
             "b01", "b02", "b03", "b04", "b05", 
             "c01", "c02", "c03", "c04", "c05", "c06", "c07", "c08", "c09", "c10"]
    n = 60
    filter_num = 30
    filter_size = 5
    pool_size = 4
    LSTM_cell_num = 100
    lr = 0.0001
    epochs = 100
    batchsize = 50
    auc, fpr, tpr = main(file_list, n, filter_num, filter_size, pool_size, LSTM_cell_num, lr, epochs, batchsize)
    print("AUC\n")
    print(auc)
    plt.plot(fpr,tpr)