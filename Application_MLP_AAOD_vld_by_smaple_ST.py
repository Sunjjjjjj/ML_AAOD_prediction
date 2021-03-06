#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 14:57:41 2018

@author: sunj
"""
import sys 
#sys.path.insert(0, '/usr/people/sunj/Documents/pyvenv/Projects/General_scripts')
sys.path.insert(0, '/Users/kanonyui/PhD_program/Scripts/General_scripts')
sys.path.insert(0, '/Users/kanonyui/PhD_program/Scripts/ML_UVAI')
import sys, os
import shutil, glob
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors
import seaborn as sns
import pandas as pd
from scipy.stats import ks_2samp
from scipy import stats
from otherFunctions import *
from MISR import readMISR
from AERONETtimeSeries_v3 import AERONETcollocation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from supportFunctions import *
from trainingData import *

# initinalization
plt.close('all')
#dataOutputDir = '/nobackup/users/sunj/'
#dataInputDir = '/nobackup_1/users/sunj/'
#figdir = '/usr/people/sunj/Dropbox/Paper_Figure/ML_AAOD/'

scriptDir = '/Users/kanonyui/PhD_program/Scripts/ML_UVAI/'
dataOutputDir = '/Users/kanonyui/PhD_program/Data/'
dataInputDir = '/Users/kanonyui/PhD_program/Data/'
figdir = '/Users/kanonyui/Dropbox/Paper_Figure/ML_AAOD/'

expName = 'DNN_AAOD_train_ST'
try:
    os.mkdir(dataOutputDir + "%s_output/" % expName)
except:
    print('Output directory already exists!')

cTrain = 'gray'
cVld = 'red'

# load AERONET data
INV = pd.read_pickle(dataInputDir + 'AERONET/INV_2014-2019.pickle')

# ROI
ROI = {'S': -90, 'N': 90, 'W': -180, 'E': 180}

# temporary functionx
def func(lat, lon):
    return round(lat * 2) / 2, round(lon * 1.6) / 1.6

t1 = time.time()

plt.close('all')

# =============================================================================
# training process
# =============================================================================
# loading training data
features = ['AI388', 'AOD550(MODIS)', 'Haer_t1',\
            'vza', 'raa', 'sza', 'As', 'Ps',\
              'lat',  'sin_lon', 'sin_doy']
data = dataTrain(features)
num_features = data.X_train.shape[1]

# construct MLP
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import ModelCheckpoint

def create_model(hidden_layers=4, layer_size=2**8, learning_rate=1e-4, activation = 'relu', optimizer = 'RMSprop', alpha = 1e-12):
    layers = [tf.keras.layers.Dense(units=layer_size, activation=activation, input_shape=(num_features,), kernel_regularizer=regularizers.l2(alpha))]
    for _ in range(hidden_layers -1):
        layers.append(tf.keras.layers.Dense(units=layer_size, 
                                            activation=activation, 
                                            kernel_regularizer=regularizers.l2(alpha)))
        # layers.append(tf.keras.layers.Dropout(0.2))
    layers.append(tf.keras.layers.Dense(1))
    model = tf.keras.models.Sequential(layers)
    
    model.compile(optimizer, loss = 'mse', metrics = ['mae', 'mse'])
    return model

# transfer to sklearn model
sk_model = KerasRegressor(create_model)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# =============================================================================
# K fold validation
# =============================================================================
t3 = time.time()
# hyperparameter values
num_epochs = [500]
batch_size = [2**6]
layer_size = [2**6, 2**7, 2**8]
hidden_layers = [1, 2, 3, 4, 5, 6]
# layer_size = [2**8]
# hidden_layers = [5]
learning_rate = [1e-4]
activation = ['relu']
optimizer =  ['Adam'] # ['RMSprop', 'Adam']
alpha = [1e-10]

# K fold split
Kfold = 10
shuffled = data.data.sample(frac = 1)
sub = np.array_split(shuffled, Kfold)



Kfold = 10
model_performance = {}
for ilayer in hidden_layers:
    for ineuron in layer_size:
        train_res = pd.DataFrame()
        for ifold in np.arange(Kfold):
            # experiment information
            testName = "%s_%i-layer_%i-neuron" % (expName, ilayer, ineuron)
            train_log_path = scriptDir + "train_log/%s_train_log/fold-%02i" % (testName, ifold)
            try:
                shutil.rmtree(train_log_path)
            except:
                pass
            os.makedirs(train_log_path)
            checkpoint_path = train_log_path + "/epoch-{epoch:03d}.hdf5"
            # make checkpoint 
            check_point = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor = 'val_loss', save_best_only=True,
                                                             save_weights_only=False, mode='auto', verbose = 1)
# =============================================================================
# training / validation data split
# =============================================================================
            mask = data.data.AI388 < 1e4
            mask[sub[ifold].index] = False
            # training data
            train = data.data[mask]
            vld = sub[ifold]
            print(ifold)
            # standaerization
            X_train_mean = train[features].mean(axis = 0).values
            X_train_std = train[features].std(axis = 0).values
            X_train_norm = standardization(train[features], X_train_mean, X_train_std).values
            X_vld_norm = standardization(vld[features], X_train_mean, X_train_std).values

            Y_train = train['Absorption_AOD[550nm]'].values
            Y_vld = vld['Absorption_AOD[550nm]'].values
# =============================================================================
#  construct DNN model           
# =============================================================================
            model = create_model(hidden_layers= ilayer, 
                                  layer_size= ineuron, 
                                  learning_rate=1e-4, 
                                  activation = 'relu', 
                                  optimizer = 'Adam', 
                                  alpha = 5e-6)
            records = model.fit(X_train_norm, np.log(Y_train), 
                                batch_size=2**6, epochs=500, 
                                validation_data = (X_vld_norm, np.log(Y_vld)),
                                verbose=1, callbacks=[check_point, early_stop])
        # records = model.fit(data.X_norm, data.Y, batch_size=2**6, epochs=500, 
        #                     validation_split = 0.1,
        #                     verbose=1, callbacks=[check_point, early_stop])
            
            # save traininig model
            filelist = sorted(glob.glob(train_log_path + "/*.hdf5"))
            model.load_weights(filelist[-1])
            # save the best model
            tf.saved_model.save(model, scriptDir + "best_model/best_model_%s_fold-%i" % (testName, ifold))
# =============================================================================
#  validation
# =============================================================================
            # load best model 
            model = tf.keras.models.load_model(scriptDir + "best_model/best_model_%s_fold-%i" % (testName, ifold))
            # prediction
            Y_train_pred = np.exp(model(X_train_norm).numpy().reshape(-1))
            Y_vld_pred = np.exp(model(X_vld_norm).numpy().reshape(-1))
            Y_pred = np.exp(model(standardization(data.data[features].values, X_train_mean, X_train_std)).numpy().reshape(-1))
            
            
            # learning curve
            plt.figure(figsize = (6, 4))
            plt.plot(records.epoch, records.history['loss'], color = cTrain, label = 'Train loss')
            plt.plot(records.epoch, records.history['val_loss'], color = cVld,  label = 'Test loss')
            plt.legend()
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Learning curve')
            
            plt.figure(figsize = (6, 4))
            plt.plot(records.epoch, records.history['mae'], '.-', color = cTrain, label = 'Train MAE')
            plt.plot(records.epoch, records.history['val_mae'],'.-', color = cVld,  label = 'Test MAE')
            plt.plot(records.epoch, np.sqrt(records.history['mse']), '--', color = cTrain,label = 'Train RMSE')
            plt.plot(records.epoch, np.sqrt(records.history['val_mse']), '--', color = cVld, label = 'Test RMSE')
            plt.legend()
            plt.xlabel('Epoch')
            plt.ylabel('Metrics')
            plt.title('Learning curve %i-layer %i-neuron' % (ilayer, ineuron))
            # # plt.savefig(figdir + 'Learning_curve_%s.png' % (expName), dpi = 300, transparent = True)
            
            # model evaluation
            stats = plotResult(Y_train, Y_vld, Y_train_pred, Y_vld_pred)
            plt.savefig(figdir + 'Training_results_%s.png' % (testName), dpi = 300, transparent = True)
            
            # plt.figure(figsize = (4,4))
            # plt.hist2d(Y_pred, data.Y, bins = 75, cmap = 'rainbow', norm = matplotlib.colors.LogNorm(), vmin = 1)
            # perc = (abs(Y_pred - data.Y) <= data.data.AAOD_err).sum() / len(Y_pred) * 100
            # plt.title('R:%1.2f RMSE:%1.4f P:%02i%%' % (np.corrcoef(Y_pred, data.Y)[0, 1], RMSE(Y_pred, data.Y), perc))
            # plt.xlim(0, 0.3)
            # plt.ylim(0, 0.3)
            # plt.close('all')
            
            train_res = train_res.append(stats)
        train_res.index = np.arange(1, Kfold + 1)
        model_performance[testName] = train_res

np.save(scriptDir + 'model_performance_%s'% expName, model_performance)

t4 = time.time()
print('Time used for parameter tuning: %1.2f s' % (t4 - t3))



#%%
model_performance = np.load(scriptDir + 'model_performance_%s.npy' % expName, allow_pickle = True).item()
import re

temp_m = pd.DataFrame()
temp_s = pd.DataFrame()
for i, iexp in enumerate(model_performance.keys()):
    sys.stdout.write(r'%s' % (iexp))
    exp = model_performance[iexp]
    exp_mean = exp.mean(axis = 0)
    exp_std = exp.std(axis = 0)
    
    num_l, num_n = re.findall('\d+', iexp)
    exp_mean = pd.DataFrame(exp_mean, columns = [i]).T 
    exp_mean['layer'] = int(num_l)
    exp_mean['neuron'] = int(num_n)
    exp_std = pd.DataFrame(exp_std, columns = [i]).T 
    exp_std['layer'] = int(num_l)
    exp_std['neuron'] = int(num_n)
    temp_m = temp_m.append(exp_mean)
    temp_s = temp_s.append(exp_std)    

print (temp_m)

#%%

# search for the optimal model
# tolerance for overfitting: difference in RMSE <= 5e-4
mask = (abs(temp_m['RMSE_v'] - temp_m['RMSE_t']).round(4) <= 5e-4) * (temp_m['k_v'] >= 0.8)
# find the best model from the remaining
temp_m['rank_RMSE'] = temp_m['RMSE_t'].rank(ascending = True)
temp_m['rank_k'] = temp_m['k_v'].rank(ascending = False)
temp_m['rank_R2'] = temp_m['R2_v'].rank(ascending = False)
temp_m['rank'] = temp_m['rank_RMSE'] + temp_m['rank_k'] + temp_m['rank_R2']
best = temp_m[mask].iloc[temp_m[mask]['rank'].argmin()]

print('The best model is %i-layer %i-neuron' % (best['layer'], best['neuron']))

#%%
# save to csv

for ipara in temp_m.columns:
    if ipara in ['k_t', 'b_t', 'R2_t', 'k_v', 'b_v', 'R2_v']: 
        temp_m[ipara] = temp_m[ipara].map(lambda x: '%1.2f' % x)
    if ipara in ['RMSE_t', 'MAE_t', 'RMSE_v', 'MAE_v']:
        temp_m[ipara] = temp_m[ipara].map(lambda x: '%1.2e' % x)
        
temp_m.index = pd.MultiIndex.from_frame(temp_m[['layer', 'neuron']])
# del temp_m['layer'], temp_m['neuron'], 
# del temp_m['rank'], temp_m['rank_k'], temp_m['rank_R2'], temp_m['rank_RMSE'] 

# temp_m.to_csv('model_performance_ST.csv')





