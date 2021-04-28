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
matplotlib.rc('font', family='DejaVu Sans')
#dataOutputDir = '/nobackup/users/sunj/'
#dataInputDir = '/nobackup_1/users/sunj/'
#figdir = '/usr/people/sunj/Dropbox/Paper_Figure/ML_AAOD/'

scriptDir = '/Users/kanonyui/PhD_program/Scripts/ML_UVAI/'
dataOutputDir = '/Users/kanonyui/PhD_program/Data/'
dataInputDir = '/Users/kanonyui/PhD_program/Data/'
figdir = '/Users/kanonyui/Dropbox/Paper_Figure/ML_AAOD/'

expName = 'DNN_AAOD_train_residue'
try:
    os.mkdir(dataOutputDir + "%s_output/" % expName)
except:
    print('Output directory already exists!')


cTrain = 'gray'
cVld = 'red'

# load AERONET data
# INV = pd.read_pickle(dataInputDir + 'AERONET/INV_2014-2019.pickle')

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
features = ['residue', 'AOD550(MODIS)', 'Haer_t1',\
            'vza', 'raa', 'sza', 'As', 'Ps',\
              'lat', 'lon', 'doy']
data = dataTrain(features)
data.data = data.data[data.data.residuestd <= 0.5]
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
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
# =============================================================================
# K fold validation
# =============================================================================
t3 = time.time()

# hyperparameter values
num_epochs = [500]
batch_size = [2**6]
# layer_size = [2**6, 2**7, 2**8]
# hidden_layers = [2, 3, 4, 5, 6]
layer_size = [2**6]
hidden_layers = [3]
learning_rate = [1e-4]
activation = ['relu']
optimizer =  ['Adam'] # ['RMSprop', 'Adam']
alpha = [1e-10]


# experiment information
for ilayer in hidden_layers:
    for ineuron in layer_size:
        expName = "%s_%i-layer_%i-neuron" % (expName, ilayer, ineuron)
        train_log_path = scriptDir + "train_log/%s_train_log" % (expName)
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
#  construct DNN model           
# =============================================================================
        model = create_model(hidden_layers= ilayer, 
                              layer_size= ineuron, 
                              learning_rate=1e-4, 
                              activation = 'relu', 
                              optimizer = 'Adam', 
                              alpha = 5e-8)
        records = model.fit(data.X_train_norm, data.Y_train, 
                            batch_size=2**6, epochs=500, 
                            validation_data = (data.X_vld_norm, data.Y_vld),
                            verbose=1, callbacks=[check_point, early_stop])
        # records = model.fit(data.X_norm, data.Y, batch_size=2**6, epochs=500, 
        #                     validation_split = 0.1,
        #                     verbose=1, callbacks=[check_point, early_stop])
        
        # save traininig model
        filelist = sorted(glob.glob(train_log_path + "/*.hdf5"))
        model.load_weights(filelist[-1])
        # save the best model
        tf.saved_model.save(model, scriptDir + "best_model/best_model_%s" % (expName))
# =============================================================================
#  validation
# =============================================================================
        # load best model 
        model = tf.keras.models.load_model(scriptDir + "best_model/best_model_%s" % (expName))
        # prediction
        Y_train_pred = model(data.X_train_norm).numpy().reshape(-1)
        Y_vld_pred = model(data.X_vld_norm).numpy().reshape(-1)
        Y_pred = model(standardization(data.X, data.X_train_mean, data.X_train_std)).numpy().reshape(-1)
        
        
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
        # plt.savefig(figdir + 'Learning_curve_%s.png' % (expName), dpi = 300, transparent = True)
        
        # model evaluation
        stats = plotResult(data.Y_train, data.Y_vld, Y_train_pred, Y_vld_pred)
        plt.savefig(figdir + 'Training_results_%s.png' % (expName), dpi = 300, transparent = True)


        plt.figure(figsize = (4,4))
        plt.hist2d(Y_pred, data.Y, bins = 75, cmap = 'rainbow', norm = matplotlib.colors.LogNorm(), vmin = 1)
        perc = (abs(Y_pred - data.Y) <= data.data.AAOD_err).sum() / len(Y_pred) * 100
        plt.title('R:%1.2f RMSE:%1.4f P:%02i%%' % (np.corrcoef(Y_pred, data.Y)[0, 1], RMSE(Y_pred, data.Y), perc))
        plt.xlim(0, 0.3)
        plt.ylim(0, 0.3)
        # plt.close('all')



t4 = time.time()
print('Time used for parameter tuning: %1.2f s' % (t4 - t3))



