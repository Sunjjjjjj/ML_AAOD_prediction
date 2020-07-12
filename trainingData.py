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
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors
import seaborn as sns
import pandas as pd
from otherFunctions import *
from supportFunctions import *
from AERONETtimeSeries_v3 import AERONETcollocation
from sklearn.utils import shuffle
from keras import regularizers
from sklearn.model_selection import train_test_split


# initinalization
plt.close('all')
scriptDir = '/usr/people/sunj/Documents/pyvenv/Projects/ML_UVAI/'
dataOutputDir = '/nobackup/users/sunj/'
dataInputDir = '/nobackup_1/users/sunj/'
figdir = '/usr/people/sunj/Dropbox/Paper_Figure/ML_AAOD/'

scriptDir = '/Users/kanonyui/PhD_program/Scripts/ML_UVAI/'
dataOutputDir = '/Users/kanonyui/PhD_program/Data/'
dataInputDir = '/Users/kanonyui/PhD_program/Data/'
figdir = '/Users/kanonyui/Dropbox/Paper_Figure/ML_AAOD/'




# ROI
ROI = {'S': -90, 'N': 90, 'W': -180, 'E': 180}

# temporary functionx
def func(lat, lon):
    return round(lat * 2) / 2, round(lon * 1.6) / 1.6

t1 = time.time()
#%%
# =============================================================================
# load  data
# =============================================================================
# startdate = '%4i-%02i-%02i' % (2014, 1, 1)
# enddate   = '%4i-%02i-%02i' % (2019, 12, 31)
# dates = pd.date_range(startdate, enddate)
# 
#data = pd.DataFrame()
#for idate in dates:
#    sys.stdout.write('\r # %04i-%02i-%02i' % (idate.year, idate.month, idate.day))
#    temp = pd.read_pickle(dataOutputDir + 'MERRA-2_OMAERUV_MODIS_AERONET_collocation/MERRA-2_OMAERUV_MODIS_AERONET_collocation_%4i-%02i-%02i.pickle' \
#                      % (idate.year, idate.month, idate.day))
#    try:
##        mask = (temp['diffAOD'] <= temp['AOD550std'])
#        data = data.append(temp)
#    except:
#        pass
#data = data.reset_index(drop = True)
#data.to_pickle(dataOutputDir + 'MERRA-2_OMAERUV_MODIS_AERONET_collocation/MERRA-2_OMAERUV_MODIS_AERONET_collocation_2014-2019.pickle')
#%%
# =============================================================================
# Prepare training data
# =============================================================================
class dataTrain():
    def __init__(self, features):
        # loading training data sets
        temp = pd.read_pickle(dataOutputDir + 'MERRA-2_OMAERUV_MODIS_AERONET_collocation/MERRA-2_OMAERUV_MODIS_AERONET_collocation_2006-2019.pickle')
        # temp = pd.read_pickle(dataOutputDir + 'MERRA-2_OMAERUV_MODIS_AERONET_collocation/MERRA-2_OMAERUV_MODIS_AERONET_collocation_2014-2019.pickle')
# =============================================================================
# pre-processing
# =============================================================================
        temp['doy'] = temp['dateTimeLocal'].dt.dayofyear + (temp['dateTimeLocal'].dt.hour + temp['dateTimeLocal'].dt.minute / 60 + temp['dateTimeLocal'].dt.second / 3600) / 24
        # MERRA-2 SSA
        temp['SSA'] = 1 - temp['AAOD'] / temp['AOD']
        # OMAERUV at 550 nm
        EAE = Angstorm(388, temp['AOD388'], 500, temp['AOD500'])
        temp['AOD550'] = wvldepAOD(500, temp['AOD500'], 550, EAE)
        
        AAE = Angstorm(388, temp['AAOD388'], 500, temp['AAOD500'])
        temp['AAOD550'] = wvldepAOD(500, temp['AAOD500'], 550, AAE)
        
        temp['SSA550'] = 1 - temp['AAOD550'] / temp['AOD550']  
    
        # AERONET AOD
        temp['AOD500(AERONET)'] = temp['Absorption_AOD[500nm]'] / (1 - temp['Single_Scattering_Albedo[500nm]'])
        temp['AOD550(AERONET)'] = temp['Absorption_AOD[550nm]'] / (1 - temp['Single_Scattering_Albedo[550nm]'])
        # AERONET AAOD uncertainties
        AAOD_err = list(map(errorPropagation, np.c_[temp['Single_Scattering_Albedo[550nm]'], temp['AOD550(AERONET)']]))
        temp['AAOD_err'] = np.array(AAOD_err)[:, -1]         
        # remove NaN and negative values
        mask = (temp['Absorption_AOD[550nm]'] > 0) & (temp['AI388std'] <= 0.5)
        temp = temp[mask]
# =============================================================================
# quality control of AOD and SSA
# =============================================================================
        # OMAERUV SSA and AOD
        temp['diffSSA'] = np.abs(temp['Single_Scattering_Albedo[500nm]'] - temp['SSA500']) <= 0.03
        dAOD = np.abs(temp['AOD500(AERONET)'] - temp['AOD500'])
        temp['diffAOD500'] = False
        temp['diffAOD500'][dAOD <= 0.1] = True
        temp['diffAOD500'][(dAOD > 0.1) & (dAOD / temp['AOD500'] <= 0.3)] = True
        # MODIS AOD
        temp['diffAOD'] = np.nan
        land = temp['landoceanMask'] >= 0.5
        temp.loc[temp['landoceanMask'] >= 0.5, 'diffAOD'] = np.abs(temp['AOD550(AERONET)'][land] - temp['AOD550(MODIS)'][land]) <= (0.05 + 0.15 * temp['AOD550(AERONET)'][land])
        temp.loc[temp['landoceanMask'] < 0.5, 'diffAOD'] = np.abs(temp['AOD550(AERONET)'][~land] - temp['AOD550(MODIS)'][~land]) <= (0.03 + 0.05 * temp['AOD550(AERONET)'][~land])
        
        temp = temp[temp.diffSSA & temp.diffAOD & (temp['AOD550(MODIS)'] >= 0.0)]
# =============================================================================
# aerosol types based on AAE, EAE and SSA
# =============================================================================
        AAE = temp['Absorption_Angstrom_Exponent_440-870nm']
        EAE = temp['Extinction_Angstrom_Exponent_440-870nm-Total']
        SSA440 = temp['Single_Scattering_Albedo[440nm]']
        
        smoke = (AAE >= 0) & (AAE <= 10) & (EAE >= 1.5) & (SSA440 <= 0.95)
        dust = (AAE >= 1)  & (EAE <= 0.5) & (SSA440 <= 0.95) 
        mixed = (AAE >= 0) & (EAE <= 1.5) & (EAE >= 0.5) & (SSA440 <= 0.95)
        urban = (AAE >= 1) & (AAE <= 10) & (EAE >= 1.5) & (EAE <= 2) & (SSA440 > 0.95)
        other = ~(smoke | dust | mixed | urban)
        
        # small = (EAE >= 1.5)
        # large = (EAE <= 1.5)
        
        temp['type'][smoke] = 'BB'
        temp['type'][dust] = 'Dust'
        temp['type'][mixed] = 'Mixed'
        temp['type'][urban] = 'Urban'
        temp['type'][other] = 'Other'
# =============================================================================
# shuffle and split   
# =============================================================================
        data = temp[temp.diffSSA & temp.diffAOD500 & temp.diffAOD & (temp['AOD550(MODIS)'] >= temp['Absorption_AOD[550nm]'])].copy()      
        data = shuffle(data, random_state = 1)
        self.data = data
        
        fig = plt.figure(figsize = (3, 3))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        cmap2 = sns.color_palette("twilight_shifted", 5)
        data.groupby('type').count()['As'].plot(kind = 'pie', colors = cmap2, autopct='%i%%',)
        plt.ylabel('')
        plt.title('Aerosol components in training data')
        plt.savefig(figdir + 'Aerosol_components_training_data.png', dpi = 300, transparent = True)
        
        # testing with a fraction of data to save time
        f = 1
        # feature selection
        # self.features = ['AI388', 'AOD550(MODIS)', 'Haer_t1',  'lat', 'lon', \
        #                   'vza', 'raa', 'As', 'doy', 'Ps']
        self.features = features
        # self.parameters = self.features + ['AAOD500', 'AAOD', 'Absorption_AOD[550nm]', 'residue']
        X = data[self.features].values[::f, :]
#        X[:, -1] *= (1 + np.random.uniform(low=-0.5, high=0.5, size=(len(X),)))
        Y = data['Absorption_AOD[550nm]'].values[::f]
        self.X = X
        self.Y = Y
     
        # training and validation split
        n = 0.90
        self.X_train, self.X_vld, self.Y_train, self.Y_vld = train_test_split(X, Y, train_size = n, test_size = 0.9999 - n, random_state = 33)
        
        # self.X_train = self.data_train[features].values
        # self.X_vld = self.data_vld[features].values
        
        self.X_train_mean, self.X_train_std = self.X_train.mean(axis = 0), self.X_train.std(axis = 0)
        self.Y_train_mean, self.Y_train_std = self.Y_train.mean(axis = 0), self.Y_train.std(axis = 0)
        
        self.X_train_norm = standardization(self.X_train, self.X_train_mean, self.X_train_std)
        self.X_vld_norm = standardization(self.X_vld, self.X_train_mean, self.X_train_std)
        self.X_norm = standardization(X, X.mean(axis = 0), X.std(axis = 0))
        
        self.Y_train_norm = standardization(self.Y_train, self.Y_train_mean, self.Y_train_std)
        self.Y_vld_norm = standardization(self.Y_vld, self.Y_train_mean, self.Y_train_std)
        self.Y_norm = standardization(Y, Y.mean(axis =0), Y.std(axis = 0))
        self.num_train_data, self.num_test_data = self.X_train.shape[0], self.X_vld.shape[0]

    def get_batch(self, batch_size):
        # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, np.shape(self.X_train)[0], batch_size)
        return self.X_train[index, :], self.Y_train[index]






