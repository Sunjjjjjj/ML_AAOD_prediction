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

scriptDir = '/usr/people/sunj/Documents/pyvenv/Projects/ML_UVAI/'
dataOutputDir = '/nobackup/users/sunj/'
dataInputDir = '/nobackup_1/users/sunj/'
figdir = '/usr/people/sunj/Dropbox/Paper_Figure/ML_AAOD/'

scriptDir = '/Users/kanonyui/PhD_program/Scripts/ML_UVAI/'
dataOutputDir = '/Users/kanonyui/PhD_program/Data/'
dataInputDir = '/Users/kanonyui/PhD_program/Data/'
figdir = '/Users/kanonyui/Dropbox/Paper_Figure/ML_AAOD/'

expName = "%s_%i-layer_%i-neuron" % ('DNN_AAOD_train_ST', 4, 2**6)
try:
    os.mkdir(dataOutputDir + "%s_output/" % expName)
except:
    print('Output directory already exists!')


cTrain = 'gray'
cVld = 'red'


# load AERONET data
INV = pd.read_pickle(dataInputDir + 'AERONET/INV_2005-2019.pickle')

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
model = tf.keras.models.load_model(scriptDir + "best_model/best_model_%s" % (expName))

#%%
cmap = matplotlib.cm.twilight_shifted_r
cmap1 = shiftedColorMap(cmap, start=0.5, midpoint=0.6, stop=1, name='shifted')
cmap = matplotlib.cm.cubehelix_r
cmap2 = shiftedColorMap(cmap, start=0, midpoint = 0.5, stop=1, name='shifted')


# prediction
Y_pred = model(data.X_norm).numpy().reshape(-1)
slope, intercept, r_value, p_value, std_err = stats.linregress(data.Y_vld, Y_pred)


fig = plt.figure(figsize = (4, 4))
ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
plt.hist2d(data.Y_vld, Y_pred, bins = 75, cmap = cmap1, 
           norm = matplotlib.colors.LogNorm(), vmin = 1, vmax = 1e2)
perc = (abs(Y_pred - data.Y_vld) <= data.data.AAOD_err.loc[data.vld_idx]).sum() / len(Y_pred) * 100
plt.text(5e-3, 1e-1, r'k: %1.2f  b: %1.2f''\n''$R^2$: %1.2f''\n''RMSE: %1.3f''\n''MAE: %1.3f''\n''P: %02i%%' \
         % (slope, intercept, np.corrcoef(Y_pred, data.Y_vld)[0, 1], 
            RMSE(Y_pred, data.Y_vld), 
            MAE(Y_pred, data.Y_vld), 
            perc))
plt.xlim(0, 0.15)
plt.ylim(0, 0.15)
plt.xticks(0, 0.15)
plt.yticks(0, 0.15)
ax.xaxis.get_major_formatter().set_powerlimits((0,1))
ax.yaxis.get_major_formatter().set_powerlimits((0,1))
plt.plot(np.arange(0, 10), np.arange(0, 10) * slope + intercept, 'k-', linewidth = 1)
plt.plot([0, 1], [0, 1], '--', color = 'gray', linewidth = 1)
plt.plot([0, 0.5], [0, 1], '--', color = 'gray', linewidth = 1)
plt.plot([0, 1], [0, 0.5], '--', color = 'gray', linewidth = 1)
cax = fig.add_axes([0.875, 0.15, 0.05, 0.7])
cb = plt.colorbar(cax = cax, fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10, \
                  label = '# num', extend = 'both', )
# cb.ax.set_yticklabels([5, 10, 50, 100, 500]) 
# plt.savefig(figdir + 'Model_performance.png', dpi = 300, transparent = True)
 

#%%
# INV['AOD550(AERONET)'] = INV['Absorption_AOD[550nm]'] / (1 - INV['Single_Scattering_Albedo[550nm]'])
# AAOD_err = list(map(errorPropagation, np.c_[INV['Single_Scattering_Albedo[550nm]'], INV['AOD550(AERONET)']]))
# INV['AAOD_err'] = np.array(AAOD_err)[:, -1]         

# fig = plt.figure(figsize = (6, 3))
# ax = fig.add_axes([0.15, 0.2, 0.8, 0.7])
# plt.hist(INV.AAOD_err, bins = 35, color = 'gray', alpha=0.7, rwidth=0.7, cumulative = False)
# plt.grid(axis='y', alpha=0.75)
# plt.axvline(INV.AAOD_err.mean(), color = 'royalblue', linestyle = '-', linewidth = 1.5, label = 'Mean error: %1.3f' % (INV.AAOD_err.mean()))
# plt.axvline(INV.AAOD_err.median(), color = 'coral', linestyle = '-', linewidth = 1.5, label = 'Median error: %1.3f' % (INV.AAOD_err.median()))
# # plt.yticks([0, 1e6, 2e6, 3e6])
# ax.yaxis.get_major_formatter().set_powerlimits((0,1))
# plt.title('Estimated error of AERONET AAOD', fontsize = 10)
# plt.legend(frameon = False)
# plt.text(0.08, 1e3, '# num: %i''\n''max: %1.3f''\n''min: %1.3f' %(len(INV), INV.AAOD_err.max(), INV.AAOD_err.min()))
# plt.yscale('log')
# plt.savefig(figdir + 'Histogram_AERONET_AAOD_error.png', dpi = 300, transparent = True)

#%% 
# histogram of input features
# import string
# plotidx = string.ascii_lowercase
# labels = [r'UVAI$_{388}$', r'AOD$_{550}^{M}$', r'H$_{aer}^{t}$', \
#           'VZA', 'RAA', 'SZA', \
#           r'a$_s$', r'P$_s$', \
#           'Lat', 'Lon', 'DOY']
# fig = plt.figure(figsize = (12, 6))
# for i, ipara in enumerate(features):
#     ax = fig.add_axes([0.05 + (i % 4) * 0.24 , 0.7 - (i // 4) * 0.3, 0.2, 0.2])
#     n, _, _ = plt.hist(data.data[ipara], bins = 25, color = 'gray', alpha=0.7, rwidth=0.7, cumulative = False)
    
#     if ipara in ['AI388', 'AOD550(MODIS)', 'Haer_t1', 'raa', 'As']: 
#         x = data.data[ipara].max() * 0.5
#     else:
#         x = data.data[ipara].min() + 0.1 * abs(data.data[ipara].min())
#     y = n.max() * 0.45
#     plt.text(x, y, 'mean: %1.2f''\n''Std.: %1.2f''\n''max: %1.2f''\n''min: %1.2f' \
#               %(data.data[ipara].mean(), data.data[ipara].std(), data.data[ipara].max(), data.data[ipara].min()
#                 ))
#     ax.yaxis.get_major_formatter().set_powerlimits((0,1))
#     plt.xlabel('(%s) %s' % (plotidx[i], labels[i]))
# plt.savefig(figdir + 'Histogram_features.png', dpi = 300, transparent = True)

# global distribution of training data

cmap = matplotlib.cm.twilight_shifted_r
cmap1 = shiftedColorMap(cmap, start=0.5, midpoint=0.8, stop=1, name='shifted')

fig = plt.figure(figsize = (8, 4))
ax = fig.add_axes([0.1, 0.1, 0.7, 0.7])
temp = data.data.groupby(['Longitude(Degrees)', 'Latitude(Degrees)']).count()
temp.reset_index(inplace = True)
bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
            lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
bm.drawcoastlines(color='gray',linewidth=1)
plt.scatter(temp['Longitude(Degrees)'],  temp['Latitude(Degrees)'], c = temp.lat_g,
            s = 50, vmin = 5e0, vmax = 5e2, edgecolor = 'k', cmap = cmap1,
            alpha = 0.5, zorder = 10, norm=matplotlib.colors.LogNorm())
plt.title('Global distribution of training data')
bm.drawparallels(np.arange(-90, 91, 45), labels=[True,False,False,False], linewidth = 0)
bm.drawmeridians(np.arange(-180, 181, 60), labels=[False,False,False,True], linewidth = 0)
cax = fig.add_axes([0.825, 0.1, 0.025, 0.7])
cb = plt.colorbar(cax = cax, fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10, \
                  label = '# num', extend = 'both', ticks = [5, 10, 50, 100, 500])
cb.ax.set_yticklabels([5, 10, 50, 100, 500]) 
plt.savefig(figdir + 'Global_distribution_training_data.png', dpi = 300, transparent = True)

    

#%%
# =============================================================================
# case application
# =============================================================================
t3 = time.time()
caseROI = {
            '1': {'S': -75, 'N': 75, 'W': -180, 'E': 180},
           }

caseDate = {
            '1': pd.date_range('2019-01-01', '2019-12-31'),
            }    
 
parameters = ['Haer_e', 'Haer_a', 'Haer_63', 'Haer_p', 'Haer_t1', 'AOD']


# validation data (collocated with AERONET)
dataVal = pd.DataFrame()
for icase in caseROI.keys():
    for idate in caseDate[icase][::1]:
        sys.stdout.write('\r Applying to %s %04i-%02i-%02i' % (icase, idate.year, idate.month, idate.day))

        dataTest = pd.read_pickle(dataOutputDir + 'MERRA-2_OMAERUV_MODIS_collocation/MERRA-2_OMAERUV_MODIS_collocation_%04i-%02i-%02i.pickle' 
                              % (idate.year, idate.month, idate.day))
        
        if len(dataTest) > 0:
            dataTest['doy'] = dataTest['dateTime'].dt.dayofyear + (dataTest['dateTime'].dt.hour + dataTest['dateTime'].dt.minute / 60 + dataTest['dateTime'].dt.second / 3600) / 24
            dataTest['sin_lon'] = np.sin(np.deg2rad(dataTest['lon']))
            dataTest['sin_doy'] = np.sin(2 * np.pi * dataTest['doy'] / 365)

# =============================================================================
# prediction
# =============================================================================
            # quality control
            mask = (dataTest.CF >= 0.) & (dataTest.AI388std <= 0.5) 
            dataTest = dataTest.dropna(how = 'any')
            dataTest = dataTest[mask].reset_index(drop = True)
        
            X_test = dataTest[data.features].values
            X_test_norm = standardization(X_test, data.X_train_mean, data.X_train_std)
            # prediction 
            dataTest['Y_pred'] = model(X_test_norm).numpy().reshape(-1)
            dataTest.to_pickle(dataOutputDir + '%s_output/%s_output_%02i-%02i-%02i.pickle' % (expName, expName, idate.year, idate.month, idate.day))
# =============================================================================
#  read AERONET
# =============================================================================
        #    sys.stdout.write('\r # AERONET %04i-%02i-%02i' % (idate.year, idate.month, idate.day))
        #    AERONET = pd.read_pickle(dataOutputDir + 'MERRA-2_OMAERUV_AERONET_collocation/MERRA-2_OMAERUV_AERONET_collocation_%4i-%02i-%02i.pickle' \
        #                         % (idate.year, idate.month, idate.day))
        #    AERONET = AERONET[AERONET['AAOD500'] >= 0].reset_index(drop = True)
        #    AERONET_mean = AERONET.groupby('Site').mean()
        #    AERONET_std = AERONET.groupby('Site').std()
            AERONET = INV[INV['dateTime'].dt.date == idate.date()]
            # collocation pixels
            temp = AERONETcollocation(AERONET, dataTest, [idate.date()],  3 * 3600, 50)
            if len(temp) > 0:
                dataCol =  temp.groupby('Site').mean()
                dataCol['Absorption_AOD[550nm]_std'] = temp.groupby('Site').std()['Absorption_AOD[550nm]']
                dataCol['Absorption_AOD[500nm]_std'] = temp.groupby('Site').std()['Absorption_AOD[500nm]']
                dataCol['Single_Scattering_Albedo[550nm]_std'] = temp.groupby('Site').std()['Single_Scattering_Albedo[550nm]']
                dataCol['Single_Scattering_Albedo[500nm]_std'] = temp.groupby('Site').std()['Single_Scattering_Albedo[500nm]']
                dataVal = dataVal.append(dataCol)
            else:
                print('No validation pixels for %4i-%02i-%02i' % (idate.year, idate.month, idate.day))
dataVal.to_pickle(dataOutputDir + '%s_output/%s_output_validation.pickle' % (expName, expName))


t4 = time.time()
print('Time used for parameter tuning: %1.2f s' % (t4 - t3))


#%%
dataVal = pd.read_pickle(dataOutputDir + '%s_output/%s_output_validation.pickle' % (expName, expName))

dataVal['AOD500(AERONET)'] = dataVal['Absorption_AOD[500nm]'] / (1 - dataVal['Single_Scattering_Albedo[500nm]'])
dataVal['AOD550(AERONET)'] = dataVal['Absorption_AOD[550nm]'] / (1 - dataVal['Single_Scattering_Albedo[550nm]'])

# OMAERUV SSA and AOD
dataVal['diffSSA'] = np.abs(dataVal['Single_Scattering_Albedo[500nm]'] - dataVal['SSA500']) <= 0.03
dAOD = np.abs(dataVal['AOD500(AERONET)'] - dataVal['AOD500'])
dataVal['diffAOD500'] = False
dataVal['diffAOD500'][dAOD <= 0.1] = True
dataVal['diffAOD500'][(dAOD > 0.1) & (dAOD / dataVal['AOD500'] <= 0.3)] = True
# MODIS AOD
dataVal['diffAOD'] = np.nan
land = dataVal['landoceanMask'] >= 0.5
dataVal.loc[dataVal['landoceanMask'] >= 0.5, 'diffAOD'] = np.abs(dataVal['AOD550(AERONET)'][land] - dataVal['AOD550(MODIS)'][land]) <= (0.05 + 0.15 * dataVal['AOD550(AERONET)'][land])
dataVal.loc[dataVal['landoceanMask'] < 0.5, 'diffAOD'] = np.abs(dataVal['AOD550(AERONET)'][~land] - dataVal['AOD550(MODIS)'][~land]) <= (0.03 + 0.05 * dataVal['AOD550(AERONET)'][~land])

AAOD_err = list(map(errorPropagation, np.c_[dataVal['Single_Scattering_Albedo[550nm]'], dataVal['AOD550(AERONET)']]))
dataVal['AAOD_err'] = np.array(AAOD_err)[:, -1]         

dataVal['diffAAOD'] = (abs(dataVal['Y_pred'] - dataVal['Absorption_AOD[550nm]']) <= dataVal.AAOD_err)

# OMAERUV at 550 nm
EAE = Angstorm(388, dataVal['AOD388'], 500, dataVal['AOD500'])
dataVal['AOD550'] = wvldepAOD(500, dataVal['AOD500'], 550, EAE)

AAE = Angstorm(388, dataVal['AAOD388'], 500, dataVal['AAOD500'])
dataVal['AAOD550'] = wvldepAOD(500, dataVal['AAOD500'], 550, AAE)

dataVal['SSA550'] = 1 - dataVal['AAOD550'] / dataVal['AOD550']  

dataVal['SSA_pred'] = 1- dataVal['Y_pred'] / dataVal['AOD550(MODIS)']
dataVal['SSA'] = 1- dataVal['AAOD'] / dataVal['AOD']
dataVal = dataVal[dataVal['SSA_pred'] >= 0]
 


mask = dataVal.diffAOD & dataVal.diffAAOD & (dataVal['AOD550(MODIS)'] >= 0.01) & (~np.isinf(dataVal['AAOD550']))
site = dataVal[mask].groupby(['Site']).mean()



for ipara in ['Y_pred', 'AAOD', 'AAOD550']:
    plt.figure(figsize = (8, 14))
    X1, X2 = site['Absorption_AOD[550nm]'], site[ipara]
    plt.subplot(311)
    bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
                lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
    bm.drawcoastlines(color='gray',linewidth=1)
    plt.scatter(site['Longitude(Degrees)'],  site['Latitude(Degrees)'], c= X1, 
                s = 50, vmin = 0, vmax = 0.1, edgecolor = 'k')
    
    plt.subplot(312)
    bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
                lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
    bm.drawcoastlines(color='gray',linewidth=1)
    plt.scatter(site['Longitude(Degrees)'],  site['Latitude(Degrees)'], c= X2, 
                s = 50, vmin = 0, vmax = 0.1, edgecolor = 'k')
    
    plt.subplot(313)
    bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
                lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
    bm.drawcoastlines(color='gray',linewidth=1)
    plt.scatter(site['Longitude(Degrees)'],  site['Latitude(Degrees)'], c= X2 - X1, cmap = 'coolwarm', 
                s = 50, vmin = -5e-3, vmax = 5e-3, edgecolor = 'k')
    
    print(RMSE(X1, X2), X1.corr(X2))
    

#%%
for ipara in ['SSA_pred', 'SSA', 'SSA550']:
    plt.figure(figsize = (8, 14))
    X1, X2 = site['Single_Scattering_Albedo[550nm]'], site[ipara]
    plt.subplot(311)
    bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
                lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
    bm.drawcoastlines(color='gray',linewidth=1)
    plt.scatter(site['Longitude(Degrees)'],  site['Latitude(Degrees)'], c= X1, 
                s = 50, vmin = 0.5, vmax = 1, edgecolor = 'k')
    
    plt.subplot(312)
    bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
                lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
    bm.drawcoastlines(color='gray',linewidth=1)
    plt.scatter(site['Longitude(Degrees)'],  site['Latitude(Degrees)'], c= X2, 
                s = 50, vmin = 0.5, vmax = 1, edgecolor = 'k')
    
    plt.subplot(313)
    bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
                lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
    bm.drawcoastlines(color='gray',linewidth=1)
    plt.scatter(site['Longitude(Degrees)'],  site['Latitude(Degrees)'], c= X2 - X1, cmap = 'coolwarm', 
                s = 50, vmin = -3e-2, vmax = 3e-2, edgecolor = 'k')
    
    print('+/-0.03 %02i %%' % ((abs(X1 - X2)<= 0.03).sum() / len(X1) * 1e2))