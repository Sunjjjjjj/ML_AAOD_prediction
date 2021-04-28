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

scriptDir = '/usr/people/sunj/Documents/pyvenv/Projects/ML_UVAI/'
dataOutputDir = '/nobackup/users/sunj/'
dataInputDir = '/nobackup_1/users/sunj/'
figdir = '/usr/people/sunj/Dropbox/Paper_Figure/ML_AAOD/'

scriptDir = '/Users/kanonyui/PhD_program/Scripts/ML_UVAI/'
dataOutputDir = '/Users/kanonyui/PhD_program/Data/'
dataInputDir = '/Users/kanonyui/PhD_program/Data/'
figdir = '/Users/kanonyui/Dropbox/Paper_Figure/ML_AAOD/'


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
import string
plotidx = string.ascii_lowercase
# =============================================================================
# training process
# =============================================================================
# loading training data
features = [['AI388', 'AOD550(MODIS)', 'Haer_t1',\
            'vza', 'raa', 'sza', 'As', 'Ps',\
              'lat', 'lon', 'doy'],
            
            ['residue', 'AOD550(MODIS)', 'Haer_t1',\
            'vza', 'raa', 'sza', 'As', 'Ps',\
              'lat', 'lon', 'doy']]

#%%
cmap = matplotlib.cm.twilight_shifted_r
cmap1 = shiftedColorMap(cmap, start=0.5, midpoint=0.6, stop=1, name='shifted')
cmap = matplotlib.cm.cubehelix_r
cmap2 = shiftedColorMap(cmap, start=0, midpoint = 0.5, stop=1, name='shifted')


expNames = ["%s_%i-layer_%i-neuron" % ('DNN_AAOD_train_F11', 5, 2**8), \
            "%s_%i-layer_%i-neuron" % ('DNN_AAOD_train_residue', 3, 2**6)]
titles = ['(a) DNN-F11', '(b) DNN-residue'] 
fig = plt.figure(figsize = (6.5, 2.5))

for i, expName in enumerate(expNames):
    
    data = dataTrain(features[i])
    model = tf.keras.models.load_model(scriptDir + "best_model/best_model_%s" % (expName))
    # prediction
    Y_pred = model(data.X_vld_norm).numpy().reshape(-1)
    slope, intercept, r_value, p_value, std_err = stats.linregress(data.Y_vld, Y_pred)

                                
    ax = fig.add_axes([0.1 + 0.425 * i, 0.175, 0.31, 0.725])
    plt.hist2d(data.Y_vld, Y_pred, bins = 75, cmap = cmap1, 
               norm = matplotlib.colors.LogNorm(), vmin = 1, vmax = 1e2)
    perc = (abs(Y_pred - data.Y_vld) <= data.data.AAOD_err.loc[data.vld_idx]).sum() / len(Y_pred) * 100
    plt.text(1.5e-2, 0.95e-1, r'k: %1.2f  b: %1.2f''\n''$R^2$: %1.2f''\n''RMSE: %1.2e''\n''MAE: %1.2e''\n''P: %02i%%' \
             % (slope, intercept, np.corrcoef(Y_pred, data.Y_vld)[0, 1], 
                RMSE(Y_pred, data.Y_vld), 
                MAE(Y_pred, data.Y_vld), 
                perc), bbox=dict(facecolor='w', edgecolor='k', pad=5, alpha = 0.75))
    plt.text(data.Y_vld.min(), 0.01, '(%s)'.rjust(40) % (plotidx[i]))
    plt.xlim(0, 0.2)
    plt.ylim(0, 0.2)
    plt.xticks(np.arange(0, 0.21, 0.05))
    plt.yticks(np.arange(0, 0.21, 0.05))
    ax.xaxis.get_major_formatter().set_powerlimits((0,1))
    ax.yaxis.get_major_formatter().set_powerlimits((0,1))
    plt.plot(np.arange(0, 10), np.arange(0, 10) * slope + intercept, 'k-', linewidth = 1)
    plt.plot([0, 1], [0, 1], '--', color = 'gray', linewidth = 1)
    plt.plot([0, 0.5], [0, 1], '--', color = 'gray', linewidth = 1)
    plt.plot([0, 1], [0, 0.5], '--', color = 'gray', linewidth = 1)
    plt.xlabel(r'AAOD$^{A}$')
    plt.ylabel(r'AAOD$^{pred}$')
    plt.title(titles[i])
    
     
cax = fig.add_axes([0.875, 0.175, 0.025, 0.725])    
cb = plt.colorbar(cax = cax, fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10, \
                  label = '# num', extend = 'both', )
    # cb.ax.set_yticklabels([5, 10, 50, 100, 500]) 
plt.savefig(figdir + 'Model_performance.png', dpi = 300, transparent = True)
 



#%%

data = dataTrain(features[0])

#%%
X1 = data.data['Absorption_AOD[550nm]']
X2 = data.data.AI388
X3 = data.data.residue


cmap = matplotlib.cm.twilight_shifted_r
cmap1 = shiftedColorMap(cmap, start=0.5, midpoint=0.6, stop=1, name='shifted')

fig = plt.figure(figsize = (6, 2.5))
ax = fig.add_axes([0.1, 0.175, 0.3, 0.7])
plt.hist2d(X1, X2, bins = 100, cmap = cmap1, norm=matplotlib.colors.LogNorm(), vmin = 1, vmax = 1e3)
plt.text(0.15, 0.5, r'$R^2$: %1.2f' % (X1.corr(X2, 'spearman')))
plt.text(0.01, 4.5, '(a)')
plt.ylim(0, 5)
plt.xlabel(r'AAOD$^A$')
plt.ylabel(r'UV Index')

ax = fig.add_axes([0.525, 0.175, 0.3, 0.7])
plt.hist2d(X1, X3, bins = 100, cmap = cmap1, norm=matplotlib.colors.LogNorm(), vmin = 1, vmax = 1e3)
plt.text(0.15, 0.5, r'$R^2$: %1.2f' % (X1.corr(X3, 'spearman')))
plt.text(0.01, 4.5, '(b)')
plt.ylim(0, 5)
plt.xlabel(r'AAOD$^A$')
plt.ylabel(r'Residue')

cax =  fig.add_axes([0.875, 0.175, 0.02, 0.7])
cb = plt.colorbar(cax = cax, fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10, \
              label = '# num', extend = 'both', )


plt.savefig(figdir + 'UVAI_vs_AAOD.png', dpi = 300, transparent = True)



#%%
temp = pd.read_pickle(dataOutputDir + 'MERRA-2_OMAERUV_MODIS_AERONET_collocation/MERRA-2_OMAERUV_MODIS_AERONET_collocation_2006-2019.pickle')
temp['doy'] = temp['dateTimeLocal'].dt.dayofyear + (temp['dateTimeLocal'].dt.hour + temp['dateTimeLocal'].dt.minute / 60 + temp['dateTimeLocal'].dt.second / 3600) / 24
temp['sin_lon'] = np.sin(np.deg2rad(temp['lon']))
temp['sin_doy'] = np.sin(2 * np.pi * temp['doy'] / 365)


features = [
            # ['AI388', 'AOD550(MODIS)', 'Haer_t1',\
            # 'vza', 'raa', 'sza', 'As', 'Ps',\
            #    'lat',  'sin_lon', 'sin_doy'], 
            ['residue', 'AOD550(MODIS)', 'Haer_t1',\
            'vza', 'raa', 'sza', 'As', 'Ps',\
              'lat',  'lon', 'doy'],             
            ]

expNames = [
              # "%s_%i-layer_%i-neuron" % ('DNN_AAOD_train_ST', 4, 2**6), \
            "%s_%i-layer_%i-neuron" % ('DNN_AAOD_train_residue', 2, 2**8), \
            # "%s_%i-layer_%i-neuron" % ('DNN_AAOD_train_F8', 4, 2**7)
            ]
titles = ['(a) DNN-F11', '(b) DNN-residue'] 
fig = plt.figure(figsize = (6.5, 2.5))

for i, expName in enumerate(expNames):
    data = dataTrain(features[i])
    model = tf.keras.models.load_model(scriptDir + "best_model/best_model_%s" % (expName))
    # prediction
    Y_pred = model(standardization(temp[features[i]].values, data.X_train_mean, data.X_train_std)).numpy().reshape(-1)
    X1, X2 = temp['Absorption_AOD[550nm]'].values, Y_pred

    temp['Y_pred'] = Y_pred
    temp.to_pickle(dataOutputDir + 'Validation_all_coincident_%s.pickle' % (expName))
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(X1, X2)
    
    mask = (~np.isnan(X1)) & (~np.isnan(X2))
    X1 = X1[mask]
    X2 = X2[mask]
    temp = temp[mask]
    
                                
    ax = fig.add_axes([0.1 + 0.425 * i, 0.175, 0.31, 0.725])
    plt.hist2d(X1, X2, bins = 75, cmap = cmap1, 
               norm = matplotlib.colors.LogNorm(), vmin = 1, vmax = 1e2)
    # perc = (abs(X1 - X2) <= temp['AAOD_err']).sum() / len(X1) * 100
    plt.text(1e-2, 0.75e-1, r'k: %1.2f  b: %1.2f''\n''$R^2$: %1.2f''\n''RMSE: %1.2e''\n''MAE: %1.2e''\n''P: %02i%%' \
             % (slope, intercept, np.corrcoef(X1, X2)[0, 1], 
                RMSE(X1, X2), 
                MAE(X1, X2), 
                perc))
    plt.text(data.Y_vld.min(), 0.01, '(%s)'.rjust(40) % (plotidx[i]))
    plt.xlim(0, 0.15)
    plt.ylim(0, 0.15)
    plt.xticks(np.arange(0, 0.16, 0.05))
    plt.yticks(np.arange(0, 0.16, 0.05))
    ax.xaxis.get_major_formatter().set_powerlimits((0,1))
    ax.yaxis.get_major_formatter().set_powerlimits((0,1))
    plt.plot(np.arange(0, 10), np.arange(0, 10) * slope + intercept, 'k-', linewidth = 1)
    plt.plot([0, 1], [0, 1], '--', color = 'gray', linewidth = 1)
    plt.plot([0, 0.5], [0, 1], '--', color = 'gray', linewidth = 1)
    plt.plot([0, 1], [0, 0.5], '--', color = 'gray', linewidth = 1)
    plt.xlabel(r'AAOD$^{A}$')
    plt.ylabel(r'AAOD$^{pred}$')
    plt.title(titles[i])
#%%
# INV['AOD550(AERONET)'] = INV['Absorption_AOD[550nm]'] / (1 - INV['Single_Scattering_Albedo[550nm]'])
# AAOD_err = list(map(errorPropagation, np.c_[INV['Single_Scattering_Albedo[550nm]'], INV['AOD550(AERONET)']]))
# INV['AAOD_err'] = np.array(AAOD_err)[:, -1]         


# fig = plt.figure(figsize = (5, 2.5))
# ax = fig.add_axes([0.125 , 0.2, 0.8, 0.7])
# plt.hist(INV.AAOD_err, bins = 35, color = 'gray', alpha=0.7, rwidth=0.7, cumulative = False)
# plt.grid(axis='y', alpha=0.75)
# plt.axvline(INV.AAOD_err.mean(), color = 'royalblue', linestyle = '-', linewidth = 1.5, label = 'Mean error: %1.3f' % (INV.AAOD_err.mean()))
# plt.axvline(INV.AAOD_err.median(), color = 'coral', linestyle = '-', linewidth = 1.5, label = 'Median error: %1.3f' % (INV.AAOD_err.median()))
# # plt.yticks([0, 1e6, 2e6, 3e6])
# ax.yaxis.get_major_formatter().set_powerlimits((0,1))
# plt.xlabel('Estimated error of AERONET AAOD', fontsize = 10)
# plt.ylabel('# num', fontsize = 10)
# plt.legend(frameon = False)
# plt.text(0.07, 1e3-500, '# num: %i''\n''max: %1.3f''\n''min: %1.3f' %(len(INV), INV.AAOD_err.max(), INV.AAOD_err.min()))
# plt.yscale('log')
# plt.savefig(figdir + 'Histogram_AERONET_AAOD_error.png', dpi = 300, transparent = True)

#%% 
# histogram of input features

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

# cmap = matplotlib.cm.twilight_shifted_r
# cmap1 = shiftedColorMap(cmap, start=0.5, midpoint=0.8, stop=1, name='shifted')

# fig = plt.figure(figsize = (8, 4))
# ax = fig.add_axes([0.1, 0.1, 0.7, 0.7])
# temp = data.data.groupby(['Longitude(Degrees)', 'Latitude(Degrees)']).count()
# temp.reset_index(inplace = True)
# bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
#             lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
# bm.drawcoastlines(color='gray',linewidth=1)
# plt.scatter(temp['Longitude(Degrees)'],  temp['Latitude(Degrees)'], c = temp.lat_g,
#             s = 50, vmin = 5e0, vmax = 5e2, edgecolor = 'k', cmap = cmap1,
#             alpha = 0.5, zorder = 10, norm=matplotlib.colors.LogNorm())
# plt.title('Global distribution of training data')
# bm.drawparallels(np.arange(-90, 91, 45), labels=[True,False,False,False], linewidth = 0)
# bm.drawmeridians(np.arange(-180, 181, 60), labels=[False,False,False,True], linewidth = 0)
# cax = fig.add_axes([0.825, 0.1, 0.025, 0.7])
# cb = plt.colorbar(cax = cax, fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10, \
#                   label = '# num', extend = 'both', ticks = [5, 10, 50, 100, 500])
# cb.ax.set_yticklabels([5, 10, 50, 100, 500]) 
# plt.savefig(figdir + 'Global_distribution_training_data.png', dpi = 300, transparent = True)

    

