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
# from MISR import readMISR
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
matplotlib.rc('font', family='DejaVu Sans', size = 10)

scriptDir = '/usr/people/sunj/Documents/pyvenv/Projects/ML_UVAI/'
dataOutputDir = '/nobackup/users/sunj/'
dataInputDir = '/nobackup_1/users/sunj/'
figdir = '/usr/people/sunj/Dropbox/Paper_Figure/ML_AAOD/'

scriptDir = '/Users/kanonyui/PhD_program/Scripts/ML_UVAI/'
dataOutputDir = '/Users/kanonyui/PhD_program/Data/'
dataInputDir = '/Users/kanonyui/PhD_program/Data/'
figdir = '/Users/kanonyui/Dropbox/Paper_Figure/ML_AAOD/'

# expName = "%s_%i-layer_%i-neuron" % ('DNN_AAOD_train_F11', 5, 2**8)
expName = "%s_%i-layer_%i-neuron" % ('DNN_AAOD_train_residue', 3, 2**6)

try:
    os.mkdir(dataOutputDir + "%s_output/" % expName)
except:
    print('Output directory already exists!')


cTrain = 'gray'
cVld = 'red'

import string
plotidx = string.ascii_lowercase

# load AERONET data
INV = pd.read_pickle(dataInputDir + 'AERONET/INV_2005-2019.pickle')

# ROI
ROI = {'S': -90, 'N': 90, 'W': -180, 'E': 180}

cmap = matplotlib.cm.twilight_shifted_r
cmap1 = shiftedColorMap(cmap, start=0.5, midpoint=0.6, stop=1, name='shifted')
cmap = matplotlib.cm.cubehelix_r
cmap2 = shiftedColorMap(cmap, start=0, midpoint = 0.5, stop=1, name='shifted')



t1 = time.time()

plt.close('all')

# =============================================================================
# training process
# =============================================================================
# loading training data
# paras = ['AI388',  'Haer_t1',\
#             'vza', 'raa', 'sza', 'As', 'Ps', 'CF', 'timeStamp', 'dateTime', \
#             'lat', 'lon', 'lat_g', 'lon_g',\
#              'Y_pred', 'AOD550(MODIS)', 'SSA_pred',\
#             'AOD', 'AAOD', 'SSA',\
#             'AOD500', 'AAOD500', 'SSA500',\
#             'AOD550', 'AAOD550', 'SSA550']

# =============================================================================
# case application
# =============================================================================
# t3 = time.time()

# ROI = {'S': -75, 'N': 75, 'W': -180, 'E': 180}
# dates = pd.date_range('2006-01-01', '2019-12-31')
# seasons = (dates.month % 12 + 3) // 3

# years = np.arange(2019, 2020)
# months = np.arange(1, 13)
# # validation data (collocated with AERONET)

# for yy in years:
#     for mm in months: 
#         mask = (dates.year == yy) & (dates.month == mm)
#         dataTest = pd.DataFrame()
#         for idate in dates[mask]:
#             try: 
#                 sys.stdout.write('\r Reading %04i-%02i-%02i' % (idate.year, idate.month, idate.day))
#                 temp = pd.read_pickle(dataOutputDir + '%s_output/%s_output_%04i-%02i-%02i.pickle'  
#                                       % (expName, expName, idate.year, idate.month, idate.day))
                
#                 # DNN and MERRA-2 SSA
#                 # temp.loc[temp['AOD550(MODIS)'] < temp['Y_pred'], 'Y_pred'] = np.nan
#                 # temp.loc[temp['AOD550(MODIS)'] < 0.1, 'Y_pred'] = np.nan
#                 # temp.loc[temp['AI388'] < 0, 'Y_pred'] = np.nan
#                 temp.loc[temp['AOD550(MODIS)'] < 0.1, 'Y_pred'] = np.nan
#                 temp['SSA_pred'] = 1 - temp['Y_pred'] / temp['AOD550(MODIS)']
#                 temp['SSA'] = 1 - temp['AAOD'] / temp['AOD']
                
#                 temp.loc[temp['AOD500'] < 0, 'AOD500'] = np.nan
#                 temp.loc[temp['AAOD500'] < 0, 'AAOD500'] = np.nan
#                 temp.loc[temp['SSA500'] < 0, 'SSA500'] = np.nan
#                 temp.loc[(temp['SSA_pred'] < 0) | (temp['SSA_pred'] > 1) | (np.isinf(temp['SSA_pred'])) , 'SSA_pred'] = np.nan
        
#                 # OMAERUV 550 nm
#                 EAE = Angstorm(388, temp['AOD388'], 500, temp['AOD500'])
#                 temp['AOD550'] = wvldepAOD(500, temp['AOD500'], 550, EAE)
#                 AAE = Angstorm(388, temp['AAOD388'], 500, temp['AAOD500'])
#                 temp['AAOD550'] = wvldepAOD(500, temp['AAOD500'], 550, AAE) 
#                 temp['SSA550'] = 1 - temp['AAOD550'] / temp['AOD550']
                
#                 temp.loc[temp['AOD550'] < 0, 'AOD550'] = np.nan
#                 temp.loc[temp['AAOD550'] < 0, 'AAOD550'] = np.nan
#                 temp.loc[temp['SSA550'] < 0, 'SSA550'] = np.nan
                
                
               
#                 # temp = temp.groupby(['lat_g', 'lon_g']).mean()
#                 # temp.reset_index(inplace = True) 
#             except:
#                 temp = pd.DataFrame()
#                 print('No prediction for %4i-%02i-%02i' % (idate.year, idate.month, idate.day))
            

#             dataTest = dataTest.append(temp)
#         dataTest = dataTest.groupby(['lat_g', 'lon_g']).mean()
#         dataTest.reset_index(inplace = True) 
            
#         dataTest.to_pickle(dataOutputDir + '%s_output/%s_output_AOD-0.10_%4i-%02i.pickle' % (expName, expName, yy, mm))
       
# t4 = time.time()
# print('Time used for parameter tuning: %1.2f s' % (t4 - t3))

#%%
# temp = dataTest
# temp.loc[temp['AOD550(MODIS)'] < temp['Y_pred'], 'SSA_pred'] = np.nan
# temp.loc[temp['AOD550(MODIS)'] < 0.1, 'SSA_pred'] = np.nan
# temp = temp[temp['AOD550(MODIS)'] > 0.1]
# temp = temp[temp['AI388'] > -100]
# temp = temp.groupby(['lon_g', 'lat_g']).mean()
# temp.reset_index(inplace = True)

# fig = plt.figure(figsize = (8, 4))
# bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
#             lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
# bm.drawcoastlines(color='gray',linewidth=1)
# plt.scatter(temp.lon_g, temp.lat_g, c = temp.Y_pred, cmap = cmap1, s = 4)
# plt.colorbar()

# fig = plt.figure(figsize = (8, 4))
# bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
#             lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
# bm.drawcoastlines(color='gray',linewidth=1)
# plt.scatter(temp.lon_g, temp.lat_g, c = temp.AI388, cmap = cmap1, s = 2, vmin = 0, vmax = 2)
# plt.colorbar()

# fig = plt.figure(figsize = (8, 4))
# bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
#             lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
# bm.drawcoastlines(color='gray',linewidth=1)
# plt.scatter(temp.lon_g, temp.lat_g, c = temp.residue, cmap = cmap1, s = 2, vmin = 0, vmax = 2)
# plt.colorbar()

# fig = plt.figure(figsize = (8, 4))
# bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
#             lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
# bm.drawcoastlines(color='gray',linewidth=1)
# plt.scatter(temp.lon_g, temp.lat_g, c = temp['AOD550(MODIS)'], cmap = cmap1, s = 2, vmax = 1)
# plt.colorbar()

# fig = plt.figure(figsize = (8, 4))
# bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
#             lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
# bm.drawcoastlines(color='gray',linewidth=1)
# plt.scatter(temp.lon_g, temp.lat_g, c = temp.SSA_pred, cmap = 'cubehelix', s = 2, vmin = 0.7)
# plt.colorbar()

# fig = plt.figure(figsize = (8, 4))
# bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
#             lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
# bm.drawcoastlines(color='gray',linewidth=1)
# plt.scatter(temp.lon_g, temp.lat_g, c = temp.SSA500, cmap = 'cubehelix', s = 2, vmin = 0.7)
# plt.colorbar()
#%%
ROI = {'S': -75, 'N': 75, 'W': -180, 'E': 180}
dates = pd.date_range('2006-01-01', '2019-12-31')
seasons = (dates.month % 12 + 3) // 3

years = np.arange(2006, 2020)
months = np.arange(1, 13)
# validation data (collocated with AERONET)

dataMonth = pd.DataFrame()
for yy in years:
    for mm in months: 
        mask = (dates.year == yy) & (dates.month == mm)
        try: 
            sys.stdout.write('\r Reading %4i-%02i' % (yy, mm))
            temp = pd.read_pickle(dataOutputDir + '%s_output/%s_output_AOD-0.10_%04i-%02i.pickle'  
                                  % (expName, expName, yy, mm))
            temp.loc[np.isinf(temp['AAOD550']), 'AAOD550'] = np.nan
            dataMonth = dataMonth.append(temp)
        except:
            print('No prediction for %4i-%02i' % (yy, mm))


dataMonth = dataMonth[(dataMonth['Y_pred'] > 0) & (dataMonth['Y_pred'] < dataMonth['AOD550(MODIS)'])]
dataMonth['YYMM'] = pd.to_datetime(dataMonth.timeStamp * 1e9).dt.to_period('M')
dataMonth['season'] = (dataMonth.YYMM.dt.month % 12 + 3) // 3

# #%%
# mask = (dataMonth['AOD550(MODIS)'] >= 0.1) & (dataMonth.AI388 >= -10) & (dataMonth.landoceanMask <= 1)

# temp = dataMonth[mask].groupby(['lat_g', 'lon_g']).mean()
# plt.figure()
# plt.scatter(temp.lon, temp.lat, c =  temp.SSA_pred, s = 4, vmin = 0.8, vmax = 1)
# plt.colorbar()

# plt.figure()
# plt.scatter(temp.lon, temp.lat, c =  temp.AI388, s = 4, vmax = 2)
# plt.colorbar()

# plt.figure()
# plt.scatter(temp.lon, temp.lat, c =  temp.Y_pred, s = 4, vmax = 5e-2)
# plt.colorbar()


dataMonth.loc[dataMonth['AOD550(MODIS)'] < dataMonth['Y_pred'], 'SSA_pred'] = np.nan
dataMonth.loc[dataMonth['AOD550(MODIS)'] < 0.1, 'SSA_pred'] = np.nan
dataMonth.loc[dataMonth['AOD550(MODIS)'] < 0.1, 'Y_pred'] = np.nan
dataMonth.loc[dataMonth['AOD550'] < 0.1, 'AAOD550'] = np.nan
dataMonth.loc[dataMonth['AOD550'] < 0.1, 'SSA550'] = np.nan
dataMonth.loc[dataMonth['AOD'] < 0.1, 'AAOD'] = np.nan
dataMonth.loc[dataMonth['AOD'] < 0.1, 'SSA'] = np.nan
dataMonth.loc[dataMonth['residuestd'] > 0.5, 'SSA_pred'] = np.nan
dataMonth.loc[dataMonth['residuestd'] > 0.5, 'Y_pred'] = np.nan
# dataMonth.loc[dataMonth['AI388'] < 0, 'SSA_pred'] = np.nan

dataSeason = dataMonth.groupby(['lat_g', 'lon_g', 'season']).mean()
dataSeason.reset_index(inplace = True)


dataSeason.loc[dataSeason['AOD550(MODIS)'] < dataSeason['Y_pred'], 'SSA_pred'] = np.nan
dataSeason.loc[dataSeason['AOD550(MODIS)'] < 0.1, 'SSA_pred'] = np.nan
dataSeason.loc[dataSeason['AOD550(MODIS)'] < 0.1, 'Y_pred'] = np.nan
dataSeason.loc[dataSeason['AOD550'] < 0.1, 'AAOD550'] = np.nan
dataSeason.loc[dataSeason['AOD550'] < 0.1, 'SSA550'] = np.nan
dataSeason.loc[dataSeason['AOD'] < 0.1, 'AAOD'] = np.nan
dataSeason.loc[dataSeason['AOD'] < 0.1, 'SSA'] = np.nan
# dataSeason.loc[dataSeason['AI388'] < 0, 'SSA_pred'] = np.nan
                
#%% global-seasonal map
cmap = matplotlib.cm.cubehelix_r
cmap2 = shiftedColorMap(cmap, start=0, midpoint = 0.5, stop=1, name='shifted')

# ROI
ROI = {'S': -90, 'N': 90, 'W': -180, 'E': 180}
seasons = ['DJF', 'MAM', 'JJA', 'SON']
labels = ['DNN-F11', 'OMAERUV', 'MERRA-2']

for lo in ['land', 'ocean']:
    c = 0
    fig = plt.figure(figsize = (8, 6.5))
    for isea in range(1, 5):
        if lo == 'land': 
            temp = dataSeason[(dataSeason.season == isea) & \
                              (dataSeason['AOD550(MODIS)'] > 0.1) & \
                              (dataSeason['residue'] > -1) & \
                              (dataSeason['landoceanMask'] > 0.1)] 
        if lo == 'ocean': 
            temp = dataSeason[(dataSeason.season == isea) & \
                              (dataSeason['AOD550(MODIS)'] > 0.1) & \
                               (dataSeason['residue'] > -1) & \
                              (dataSeason['landoceanMask'] < 0.1)]         
        lat = temp['lat_g']
        lon = temp['lon_g']
        
        for i, ipara in enumerate(['Y_pred', 'AAOD550', 'AAOD']):
            # plt.subplot(4, 3, c + 1)
            ax = fig.add_axes([0.03 + i * 0.325, 0.77 - (isea - 1) * 0.2, 0.275, 0.17])
            X = temp[ipara] 
            bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
                        lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
            bm.drawcoastlines(color='gray',linewidth=1)
            bm.drawparallels(np.arange(-45, 46, 45), labels=[False,True,False,False], linewidth = 0, fontsize = 8)
            bm.drawmeridians(np.arange(-90, 91, 90), labels=[False,False,False,True], linewidth = 0, fontsize = 8)
            XX, YY = np.meshgrid(np.arange(-180, 181, 10), np.arange(-90, 91, 10))
            plt.scatter(XX, YY, c = 'lightgray', s = 100)
            cb = plt.scatter(lon, lat, c = X, cmap = cmap2, norm=matplotlib.colors.LogNorm(),
                        s = 4, vmin = 1e-2, vmax = 1e-1, )
            
            
            if i == 0: 
                plt.ylabel('%s' % (seasons[isea-1]), rotation = 90)
            if isea == 1:
                plt.title(labels[i])
            plt.text(140, -80, '(%s)' % plotidx[c])
            c +=1
    
    cax = fig.add_axes([0.355, 0.1, 0.275, 0.02])
    plt.colorbar(cb, cax, orientation = 'horizontal', label = 'AAOD', extend = 'both')
    plt.savefig(figdir + 'Climatology_AAOD_%s_residue.png' % (lo), dpi = 300, transparent = True)  
# #%%
# cmap = matplotlib.cm.cubehelix_r
# cmap2 = shiftedColorMap(cmap, start=0, midpoint = 0.75, stop=1, name='shifted')

# for lo in ['land', 'ocean']:
#     c = 0
#     fig = plt.figure(figsize = (8, 6.5))
#     for isea in range(1, 5):
#         if lo == 'land': 
#             temp = dataSeason[(dataSeason.season == isea) & \
#                               (dataSeason['AOD550(MODIS)'] > 0.1) & \
#                               (dataSeason['landoceanMask'] > 0.1)] 
#         if lo == 'ocean': 
#             temp = dataSeason[(dataSeason.season == isea) & \
#                               (dataSeason['AOD550(MODIS)'] > 0.1) & \
#                               (dataSeason['landoceanMask'] < 0.1)]          
#         lat = temp['lat_g']
#         lon = temp['lon_g']
        
#         for i, ipara in enumerate(['AOD550(MODIS)', 'AOD550', 'AOD']):
#             # plt.subplot(4, 3, c + 1)
#             ax = fig.add_axes([0.03 + i * 0.325, 0.77 - (isea - 1) * 0.2, 0.275, 0.17])
#             X = temp[ipara] 
#             bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
#                         lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
#             bm.drawcoastlines(color='gray',linewidth=1)
#             bm.drawparallels(np.arange(-45, 46, 45), labels=[False,True,False,False], linewidth = 0, fontsize = 8)
#             bm.drawmeridians(np.arange(-90, 91, 90), labels=[False,False,False,True], linewidth = 0, fontsize = 8)
#             XX, YY = np.meshgrid(np.arange(-180, 181, 10), np.arange(-90, 91, 10))
#             plt.scatter(XX, YY, c = 'lightgray', s = 100)
#             cb = plt.scatter(lon, lat, c = X, cmap = cmap2, norm=matplotlib.colors.LogNorm(),
#                         s = 4, vmin = 1e-1, vmax = 1, )
            
#             if i == 0: 
#                 plt.ylabel('%s' % (seasons[isea-1]), rotation = 90)
#             if isea == 1:
#                 plt.title(labels[i])
#             plt.text(140, -80, '(%s)' % plotidx[c])
#             c +=1
    
#     cax = fig.add_axes([0.355, 0.1, 0.275, 0.02])
#     plt.colorbar(cb, cax, orientation = 'horizontal', label = 'AOD', extend = 'both')
#     plt.savefig(figdir + 'Climatology_AOD_%s.png' % (lo), dpi = 300, transparent = True)

# #%%
# cmap = matplotlib.cm.cubehelix_r
# cmap2 = shiftedColorMap(cmap, start=0, midpoint = 0.75, stop=1, name='shifted')

# for lo in ['land', 'ocean']:
#     c = 0
#     fig = plt.figure(figsize = (8, 6.5))
#     for isea in range(1, 5):
#         if lo == 'land': 
#             temp = dataSeason[(dataSeason.season == isea) & \
#                               (dataSeason['AOD550(MODIS)'] > 0.1) & \
#                               (dataSeason['landoceanMask'] > 0.1)] 
#         if lo == 'ocean': 
#             temp = dataSeason[(dataSeason.season == isea) & \
#                               (dataSeason['AOD550(MODIS)'] > 0.1) & \
#                               (dataSeason['landoceanMask'] < 0.1)]          
#         lat = temp['lat_g']
#         lon = temp['lon_g']
        
#         for i, ipara in enumerate(['ALH', 'Haer_t1', 'Haer_63']):
#             # plt.subplot(4, 3, c + 1)
#             ax = fig.add_axes([0.03 + i * 0.325, 0.77 - (isea - 1) * 0.2, 0.275, 0.17])
#             X = temp[ipara] 
#             bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
#                         lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
#             bm.drawcoastlines(color='gray',linewidth=1)
#             bm.drawparallels(np.arange(-45, 46, 45), labels=[False,True,False,False], linewidth = 0, fontsize = 8)
#             bm.drawmeridians(np.arange(-90, 91, 90), labels=[False,False,False,True], linewidth = 0, fontsize = 8)
#             XX, YY = np.meshgrid(np.arange(-180, 181, 10), np.arange(-90, 91, 10))
#             plt.scatter(XX, YY, c = 'lightgray', s = 100)
#             cb = plt.scatter(lon, lat, c = X, cmap = cmap2, norm=matplotlib.colors.LogNorm(),
#                         s = 4, vmin = 1, vmax = 10, )
            
#             if i == 0: 
#                 plt.ylabel('%s' % (seasons[isea-1]), rotation = 90)
#             if isea == 1:
#                 plt.title(labels[i])
#             plt.text(140, -80, '(%s)' % plotidx[c])
#             c +=1
    
#     cax = fig.add_axes([0.355, 0.1, 0.275, 0.02])
#     plt.colorbar(cb, cax, orientation = 'horizontal', label = 'AOD', extend = 'both')
#     plt.savefig(figdir + 'Climatology_ALH_%s.png' % (lo), dpi = 300, transparent = True)

cmap = matplotlib.cm.cubehelix
cmap2 = shiftedColorMap(cmap, start=0, midpoint = 0.25, stop=1, name='shifted')
    
for lo in ['land', 'ocean']:
    c = 0
    fig = plt.figure(figsize = (8, 6.5))
    for isea in range(1, 5):
        if lo == 'land': 
            temp = dataSeason[(dataSeason.season == isea) & \
                              (dataSeason['AOD550(MODIS)'] > 0.1) & \
                              (dataSeason['landoceanMask'] > 0.1)] 
        if lo == 'ocean': 
            temp = dataSeason[(dataSeason.season == isea) & \
                              (dataSeason['AOD550(MODIS)'] > 0.1) & \
                              (dataSeason['landoceanMask'] < 0.1)]         
        lat = temp['lat_g']
        lon = temp['lon_g']
        
        for i, ipara in enumerate(['SSA_pred', 'SSA550', 'SSA']):
            # plt.subplot(4, 3, c + 1)
            ax = fig.add_axes([0.03 + i * 0.325, 0.77 - (isea - 1) * 0.2, 0.275, 0.17])
            X = temp[ipara] 
            bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
                        lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
            bm.drawcoastlines(color='gray',linewidth=1)
            bm.drawparallels(np.arange(-45, 46, 45), labels=[False,True,False,False], linewidth = 0, fontsize = 8)
            bm.drawmeridians(np.arange(-90, 91, 90), labels=[False,False,False,True], linewidth = 0, fontsize = 8)
            XX, YY = np.meshgrid(np.arange(-180, 181, 10), np.arange(-90, 91, 10))
            plt.scatter(XX, YY, c = 'lightgray', s = 100)
            cb = plt.scatter(lon, lat, c = X, cmap = cmap2, 
                        s = 4, vmin = 0.8, vmax = 1, )
            
            
            if i == 0: 
                plt.ylabel('%s' % (seasons[isea-1]), rotation = 90)
            if isea == 1:
                plt.title(labels[i])
            plt.text(140, -80, '(%s)' % plotidx[c])
            c +=1
    
    cax = fig.add_axes([0.355, 0.1, 0.275, 0.02])
    plt.colorbar(cb, cax, orientation = 'horizontal', label = 'SSA', extend = 'both')
    plt.savefig(figdir + 'Climatology_SSA_%s_residue.png' % (lo), dpi = 300, transparent = True)  

# #%%
# cmap = matplotlib.cm.cubehelix_r
# cmap2 = shiftedColorMap(cmap, start=0, midpoint = 0.5, stop=1, name='shifted')
    
# c = 0
# fig = plt.figure(figsize = (6, 4))
# for isea in range(1, 5):
#     temp = dataSeason[(dataSeason.season == isea) &\
#                       (dataSeason['AOD550(MODIS)'] >= 0.1)]
    
#     lat = temp['lat_g']
#     lon = temp['lon_g']
    
#     for i, ipara in enumerate(['AI388']):
#         # plt.subplot(4, 3, c + 1)
#         ax = fig.add_axes([0.05 + (c % 2) * 0.45, 0.55 - (c // 2) * 0.4, 0.375, 0.4])
#         X = temp[ipara]
#         bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
#                     lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
#         bm.drawcoastlines(color='gray',linewidth=1)
#         bm.drawparallels(np.arange(-45, 46, 45), labels=[False,True,False,False], linewidth = 0, fontsize = 8)
#         bm.drawmeridians(np.arange(-90, 91, 90), labels=[False,False,False,True], linewidth = 0, fontsize = 8)
#         XX, YY = np.meshgrid(np.arange(-180, 181, 10), np.arange(-90, 91, 10))
#         plt.scatter(XX, YY, c = 'lightgray', s = 100)
#         cb = plt.scatter(lon, lat, c = X, cmap = cmap2, 
#                     s = 4, vmin = 0, vmax = 2,)
        
#         plt.title('%s' % (seasons[isea-1]))
#         plt.text(140, -70, '(%s)' % plotidx[c])
#         c +=1

# cax = fig.add_axes([0.325, 0.125, 0.275, 0.02])
# plt.colorbar(cb, cax, orientation = 'horizontal', label = 'AI index', extend = 'both')
# plt.savefig(figdir + 'Climatology_UVAI.png', dpi = 300, transparent = True)  


# cmap = matplotlib.cm.cubehelix_r
# cmap2 = shiftedColorMap(cmap, start=0, midpoint = 0.5, stop=1, name='shifted')
    
# c = 0
# fig = plt.figure(figsize = (6, 4))
# for isea in range(1, 5):
#     temp = dataSeason[(dataSeason.season == isea) & (dataSeason['AOD550(MODIS)'] >= 0.1)]
    
#     lat = temp['lat_g']
#     lon = temp['lon_g']
    
#     for i, ipara in enumerate(['residue']):
#         # plt.subplot(4, 3, c + 1)
#         ax = fig.add_axes([0.05 + (c % 2) * 0.45, 0.55 - (c // 2) * 0.4, 0.375, 0.4])
#         X = temp[ipara] 
#         bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
#                     lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
#         bm.drawcoastlines(color='gray',linewidth=1)
#         bm.drawparallels(np.arange(-45, 46, 45), labels=[False,True,False,False], linewidth = 0, fontsize = 8)
#         bm.drawmeridians(np.arange(-90, 91, 90), labels=[False,False,False,True], linewidth = 0, fontsize = 8)
#         XX, YY = np.meshgrid(np.arange(-180, 181, 10), np.arange(-90, 91, 10))
#         plt.scatter(XX, YY, c = 'lightgray', s = 100)
#         cb = plt.scatter(lon, lat, c = X, cmap = cmap2, 
#                     s = 4, vmin = 0, vmax = 2, )
        
#         plt.title('%s' % (seasons[isea-1]))
#         plt.text(140, -70, '(%s)' % plotidx[c])
#         c +=1

# cax = fig.add_axes([0.325, 0.125, 0.275, 0.02])
# plt.colorbar(cb, cax, orientation = 'horizontal', label = 'Residue', extend = 'both')
# plt.savefig(figdir + 'Climatology_residue.png', dpi = 300, transparent = True)  

#%%
from matplotlib.patches import Polygon
colors = sns.color_palette("hls", 3)

ROIs = {
        'Global': {'E': 180, 'W': -180, 'N': 90, 'S': -90},
        'SE Asia': {'E': 120, 'W': 95, 'N': 25, 'S': 0},
        'S America': {'E': -40, 'W': -70, 'N': 5, 'S': -30},
        'C Africa': {'E': 10, 'W': -15, 'N': 15, 'S': 5},
        'S Africa': {'E': 37.5, 'W': 12.5, 'N': 0, 'S': -35},
        'Arabia': {'E': 60, 'W': 35, 'N': 35, 'S': 10},
        'N Africa': {'E': 30, 'W': -15, 'N': 35, 'S': 15}, 
        # 'Australia': {'E': 150, 'W': 115, 'N': -15, 'S': -37.5},
        # 'W America': {'E': -75, 'W': -125, 'N': 60, 'S': 30},
        # 'N Atlantic': {'E': -17.5, 'W': -45, 'N': 35, 'S': 5},
        # 'S Atlantic': {'E': 10, 'W': -25, 'N': 0, 'S': -35},
        }

plt.figure(figsize = (5, 3))
bm = Basemap(llcrnrlon=-180, llcrnrlat=-90, urcrnrlon=180, urcrnrlat=90, \
            lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
bm.drawcoastlines(color='gray',linewidth=1)
bm.drawparallels(np.arange(-45, 46, 45), labels=[False,True,False,False], linewidth = 0, fontsize = 8)
bm.drawmeridians(np.arange(-120, 121, 60), labels=[False,False,False,True], linewidth = 0, fontsize = 8)


for j, iROI in enumerate(ROIs.keys()): 
    ROI = ROIs[iROI]
    x1,y1 = bm(ROI['W'], ROI['S'])
    x2,y2 = bm(ROI['W'], ROI['N'])
    x3,y3 = bm(ROI['E'], ROI['N'])
    x4,y4 = bm(ROI['E'], ROI['S'])
    
    # if iROI == 'Global': 
    #     pass
    # else:
    if (j > 0) & (j<5):
        c = 'g'
    if j >= 5:
        c = 'brown'
    if j == 0: 
        c = 'k'
    print(j, c)
    
    poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor = 'none', edgecolor= c, alpha = 0.7, linewidth=2)
    plt.gca().add_patch(poly)
    plt.text((ROI['W'] + 5), (ROI['N'] - 15), plotidx[j], fontsize = 12)
    plt.text(-170, 45 - j * 15, '%s: %s' %(plotidx[j], iROI), fontsize = 8)
plt.savefig(figdir + 'Regions.png', dpi = 300, transparent = True)     
    
    
fig = plt.figure(figsize = (8, 9))
for j, iROI in enumerate(ROIs.keys()): 
    ROI = ROIs[iROI]
    x, y = dataMonth['lon_g'], dataMonth['lat_g']
    ROImask = (x <= ROI['E']) & (x >= ROI['W']) & (y <= ROI['N']) & (y >= ROI['S'])
    temp = dataMonth[(dataMonth['AOD550(MODIS)'] >= 0.1) & \
                     (dataMonth['AOD550'] >= 0.1) & \
                     (dataMonth['AOD'] >= 0.1) & \
                     (dataMonth['landoceanMask'] >= 0.1) & \
                         ROImask]
    
    mm = temp.groupby(['YYMM']).mean()
    count = temp.groupby(['YYMM']).count()['lat']
    mm[count < 1e2] = np.nan
    # std = temp.groupby(['YYMM']).std()
    X1 = mm['Y_pred']
    X2 = mm['AAOD550']
    X3 = mm['AAOD']
    
    if j == 0:
        ax = fig.add_axes([0.1, 0.85, 0.825, 0.1])
        
    if j > 0:
        j -= 1
        ax = fig.add_axes([0.1 + (j) % 2 * 0.45, 0.65 - (j ) // 2 * 0.2 , 0.37, 0.1])
        j += 1
        
    X1.plot(label = 'DNN-F11', marker = '', linewidth = 2, color = 'k')
    X2.plot(label = 'OMAERUV', marker = 'x', linewidth = 1, color = colors[1])
    X3.plot(label = 'MERRA-2', marker = '.', linewidth = 1, color = colors[2], markerfacecolor = 'none')
    plt.title('(%s) %s (R$^2_{O}$=%1.2f  R$^2_{M}$=%1.2f)' 
              % (plotidx[j], iROI, X1.corr(X2), X1.corr(X3)), fontsize = 10)
    # ax.yaxis.get_major_formatter().set_powerlimits((0,1))
    plt.ylabel('AAOD')
    plt.xticks(fontsize = 8)
    plt.yticks(fontsize = 8)
    
    if j == 0:
        plt.legend(ncol = 3)
    

    # print(iROI, X1.corr(X2), X1.corr(X3))
      
plt.savefig(figdir + 'Time_series_AAOD_residue.png', dpi = 300, transparent = True)     


fig = plt.figure(figsize = (8, 9))
for j, iROI in enumerate(ROIs.keys()): 
    ROI = ROIs[iROI]
    x, y = dataMonth['lon_g'], dataMonth['lat_g']
    ROImask = (x <= ROI['E']) & (x >= ROI['W']) & (y <= ROI['N']) & (y >= ROI['S'])
    temp = dataMonth[(dataMonth['AOD550(MODIS)'] >= 0.1) & \
                     (dataMonth['AOD550'] >= 0.1) & \
                     (dataMonth['AOD'] >= 0.1) & \
                     (dataMonth['landoceanMask'] >= 0.1) & \
                         ROImask]    
    mm = temp.groupby(['YYMM']).mean()
    count = temp.groupby(['YYMM']).count()['lat']
    mm[count < 1e2] = np.nan
    # std = temp.groupby(['YYMM']).std()
    X1 = mm['SSA_pred']
    X2 = mm['SSA550']
    X3 = mm['SSA']
    X4 = mm['AI388']

    if j == 0:
        ax = fig.add_axes([0.1, 0.85, 0.825, 0.1])
        
    if j > 0:
        j -= 1
        ax = fig.add_axes([0.1 + (j) % 2 * 0.45, 0.65 - (j ) // 2 * 0.2 , 0.37, 0.1])
        j += 1
        
    X1.plot(label = 'DNN-F11', marker = '', linewidth = 2, color = 'k')
    X2.plot(label = 'OMAERUV', marker = 'x', linewidth = 1, color = colors[1])
    X3.plot(label = 'MERRA-2', marker = '.', linewidth = 1, color = colors[2], markerfacecolor = 'none')
    # X4.plot(label = 'UVAI', marker = '+', linewidth = 1, color = colors[0])
    print(X4.corr(X1), X4.corr(X2), X4.corr(X3))
    plt.title('(%s) %s (R$^2_{O}$=%1.2f  R$^2_{M}$=%1.2f)' 
              % (plotidx[j], iROI, X1.corr(X2), X1.corr(X3)), fontsize = 10)
    # ax.yaxis.get_major_formatter().set_powerlimits((0,1))
    plt.ylabel('SSA')
    plt.xticks(fontsize = 8)
    plt.yticks(fontsize = 8)
    
    
    if j == 0:
        plt.legend(ncol = 3)
    

plt.savefig(figdir + 'Time_series_SSA_residue.png', dpi = 300, transparent = True)   


fig = plt.figure(figsize = (8, 9))
for j, iROI in enumerate(ROIs.keys()): 
    ROI = ROIs[iROI]
    x, y = dataMonth['lon_g'], dataMonth['lat_g']
    ROImask = (x <= ROI['E']) & (x >= ROI['W']) & (y <= ROI['N']) & (y >= ROI['S'])
    temp = dataMonth[(dataMonth['AOD550(MODIS)'] >= 0.1) & \
                     (dataMonth['AOD550'] >= 0.1) & \
                     (dataMonth['AOD'] >= 0.1) & \
                     (dataMonth['landoceanMask'] >= 0.1) & \
                         ROImask]   
    mm = temp.groupby(['YYMM']).mean()
    count = temp.groupby(['YYMM']).count()['lat']
    mm[count < 1e2] = np.nan
    # std = temp.groupby(['YYMM']).std()
    X1 = mm['AOD550(MODIS)']
    X2 = mm['AOD550']
    X3 = mm['AOD']
    X4 = mm['AI388']
    
    if j == 0:
        ax = fig.add_axes([0.1, 0.85, 0.825, 0.1])
        
    if j > 0:
        j -= 1
        ax = fig.add_axes([0.1 + (j) % 2 * 0.45, 0.65 - (j ) // 2 * 0.2 , 0.37, 0.1])
        j += 1
        
    X1.plot(label = 'DNN-F11', marker = '', linewidth = 2, color = 'k')
    X2.plot(label = 'OMAERUV', marker = 'x', linewidth = 1, color = colors[1])
    X3.plot(label = 'MERRA-2', marker = '.', linewidth = 1, color = colors[2], markerfacecolor = 'none')
    # X4.plot(label = 'UVAI', marker = '+', linewidth = 1, color = colors[0])
    plt.title('(%s) %s (R$^2_{O}$=%1.2f  R$^2_{M}$=%1.2f)' 
              % (plotidx[j], iROI, X1.corr(X2), X1.corr(X3)), fontsize = 10)
    ax.yaxis.get_major_formatter().set_powerlimits((0,1))
    plt.ylabel('AOD')
    plt.xticks(fontsize = 8)
    plt.yticks(fontsize = 8)
    

    
    if j == 0:
        plt.legend(ncol = 3)
    # print(iROI, X1.corr(X2), X1.corr(X3))
plt.savefig(figdir + 'Time_series_AOD_residue.png', dpi = 300, transparent = True)   

#%% global-zonal
def func(lat):
    return round(lat * .1) / .1

dataMonth['zonal'] = list(map(func, dataMonth.lat_g))
dataZonal = dataMonth.groupby(['zonal', 'season']).mean()     
dataZonal.reset_index(inplace = True)
dataZonal.index = dataZonal.zonal

for isea in range(1, 5):
    plt.figure(figsize = (12, 5))
    temp = dataZonal[dataZonal.season == isea]
    temp[['Y_pred', 'AAOD', 'AAOD550']].plot(kind = 'line')
    

    
 
#%%
import string
plotidx = string.ascii_lowercase
features = [
            ['residue', 'AOD550(MODIS)', 'Haer_t1',\
            'vza', 'raa', 'sza', 'As', 'Ps',\
            'lat', 'lon', 'doy'],
            
            # ['AI388', 'AOD550(MODIS)', 'Haer_t1',\
            # 'vza', 'raa', 'sza', 'As', 'Ps',\
            # 'lat', 'lon', 'doy'],  
            
            # ['AI388', 'AOD550(MODIS)', 'Haer_t1',\
            # 'vza', 'raa', 'sza', 'As', 'Ps']
                ]

expNames = [
            "%s_%i-layer_%i-neuron" % ('DNN_AAOD_train_residue', 2, 2**8),
            # "%s_%i-layer_%i-neuron" % ('DNN_AAOD_train_F11', 5, 2**8),
            # "%s_%i-layer_%i-neuron" % ('DNN_AAOD_train_F8', 4, 2**7)
            ]

titles = ['DNN-residue', 'DNN-F11', 'DNN-F8']
for i, expName in enumerate(expNames):
    # dataVal = pd.read_pickle(dataOutputDir + '%s_output/%s_output_validation.pickle' % (expName, expName))
    dataVal = pd.read_pickle(dataOutputDir + 'Validation_all_coincident_%s.pickle' % (expName))
    dataVal['AOD500(AERONET)'] = dataVal['Absorption_AOD[500nm]'] / (1 - dataVal['Single_Scattering_Albedo[500nm]'])
    dataVal['AOD550(AERONET)'] = dataVal['Absorption_AOD[550nm]'] / (1 - dataVal['Single_Scattering_Albedo[550nm]'])
    
    # OMAERUV SSA and AOD
    dataVal['diffSSA_OMAERUV'] = np.abs(dataVal['Single_Scattering_Albedo[500nm]'] - dataVal['SSA500']) <= 0.03
    dAOD = np.abs(dataVal['AOD500(AERONET)'] - dataVal['AOD500'])
    dataVal['diffAOD_OMAERUV'] = False
    dataVal['diffAOD_OMAERUV'][(dAOD <= 0.1) | (dAOD > 0.1) & (dAOD / dataVal['AOD500(AERONET)'] <= 0.3)] = True
    # MODIS AOD
    dataVal['diffAOD_MODIS'] = np.nan
    land = dataVal['landoceanMask'] >= 0.5
    dataVal.loc[dataVal['landoceanMask'] >= 0.5, 'diffAOD_MODIS'] = np.abs(dataVal['AOD550(AERONET)'][land] - dataVal['AOD550(MODIS)'][land]) <= (0.05 + 0.15 * dataVal['AOD550(AERONET)'][land])
    # dataVal.loc[dataVal['landoceanMask'] < 0.5, 'diffAOD_MODIS'] = np.abs(dataVal['AOD550(AERONET)'][~land] - dataVal['AOD550(MODIS)'][~land]) <= (0.03 + 0.05 * dataVal['AOD550(AERONET)'][~land])
    dataVal.loc[dataVal['landoceanMask'] < 0.5, 'diffAOD_MODIS'] = ((dataVal['AOD550(MODIS)'][~land] - dataVal['AOD550(AERONET)'][~land]) <= (0.04 + 0.1 * dataVal['AOD550(AERONET)'][~land])) & ((dataVal['AOD550(MODIS)'][~land] - dataVal['AOD550(AERONET)'][~land]) >= -(0.02 + 0.1 * dataVal['AOD550(AERONET)'][~land]))

    AAOD_err = list(map(errorPropagation, np.c_[dataVal['Single_Scattering_Albedo[550nm]'], dataVal['AOD550(AERONET)']]))
    dataVal['AAOD_err'] = np.array(AAOD_err)[:, -1]         
    
    dataVal['diffAAOD_MODIS'] = (abs(dataVal['Y_pred'] - dataVal['Absorption_AOD[550nm]']) <= dataVal.AAOD_err)
    dataVal['diffAAOD_OMAERUV'] = (abs(dataVal['AAOD500'] - dataVal['Absorption_AOD[500nm]']) <= dataVal.AAOD_err)
    dataVal['diffAAOD_MERRA2'] = (abs(dataVal['AAOD'] - dataVal['Absorption_AOD[550nm]']) <= dataVal.AAOD_err)
    
    # MERRA-2 AOD
    dataVal['diffAOD_MERRA2'] = np.nan
    land = dataVal['landoceanMask'] >= 0.5
    dataVal.loc[dataVal['landoceanMask'] >= 0.5, 'diffAOD_MERRA2'] = np.abs(dataVal['AOD550(AERONET)'][land] - dataVal['AOD'][land]) <= (0.05 + 0.15 * dataVal['AOD550(AERONET)'][land])
    dataVal.loc[dataVal['landoceanMask'] < 0.5, 'diffAOD_MERRA2'] = np.abs(dataVal['AOD550(AERONET)'][~land] - dataVal['AOD'][~land]) <= (0.03 + 0.05 * dataVal['AOD550(AERONET)'][~land])
    
    # OMAERUV at 550 nm
    EAE = Angstorm(388, dataVal['AOD388'], 500, dataVal['AOD500'])
    dataVal['AOD550'] = wvldepAOD(500, dataVal['AOD500'], 550, EAE)
    
    AAE = Angstorm(388, dataVal['AAOD388'], 500, dataVal['AAOD500'])
    dataVal['AAOD550'] = wvldepAOD(500, dataVal['AAOD500'], 550, AAE)
    
    dataVal['SSA550'] = 1 - dataVal['AAOD550'] / dataVal['AOD550']  
    
    dataVal['SSA_pred'] = 1- dataVal['Y_pred'] / dataVal['AOD550(MODIS)']
    dataVal['SSA'] = 1- dataVal['AAOD'] / dataVal['AOD']
    dataVal = dataVal[dataVal['SSA_pred'] >= 0]
     
    dataVal.reset_index(inplace = True)
    dataVal.loc[np.isnan(dataVal.AAOD550), 'AAOD550'] = np.nan
# =============================================================================
# mask invalid samples
# =============================================================================
    mask = (~np.isinf(dataVal['SSA_pred'])) & (dataVal['SSA_pred']<=1) & \
           (~np.isnan(dataVal['Absorption_AOD[550nm]'])) & \
           (~np.isnan(dataVal['AOD550(AERONET)'])) & \
           (~np.isinf(dataVal['SSA_pred'])) & (dataVal['AOD550(MODIS)'] >= dataVal['Y_pred']) & \
           (dataVal['As'] <= 0.3)
    
    dataVal = dataVal[mask]
# =============================================================================
# data without QC
# =============================================================================
#%%
    # validate AAOD
    fig = plt.figure(figsize = (6, 2.75))
    ax = fig.add_axes([0.1, 0.15, 0.325, 0.7])
    X1 = dataVal['Absorption_AOD[550nm]']
    X2 = dataVal['Y_pred']
    X3 = dataVal.AAOD_err
    slope, intercept, r_value, p_value, std_err = stats.linregress(X1, X2)
    perc = (abs(X1 - X2) <= X3).sum() / len(X1) * 1e2
    plt.hist2d(X1, X2, cmap = cmap1, bins = 50, norm=matplotlib.colors.LogNorm(), vmin = 10, vmax = 1e4)
    plt.ylabel(r'AAOD$^{pred}$')
    plt.xlabel(r'AAOD$^{A}$')
    plt.text(5e-2, 2e-1, r'k: %1.2f  b: %1.2f''\n''$R^2$: %1.2f''\n''RMSE: %1.2e''\n''MAE: %1.2e''\n''# num: %i''\n''P: %02i%%' \
              % (slope, intercept, np.corrcoef(X1, X2)[0, 1], 
                RMSE(X1, X2), 
                MAE(X1, X2), len(X1), perc))
    plt.plot(np.arange(0, 10), np.arange(0, 10) * slope + intercept, 'k-', linewidth = 1)
    plt.plot([0, 1], [0, 1], ':', color = 'gray', linewidth = 1)
    plt.plot([0, 0.5], [0, 1], '--', color = 'gray', linewidth = 1)
    plt.plot([0, 1], [0, 0.5], '--', color = 'gray', linewidth = 1)
    plt.xlim(0, 0.5)
    plt.ylim(0, 0.5)
    plt.text(X1.min(), 5e-2, '(a)'.rjust(40))
    ax.xaxis.get_major_formatter().set_powerlimits((0,1))
    ax.yaxis.get_major_formatter().set_powerlimits((0,1))
    
    # validate SSA
    ax = fig.add_axes([0.55, 0.15, 0.325, 0.7])
    X1 = dataVal['Single_Scattering_Albedo[550nm]'][mask]
    X2 = dataVal['SSA_pred'][mask]
    X3 = dataVal.AAOD_err[mask]
    slope, intercept, r_value, p_value, std_err = stats.linregress(X1, X2)
    perc = (abs(X1 - X2) <= X3).sum() / len(X1) * 1e2
    plt.hist2d(X1, X2, cmap = cmap1, bins = 50, norm=matplotlib.colors.LogNorm(), vmin = 10, vmax = 1e4)
    plt.ylabel(r'SSA$^{pred}$')
    plt.xlabel(r'SSA$^{A}$')
    plt.text(1e-1, 4e-1, r'k: %1.2f  b: %1.2f''\n''$R^2$: %1.2f''\n''RMSE: %1.2e''\n''MAE: %1.2e''\n''# num: %i''\n''P($\pm$0.03): %i%%' \
              % (slope, intercept, np.corrcoef(X1, X2)[0, 1], 
                RMSE(X1, X2), 
                MAE(X1, X2), len(X1), perc))
    plt.plot(np.arange(0, 10), np.arange(0, 10) * slope + intercept, 'k-', linewidth = 1)
    plt.plot([0.03, 1], [0, 0.97], '--', color = 'gray', linewidth = 1)
    plt.plot([0, 0.97], [0.03, 1], '--', color = 'gray', linewidth = 1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.text(0, 1e-1, '(b)'.rjust(40))
    ax.xaxis.get_major_formatter().set_powerlimits((0,1))
    ax.yaxis.get_major_formatter().set_powerlimits((0,1))

    cax =  fig.add_axes([0.915, 0.15, 0.02, 0.7])
    cb = plt.colorbar(cax = cax, fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10, \
                  label = '# num', extend = 'both', )
    plt.savefig(figdir + 'Validation_%s_residue.png' % (titles[i]), dpi = 300, transparent = True)        
        
 #%%       
    mask =  dataVal['diffAOD_MODIS'] & dataVal['diffAOD_OMAERUV'] & dataVal['diffSSA_OMAERUV'] &\
            dataVal['diffSSA_OMAERUV'] & dataVal['diffAOD_MERRA2'] 
    
    xlabels = ['UVAI', r'AOD$^M$', 'ALH [km]', r'VZA [${\circ}$]', r'RAA [${\circ}$]', r'SZA [${\circ}$]', r'a$_s$', r'P$_s$ [hPa]', r'Lat [${\circ}$]', r'Lon [${\circ}$]', 'DOY', 'x']
    # AAOD difference against features
    fig = plt.figure(figsize = (12, 8))
    for j, ipara in enumerate(features[i] + ['dAOD']):
        x, y = (j) % 4, (j) // 4
        ax = fig.add_axes([0.075 + 0.225 * x, 0.7 - 0.3 * y, 0.17, 0.225])
        
        if j == 11:
            X1 = (dataVal['AOD550(MODIS)'] - dataVal['AOD550(AERONET)'])
            plt.xlabel(r'AOD$^{M}$ - AOD$^{A}$')
        else:
            X1 = dataVal[ipara]
            plt.xlabel(xlabels[j])
        X2 = dataVal['Y_pred'] - dataVal['Absorption_AOD[550nm]']
        
        X1, X2 = X1[mask], X2[mask]
        
        X3 = pd.DataFrame(np.c_[X1, X2], columns = ['X1', 'X2'], index = X1.index)
        values = np.linspace(X1.min(), X1.max(), 10)

        X3 = X3.sort_values(['X1'])
        X3.reset_index(inplace = True)
        X3['group'] = 1 
        # for k, v in enumerate(values[1:]):
        #     print(values[k], v)
        #     X3.loc[(X1 >= values[k]) & (X1 < v), 'group'] = k + 1
        values = np.arange(0, len(X3) + len(X3) % 20, len(X3) // 20)
        for k, v in enumerate(values[1:]):
            X3.iloc[values[k]:v]['group'] = k + 1
            
        X3_gm = X3.groupby(['group']).mean()   
        X3_gs = X3.groupby(['group']).std()   
        
        
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(X1, X2)
        plt.hist2d(X1, X2, cmap = cmap1, bins = 50, norm=matplotlib.colors.LogNorm(), vmin = 10)
        
        plt.errorbar(X3_gm.X1, X3_gm.X2, yerr = X3_gs.X2, label = 'Grouped statistics', alpha = 0.75,\
                     color = 'r', ecolor = 'r', marker = 'o', elinewidth = 1, mfc = 'None', linewidth = 0)
        plt.hlines(0, -1e4, 1e4, color = 'k', linestyle = '--', linewidth = 1, alpha = 0.75, zorder = 100)
        if j == 11:
            plt.legend(loc = 2)
        if j % 4 == 0:
            plt.ylabel(r'AAOD$^{pred}$ - AAOD$^{A}$')
        
        plt.text(X1.min(), -1.5e-2, r'    k: %1.2f  b: %1.2f''\n''    $R^2$: %1.2f' \
                  % (slope, intercept, X1.corr(X2)))
        plt.text(X1.min(), -1.5e-2, '(%s)'.rjust(45)  % plotidx[j])
        ax.xaxis.get_major_formatter().set_powerlimits((0,1))
        # ax.yaxis.get_major_formatter().set_powerlimits((0,1))
        plt.ylim(-0.02, 0.02)
        
        
    cax =  fig.add_axes([0.925, 0.1, 0.015, 0.225])
    cb = plt.colorbar(cax = cax, fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10, \
                  label = '# num', extend = 'both')
    plt.savefig(figdir + 'dAAOD_against_features_%s.png' % (titles[i]), dpi = 300, transparent = True)        
#%%

    # SSA difference against features   
 
    fig = plt.figure(figsize = (12, 8))
    for j, ipara in enumerate(features[i] + ['dAOD']):
        x, y = (j) % 4, (j) // 4
        ax = fig.add_axes([0.075 + 0.225 * x, 0.7 - 0.3 * y, 0.17, 0.225])
        
        if j == 11:
            X1 = (dataVal['AOD550(MODIS)'] - dataVal['AOD550(AERONET)'])
            plt.xlabel(r'AOD$^{M}$ - AOD$^{A}$')
        else:
            X1 = dataVal[ipara]
            plt.xlabel(xlabels[j])
            
        X2 = dataVal['SSA_pred'] - dataVal['Single_Scattering_Albedo[550nm]']

        X1, X2 = X1[mask], X2[mask]

        X3 = pd.DataFrame(np.c_[X1, X2], columns = ['X1', 'X2'], index = X1.index)
        values = np.linspace(X1.min(), X1.max(), 10)
        X3 = X3.sort_values(['X1'])
        X3.reset_index(inplace = True)
        X3['group'] = 1 
        # for k, v in enumerate(values[1:]):
        #     print(values[k], v)
        #     X3.loc[(X1 >= values[k]) & (X1 < v), 'group'] = k + 1
        values = np.arange(0, len(X3) + len(X3) % 20, len(X3) // 20)
        for k, v in enumerate(values[1:]):
            X3.iloc[values[k]:v]['group'] = k + 1
            
        X3_gm = X3.groupby(['group']).mean()   
        X3_gs = X3.groupby(['group']).std()   
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(X1, X2)
        plt.hist2d(X1, X2, cmap = cmap1, bins = 50, norm=matplotlib.colors.LogNorm(), vmin = 10)
        # plt.ylabel(r'SSA$^{pred}$ - SSA$^{A}$')
        plt.errorbar(X3_gm.X1, X3_gm.X2, yerr = X3_gs.X2, label = 'Grouped statistics', alpha = 0.75, \
                     color = 'r', ecolor = 'r', marker = 'o', elinewidth = 1, mfc = 'None', linewidth = 0)
        plt.hlines(0, -1e4, 1e4, color = 'k', linestyle = '--', linewidth = 1, alpha = 0.75, zorder = 100)

        if j == 11:
            plt.legend(loc = 8)
        if j % 4 == 0:
            plt.ylabel(r'SSA$^{pred}$ - SSA$^{A}$')

        plt.text(X1.min(), 1.5e-1, r'    k: %1.2f  b: %1.2f''\n''    $R^2$: %1.2f' \
                  % (slope, intercept, X1.corr(X2)))
        plt.text(X1.min(), 1.5e-1, '(%s)'.rjust(45)  % plotidx[j])

        ax.xaxis.get_major_formatter().set_powerlimits((0,1))
        # ax.yaxis.get_major_formatter().set_powerlimits((0,1)
        plt.ylim(-0.3, 0.3)
            
    cax =  fig.add_axes([0.925, 0.1, 0.015, 0.225])
    cb = plt.colorbar(cax = cax, fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10, \
                  label = '# num', extend = 'both', )
    plt.savefig(figdir + 'dSSA_against_features_%s.png' % (titles[i]), dpi = 300, transparent = True)        
      

#%%
    # validate SSA
    fig = plt.figure(figsize = (9, 6))
    for i, v0 in enumerate([0, 0.01, 0.05, 0.1, 0.2, 0.5 ]):
        mask =  dataVal['diffAOD_MODIS'] & dataVal['diffAOD_OMAERUV']  &\
                dataVal['diffSSA_OMAERUV'] & dataVal['diffAOD_MERRA2'] &\
                (dataVal['AOD550(AERONET)'] >= v0)
        
        # mask = (dataVal['Haer_t1'] >= v0)
    
        ax = fig.add_axes([0.08 + (i % 3) * 0.3, 0.6 - (i // 3) * 0.5, 0.22, 0.325])
        X1 = dataVal['Single_Scattering_Albedo[550nm]'][mask]
        X2 = dataVal['SSA_pred'][mask]
        X3 = dataVal.AAOD_err[mask]
        slope, intercept, r_value, p_value, std_err = stats.linregress(X1, X2)
        perc = (abs(X1 - X2) <= 3e-2).sum() / len(X1) * 1e2
        plt.hist2d(X1, X2, cmap = cmap1, bins = 50, norm=matplotlib.colors.LogNorm(), vmin = 1, vmax = 1e3)
        plt.ylabel(r'SSA$^{pred}$')
        plt.xlabel(r'SSA$^{A}$')
        plt.text(np.floor(X2.min() * 10) / 10, (X2.max()-np.floor(X2.min() * 10) / 10)/2.5 + np.floor(X2.min() * 10) / 10, 
                 r'  k: %1.2f  b: %1.2f''\n''  $R^2$: %1.2f''\n''  RMSE: %1.2e''\n''  MAE: %1.2e''\n''  # num: %i''\n''  P($\pm$0.03): %02i%%' \
                  % (slope, intercept, np.corrcoef(X1, X2)[0, 1], 
                    RMSE(X1, X2), 
                    MAE(X1, X2), len(X1), perc))
        plt.plot(np.arange(0, 10), np.arange(0, 10) * slope + intercept, 'k-', linewidth = 1)
        plt.plot([0.03, 1], [0, 0.97], '--', color = 'gray', linewidth = 1)
        plt.plot([0, 0.97], [0.03, 1], '--', color = 'gray', linewidth = 1)
        plt.xlim(X2.min(), 1)
        plt.ylim(X2.min(), 1)
        plt.text(np.floor(X2.min() * 10) / 10, np.floor(X2.min() * 10) / 10, '(%s)''\n'.rjust(42) % plotidx[i])
        plt.xticks(np.linspace(np.floor(X2.min() * 10) / 10, 1, 5))
        plt.yticks(np.linspace(np.floor(X2.min() * 10) / 10, 1, 5))
        
        plt.title(r'AOD$^A \geq$%1.2f' % v0)
    cax =  fig.add_axes([0.925, 0.1, 0.02, 0.325])
    cb = plt.colorbar(cax = cax, fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10, \
                  label = '# num', extend = 'both', ticks = [1, 10, 1e2, 1e3] )
    plt.savefig(figdir + 'Validation_SSA_sensitivity_to_AOD_residue.png', dpi = 300, transparent = True) 


    fig = plt.figure(figsize = (9, 6))
    for i, v0 in enumerate([-1, 0, 0.1, 0.25, 0.5, 1]):
        mask =  dataVal['diffAOD_MODIS'] & dataVal['diffAOD_OMAERUV']  &\
                dataVal['diffSSA_OMAERUV'] & dataVal['diffAOD_MERRA2'] &\
                (dataVal['AOD550(AERONET)'] >= 0) & (dataVal['AI388'] >= v0)
        
        # mask = (dataVal['Haer_t1'] >= v0)
    
        ax = fig.add_axes([0.08 + (i % 3) * 0.3, 0.6 - (i // 3) * 0.5, 0.22, 0.325])
        X1 = dataVal['Single_Scattering_Albedo[550nm]'][mask]
        X2 = dataVal['SSA_pred'][mask]
        X3 = dataVal.AAOD_err[mask]
        slope, intercept, r_value, p_value, std_err = stats.linregress(X1, X2)
        perc = (abs(X1 - X2) <= 3e-2).sum() / len(X1) * 1e2
        plt.hist2d(X1, X2, cmap = cmap1, bins = 50, norm=matplotlib.colors.LogNorm(), vmin = 1, vmax = 1e3)
        plt.ylabel(r'SSA$^{pred}$')
        plt.xlabel(r'SSA$^{A}$')
        plt.text(np.floor(X2.min() * 10) / 10, (X2.max()-np.floor(X2.min() * 10) / 10)/2.5 + np.floor(X2.min() * 10) / 10, 
                 r'  k: %1.2f  b: %1.2f''\n''  $R^2$: %1.2f''\n''  RMSE: %1.2e''\n''  MAE: %1.2e''\n''  # num: %i''\n''  P($\pm$0.03): %02i%%' \
                  % (slope, intercept, np.corrcoef(X1, X2)[0, 1], 
                    RMSE(X1, X2), 
                    MAE(X1, X2), len(X1), perc))
        plt.plot(np.arange(0, 10), np.arange(0, 10) * slope + intercept, 'k-', linewidth = 1)
        plt.plot([0.03, 1], [0, 0.97], '--', color = 'gray', linewidth = 1)
        plt.plot([0, 0.97], [0.03, 1], '--', color = 'gray', linewidth = 1)
        plt.xlim(X2.min(), 1)
        plt.ylim(X2.min(), 1)
        plt.text(np.floor(X2.min() * 10) / 10, np.floor(X2.min() * 10) / 10, '(%s)''\n'.rjust(42) % plotidx[i])
        plt.xticks(np.linspace(np.floor(X2.min() * 10) / 10, 1, 5))
        plt.yticks(np.linspace(np.floor(X2.min() * 10) / 10, 1, 5))
        
        plt.title(r'UVAI $\geq$%1.2f' % v0)
    cax =  fig.add_axes([0.925, 0.1, 0.02, 0.325])
    cb = plt.colorbar(cax = cax, fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10, \
                  label = '# num', extend = 'both', ticks = [1, 10, 1e2, 1e3] )
    plt.savefig(figdir + 'Validation_SSA_sensitivity_to_UVAI_residue.png', dpi = 300, transparent = True)   
#%%
    # validation screened
    paras = {'MODIS': 'Y_pred',
         'OMAERUV': 'AAOD500',
         'MERRA2': 'AAOD'}
    
    titles = ['DNN-F11', 'OMAERUV', 'MERRA-2']
    labels = ['pred', 'O', 'M']
    fig = plt.figure(figsize = (9, 2.75))
    for i, idata in enumerate(list(paras.keys())):
        ipara = paras[idata]
        
        # mask = dataVal['diffAOD_%s' % idata] & dataVal['diffSSA_OMAERUV'] & dataVal['diffAOD_OMAERUV']
        mask = dataVal['diffAOD_MODIS'] & dataVal['diffAOD_OMAERUV']  &\
        dataVal['diffSSA_OMAERUV'] & dataVal['diffAOD_MERRA2'] &\
            (dataVal['AOD550(AERONET)'] >= 0.1) #& (dataVal['AOD500'] >= 0.05) & (dataVal['AOD'] >= 0.05)
    
        
        X1, X2 = dataVal['Absorption_AOD[550nm]'][mask], dataVal[ipara][mask]
        if idata == 'OMAERUV':
            X1 = dataVal['Absorption_AOD[500nm]'][mask]
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(X1, X2)
        perc = (abs(X1 - X2) <= dataVal[mask].AAOD_err).sum() / len(X1) * 1e2
        ax = fig.add_axes([0.075 + i * 0.3, 0.15, 0.22, 0.7])
        plt.hist2d(X1, X2, bins = 50,
                   cmap = cmap1, norm = matplotlib.colors.LogNorm(), vmin = 1, vmax = 1e3)
        plt.plot(np.arange(0, 10), np.arange(0, 10) * slope + intercept, 'k-', linewidth = 1)
        plt.plot([0, 1], [0, 1], ':', color = 'gray', linewidth = 1)
        plt.plot([0, 0.5], [0, 1], '--', color = 'gray', linewidth = 1)
        plt.plot([0, 1], [0, 0.5], '--', color = 'gray', linewidth = 1)
        plt.xlim(0, 0.3)
        plt.ylim(0, 0.3)
        plt.text(X1.min(), 1.25e-1, r'  k: %1.2f  b: %1.2f''\n''  $R^2$: %1.2f''\n''  RMSE: %1.2e''\n''  MAE: %1.2e''\n''  P: %02i%%''\n''  # num: %i' \
                 % (slope, intercept, np.corrcoef(X1, X2)[0, 1], 
                    RMSE(X1, X2), 
                    MAE(X1, X2), perc, len(X1)))
        plt.text(0.025, 0.1e-1, '(%s)'.rjust(40) % plotidx[i])
        plt.ylabel(r'AAOD$^{%s}$' % (labels[i]))
        plt.xlabel(r'AAOD$^{A}$')
        plt.title(titles[i])
        # ax.xaxis.get_major_formatter().set_powerlimits((0,1))
        # ax.yaxis.get_major_formatter().set_powerlimits((0,1))
    cax =  fig.add_axes([0.915, 0.15, 0.02, 0.7])
    cb = plt.colorbar(cax = cax, fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10, \
                  label = '# num', extend = 'both', )
    plt.savefig(figdir + 'Validation_screened_residue.png', dpi = 300, transparent = True)        


    paras = {'MODIS': 'SSA_pred',
         'OMAERUV': 'SSA500',
         'MERRA2': 'SSA'}
    titles = ['DNN-F11', 'OMAERUV', 'MERRA-2']
    labels = ['pred', 'O', 'M']
    fig = plt.figure(figsize = (9, 2.75))
    for i, idata in enumerate(list(paras.keys())):
        ipara = paras[idata]
        
        mask = dataVal['diffAOD_MODIS'] & dataVal['diffAOD_OMAERUV'] &\
                dataVal['diffSSA_OMAERUV'] & dataVal['diffAOD_MERRA2'] &\
                (dataVal['AOD550(AERONET)'] >= 0.1) 
        X1, X2 = dataVal['Single_Scattering_Albedo[550nm]'][mask], dataVal[ipara][mask]
        if idata == 'OMAERUV':
            X1 = dataVal['Single_Scattering_Albedo[500nm]'][mask]
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(X1, X2)
        perc = (abs(X1 - X2) <= 3e-2).sum() / len(X1) * 1e2
        ax = fig.add_axes([0.075 + i * 0.3, 0.15, 0.22, 0.7])
        plt.hist2d(X1, X2, bins = 50,
                   cmap = cmap1, norm = matplotlib.colors.LogNorm(), vmin = 1, vmax = 1e2)
        plt.plot(np.arange(0, 10), np.arange(0, 10) * slope + intercept, 'k-', linewidth = 1)
        plt.plot([0.03, 1], [0, 0.97], '--', color = 'gray', linewidth = 1)
        plt.plot([0, 0.97], [0.03, 1], '--', color = 'gray', linewidth = 1)
        plt.xlim(0.7, 1)
        plt.ylim(0.7, 1)
        plt.text(0.7, 0.825, r'  k: %1.2f  b: %1.2f''\n''  $R^2$: %1.2f''\n''  RMSE: %1.2e''\n''  MAE: %1.2e''\n''  P($\pm$0.03): %02i%%''\n''  # num: %i' \
                 % (slope, intercept, np.corrcoef(X1, X2)[0, 1], 
                    RMSE(X1, X2), 
                    MAE(X1, X2), perc, len(X1)))
        plt.text(0.95, 0.725, '(%s)'.rjust(0) % plotidx[i])
        plt.ylabel(r'SSA$^{%s}$' % (labels[i]))
        plt.xlabel(r'SSA$^{A}$')
        plt.title(titles[i])
        ax.xaxis.get_major_formatter().set_powerlimits((0,1))
        ax.yaxis.get_major_formatter().set_powerlimits((0,1))
    cax =  fig.add_axes([0.915, 0.15, 0.02, 0.7])
    cb = plt.colorbar(cax = cax, fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10, \
                  label = '# num', extend = 'both', )
    plt.savefig(figdir + 'Validation_screened_SSA_residue.png', dpi = 300, transparent = True)       


    # paras = {'MODIS': 'AOD550(MODIS)',
    #      'OMAERUV': 'AOD500',
    #      'MERRA2': 'AOD'}
    # titles = ['DNN-F11', 'OMAERUV', 'MERRA-2']
    # labels = ['pred', 'O', 'M']
    # fig = plt.figure(figsize = (9, 2.75))
    # for i, idata in enumerate(list(paras.keys())):
    #     ipara = paras[idata]
        
    #     mask = dataVal['diffAOD_MODIS'] & dataVal['diffAOD_OMAERUV'] &\
    #             dataVal['diffSSA_OMAERUV'] & dataVal['diffAOD_MERRA2'] &\
    #             (dataVal['AOD550(AERONET)'] >= 0.1) 
    #     X1, X2 = dataVal['AOD550(AERONET)'][mask], dataVal[ipara][mask]
    #     if idata == 'OMAERUV':
    #         X1 = dataVal['AOD550(AERONET)'][mask]
        
    #     slope, intercept, r_value, p_value, std_err = stats.linregress(X1, X2)
    #     perc = (abs(X1 - X2) <= 3e-2).sum() / len(X1) * 1e2
    #     ax = fig.add_axes([0.05 + i * 0.3, 0.15, 0.22, 0.7])
    #     plt.hist2d(X1, X2, bins = 50,
    #                cmap = cmap1, norm = matplotlib.colors.LogNorm(), vmin = 1, vmax = 1e2)
    #     plt.plot(np.arange(0, 10), np.arange(0, 10) * slope + intercept, 'k-', linewidth = 1)
    #     plt.plot([0, 10], [0, 10], ':', color = 'gray', linewidth = 1)
    #     plt.plot([0, 5], [0, 10], '--', color = 'gray', linewidth = 1)
    #     plt.plot([0, 10], [0, 5], '--', color = 'gray', linewidth = 1)
    #     plt.xlim(0, 2)
    #     plt.ylim(0, 2)
    #     plt.text(0.1, 0.825, r'  k: %1.2f  b: %1.2f''\n''  $R^2$: %1.2f''\n''  RMSE: %1.2e''\n''  MAE: %1.2e''\n''  P($\pm$0.03): %02i%%''\n''  # num: %i' \
    #              % (slope, intercept, np.corrcoef(X1, X2)[0, 1], 
    #                 RMSE(X1, X2), 
    #                 MAE(X1, X2), perc, len(X1)))
    #     plt.text(1.75, 0.15, '(%s)'.rjust(0) % plotidx[i])
    #     plt.ylabel(r'AOD$^{%s}$' % (labels[i]))
    #     plt.xlabel(r'AOD$^{A}$')
    #     plt.title(titles[i])
    #     ax.xaxis.get_major_formatter().set_powerlimits((0,1))
    #     ax.yaxis.get_major_formatter().set_powerlimits((0,1))
    # cax =  fig.add_axes([0.915, 0.15, 0.02, 0.7])
    # cb = plt.colorbar(cax = cax, fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10, \
    #               label = '# num', extend = 'both', )
    # plt.savefig(figdir + 'Validation_screened_AOD.png', dpi = 300, transparent = True)             
        
          #%%  
    #  global map AAOD
    ROI = {'S': -90, 'N': 90, 'W': -180, 'E': 180}
    paras = {'MODIS': 'Y_pred',
         'OMAERUV': 'AAOD500',
         'MERRA2': 'AAOD'}
    cmap = matplotlib.cm.Spectral_r
    cmap3 = shiftedColorMap(cmap, start=0, midpoint = 0.5, stop=1, name='shifted')
    
    fig = plt.figure(figsize = (10, 4))
    for i, idata in enumerate(list(paras.keys())):
        ipara = paras[idata]
        mask = dataVal['diffAOD_MODIS'] & dataVal['diffAOD_OMAERUV'] &\
                dataVal['diffSSA_OMAERUV'] & dataVal['diffAOD_MERRA2'] &\
                (dataVal['AOD550(AERONET)'] >= 0.1) 
    
        site = dataVal[mask].groupby('Site').mean()
        
        X1, X2 = site['Absorption_AOD[550nm]'], site[ipara]
        if idata == 'OMAERUV':
            X1 = site['Absorption_AOD[500nm]']
            
        ax = fig.add_axes([0.05 + i * 0.3, 0.5, 0.25, 0.4])
        bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
                    lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
        bm.drawcoastlines(color='gray',linewidth=1)
        plt.scatter(site['Longitude(Degrees)'], site['Latitude(Degrees)'], c = X2, cmap = cmap3,
                    s = 25, vmin = 5e-3, vmax = 5e-2, edgecolor = 'k', 
                    norm=matplotlib.colors.LogNorm(),
                    linewidth = 0.5, zorder = 10, alpha = 0.5)
        plt.text(-170, -80, '(%s)' % plotidx[i])
        plt.title(titles[i])
        bm.drawparallels(np.arange(-45, 46, 45), labels=[True,False,False,False], linewidth = 0, fontsize = 8)
        bm.drawmeridians(np.arange(-120, 121, 60), labels=[False,False,False,True], linewidth = 0, fontsize = 8)
        
        if i == 2: 
            cax1 = fig.add_axes([0.92, 0.5, 0.01, 0.4])
            cb = plt.colorbar(cax = cax1, fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10, \
                      label = 'AAOD', extend = 'both', ticks = [5e-3, 1e-2, 5e-2])
        
    
        ax = fig.add_axes([0.05 + i * 0.3 , 0.05, 0.25, 0.4])
        bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
                    lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
        bm.drawcoastlines(color='gray',linewidth=1)
        plt.scatter(site['Longitude(Degrees)'],  site['Latitude(Degrees)'], c= X2 - X1, cmap = 'bwr', 
                    s = 25, vmin = -5e-3, vmax = 5e-3, edgecolor = 'k', 
                    linewidth = 0.5, zorder = 10, alpha = 0.5)
        plt.text(-170, -80, '(%s)' % plotidx[i + 3])
        bm.drawparallels(np.arange(-45, 46, 45), labels=[True,False,False,False], linewidth = 0, fontsize = 8)
        bm.drawmeridians(np.arange(-120, 121, 60), labels=[False,False,False,True], linewidth = 0, fontsize = 8)
        if i == 2: 
            cax2 = fig.add_axes([0.92, 0.05, 0.01, 0.4])
            cb = plt.colorbar(cax = cax2, fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10, \
                      label = 'AAOD difference', extend = 'both', ticks = [-5e-3, 0, 5e-3])
            cb.formatter.set_powerlimits((0, 1))

        plt.savefig(figdir + 'Spatial_vld.png', dpi = 300, transparent = True)
        print(RMSE(X1, X2), X1.corr(X2), (abs(X1 - X2) <= site.AAOD_err).sum() / len(X1) * 1e2)
        print((X2 - X1).mean())
          #%%  
    #  global map AOD
    paras = {'MODIS': 'AOD550(MODIS)',
         'OMAERUV': 'AOD500',
         'MERRA2': 'AOD'}
    cmap = matplotlib.cm.Spectral_r
    cmap3 = shiftedColorMap(cmap, start=0, midpoint = 0.4, stop=1, name='shifted')
    
    fig = plt.figure(figsize = (10, 4))
    for i, idata in enumerate(list(paras.keys())):
        ipara = paras[idata]
        mask = dataVal['diffAOD_MODIS'] & dataVal['diffAOD_OMAERUV'] &\
                dataVal['diffSSA_OMAERUV'] & dataVal['diffAOD_MERRA2'] &\
                (dataVal['AOD550(AERONET)'] >= 0.1) 
    
        site = dataVal[mask].groupby('Site').mean()
        
        X1, X2 = site['AOD550(AERONET)'], site[ipara]
        if idata == 'OMAERUV':
            X1 = site['AOD500(AERONET)']
            
        ax = fig.add_axes([0.05 + i * 0.3, 0.5, 0.25, 0.4])
        bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
                    lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
        bm.drawcoastlines(color='gray',linewidth=1)
        plt.scatter(site['Longitude(Degrees)'], site['Latitude(Degrees)'], c = X2, cmap = cmap3,
                    s = 25, vmin = 1e-1, vmax = 1, edgecolor = 'k', 
                    norm=matplotlib.colors.LogNorm(),
                    linewidth = 0.5, zorder = 10, alpha = 0.5)
        plt.text(-170, -80, '(%s)' % plotidx[i])
        plt.title(titles[i])
        bm.drawparallels(np.arange(-45, 46, 45), labels=[True,False,False,False], linewidth = 0, fontsize = 8)
        bm.drawmeridians(np.arange(-120, 121, 60), labels=[False,False,False,True], linewidth = 0, fontsize = 8)
        
        if i == 2: 
            cax1 = fig.add_axes([0.92, 0.5, 0.01, 0.4])
            cb = plt.colorbar(cax = cax1, fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10, \
                      label = 'AOD', extend = 'both', ticks = [1e-1, 5e-1, 1])
        
    
        ax = fig.add_axes([0.05 + i * 0.3 , 0.05, 0.25, 0.4])
        bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
                    lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
        bm.drawcoastlines(color='gray',linewidth=1)
        plt.scatter(site['Longitude(Degrees)'],  site['Latitude(Degrees)'], c= X2 - X1, cmap = 'bwr', 
                    s = 25, vmin = -5e-2, vmax = 5e-2, edgecolor = 'k', 
                    linewidth = 0.5, zorder = 10, alpha = 0.5)
        plt.text(-170, -80, '(%s)' % plotidx[i + 3])
        bm.drawparallels(np.arange(-45, 46, 45), labels=[True,False,False,False], linewidth = 0, fontsize = 8)
        bm.drawmeridians(np.arange(-120, 121, 60), labels=[False,False,False,True], linewidth = 0, fontsize = 8)
        if i == 2: 
            cax2 = fig.add_axes([0.92, 0.05, 0.01, 0.4])
            cb = plt.colorbar(cax = cax2, fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10, \
                      label = 'AOD difference', extend = 'both', ticks = [-5e-2, 0, 5e-2])
            cb.formatter.set_powerlimits((0, 1))

        plt.savefig(figdir + 'Spatial_vld_AOD_residue.png', dpi = 300, transparent = True)
        print(RMSE(X1, X2), X1.corr(X2), (abs(X1 - X2) <= site.AAOD_err).sum() / len(X1) * 1e2)

#%%
    fig = plt.figure(figsize = (4, 2 ))
    
    ipara = paras[idata]
    mask = dataVal['diffAOD_MODIS'] & dataVal['diffAOD_OMAERUV'] &\
            dataVal['diffSSA_OMAERUV'] & dataVal['diffAOD_MERRA2'] &\
            (dataVal['AOD550(AERONET)'] >= 0.1) 

    site = dataVal[mask].groupby('Site').mean()
    
    X1 = site['AI388']
        
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.7])
    bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
                lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
    bm.drawcoastlines(color='gray',linewidth=1)
    plt.scatter(site['Longitude(Degrees)'], site['Latitude(Degrees)'], c = X1, cmap = cmap3,
                s = 25, vmin = -0, vmax = 1, edgecolor = 'k', 
                linewidth = 0.5, zorder = 10, alpha = 0.5)
    plt.title('OMAERUV UVAI')
    bm.drawparallels(np.arange(-45, 46, 45), labels=[True,False,False,False], linewidth = 0, fontsize = 8)
    bm.drawmeridians(np.arange(-120, 121, 60), labels=[False,False,False,True], linewidth = 0, fontsize = 8)

    cax1 = fig.add_axes([0.85 , 0.1, 0.02, 0.7])
    cb = plt.colorbar(cax = cax1, fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10, \
              label = 'UVAI', extend = 'both', ticks = [-0.5, 0, 0.5, 1])
    plt.savefig(figdir + 'Spatial_vld_UVAI_residue.png', dpi = 300, transparent = True)
    
    
    #%%
    # global map SSA
    cmap = matplotlib.cm.Spectral
    cmap4 = shiftedColorMap(cmap, start=0, midpoint = 0.5, stop=1, name='shifted')

    paras = {'MODIS': 'SSA_pred',
             'OMAERUV': 'SSA500',
             'MERRA2': 'SSA'}
    
    fig = plt.figure(figsize = (10, 4))
    for i, idata in enumerate(list(paras.keys())):
        ipara = paras[idata]
        mask = dataVal['diffAOD_MODIS'] & dataVal['diffAOD_OMAERUV'] &\
                dataVal['diffSSA_OMAERUV'] & dataVal['diffAOD_MERRA2'] &\
                (dataVal['AOD550(AERONET)'] >= 0.1) 

    
        site = dataVal[mask].groupby('Site').mean()
        
        X1, X2 = site['Single_Scattering_Albedo[550nm]'], site[ipara]
        if idata == 'OMAERUV':
            X1 = site['Single_Scattering_Albedo[500nm]']
            
        ax = fig.add_axes([0.05 + i * 0.3, 0.5, 0.25, 0.4])
        bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
                    lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
        bm.drawcoastlines(color='gray',linewidth=1)
        plt.scatter(site['Longitude(Degrees)'], site['Latitude(Degrees)'], c = X2, cmap = cmap4,
                    s = 25, vmin = 0.85, vmax = 1, edgecolor = 'k', 
                    linewidth = 0.5, zorder = 10, alpha = 0.5)
        plt.text(-170, -80, '(%s)' % plotidx[i])
        plt.title(titles[i])
        bm.drawparallels(np.arange(-45, 46, 45), labels=[True,False,False,False], linewidth = 0, fontsize = 8)
        bm.drawmeridians(np.arange(-120, 121, 60), labels=[False,False,False,True], linewidth = 0, fontsize = 8)
        
        if i == 2: 
            cax1 = fig.add_axes([0.92, 0.5, 0.01, 0.4])
            cb = plt.colorbar(cax = cax1, fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10, \
                      label = 'SSA', extend = 'both', ticks = [0.85, 0.9, 0.95, 1])
        
        
        ax = fig.add_axes([0.05 + i * 0.3, 0.05, 0.25, 0.4])
        bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
                    lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
        bm.drawcoastlines(color='gray',linewidth=1)
        plt.scatter(site['Longitude(Degrees)'],  site['Latitude(Degrees)'], c= X2 - X1, cmap = 'bwr', 
                    s = 25, vmin = -3e-2, vmax = 3e-2, edgecolor = 'k', 
                    linewidth = 0.5, zorder = 10, alpha = 0.5)
        plt.text(-170, -80, '(%s)' % plotidx[i + 3])
        bm.drawparallels(np.arange(-45, 46, 45), labels=[True,False,False,False], linewidth = 0, fontsize = 8)
        bm.drawmeridians(np.arange(-120, 121, 60), labels=[False,False,False,True], linewidth = 0, fontsize = 8)
        if i == 2: 
            cax2 = fig.add_axes([0.92, 0.05, 0.01, 0.4])
            cb = plt.colorbar(cax = cax2, fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10, \
                      label = 'SSA difference', extend = 'both', ticks = [-3e-2, 0, 3e-2])
            cb.formatter.set_powerlimits((0, 1))

        plt.savefig(figdir + 'Spatial_vld_SSA_residue.png', dpi = 300, transparent = True)
        print('+/-0.03 %02i %%' % ((abs(X1 - X2)<= 0.03).sum() / len(X1) * 1e2))
# #%%
#         site = dataVal[mask].groupby('Site').mean()
#         X1, X2 = dataVal[mask]['Absorption_AOD[550nm]'], dataVal[mask]['Y_pred']
#         X3, X4 = dataVal[mask]['Single_Scattering_Albedo[550nm]'], dataVal[mask]['SSA_pred']
#         X5, X6 = dataVal[mask]['AOD550(AERONET)'], dataVal[mask]['AOD550(MODIS)']
        
#         plt.figure()
#         plt.scatter(X6 - X5, X4 - X3, c = X2 - X1, cmap = 'bwr', s = X6*10, alpha = 0.5, vmin = -5e-3, vmax = 5e-3)
#         plt.xlabel('dAAOD')
#         plt.ylabel('dSSA')
#         plt.colorbar()           
        
            
#%%
    paras = {'MODIS': 'Y_pred',
         'OMAERUV': 'AAOD500',
         'MERRA2': 'AAOD'}

    dataVal['dateTime'] = pd.to_datetime(dataVal['timeStamp'] * 1e9)
    dataVal['YYMM'] = dataVal['dateTime'].dt.to_period('M')
    dataVal['season'] = (dataVal.YYMM.dt.month % 12 + 3) // 3
    
    
    ROIs = {'Global': {'E': 180, 'W': -180, 'N': 90, 'S': -90},
            'SE Asia': {'E': 120, 'W': 60, 'N': 30, 'S': -20},
            'N Africa': {'E': 30, 'W': -20, 'N': 40, 'S': 0},
            'S Africa': {'E': 30, 'W': -10, 'N': -10, 'S': -40},
            'N America': {'E': -60, 'W': -130, 'N': 60, 'S': 30},
            'S America': {'E': -40, 'W': -80, 'N': 10, 'S': -60}}
    
    stats_roi = pd.DataFrame()
    fig = plt.figure(figsize = (12, 8))
    for j, iROI in enumerate(ROIs.keys()): 
        ROI = ROIs[iROI]
        x, y = dataVal['Longitude(Degrees)'], dataVal['Latitude(Degrees)']
        ROImask = (x <= ROI['E']) & (x >= ROI['W']) & (y <= ROI['N']) & (y >= ROI['S'])
    
        colors = sns.color_palette("hls", 3)
        labels = ['DNN-F11', 'OMAERUV', 'MERRA-2']
        
        
        ax = fig.add_axes([0.1 + (j) % 2 * 0.4, 0.7 - (j ) // 2 * 0.3, 0.325, 0.2])
        
        temp = pd.DataFrame(index = labels, columns = ['RMSE', 'MAE', 'R^2', 'P', 'ROI', 'Data'])
        for i, idata in enumerate(list(paras.keys())):
            ipara = paras[idata]
            mask = ROImask & dataVal['diffAOD_MODIS'] & dataVal['diffAOD_OMAERUV'] &\
                    dataVal['diffAOD_MERRA2'] & dataVal['diffSSA_OMAERUV'] &\
                    (dataVal['AOD550(AERONET)'] >= 0.1)
        
            mm = dataVal[mask].groupby('YYMM').mean()
            mstd = dataVal[mask].groupby('YYMM').std()
            
            X1, X2 = mm['Absorption_AOD[550nm]'], mm[ipara]
            X1_std = mstd['Absorption_AOD[550nm]']
            if idata == 'OMAERUV':
                X1 = mm['Absorption_AOD[500nm]']
                X1_2 = mm['Absorption_AOD[500nm]']
                X1_2_std = mstd['Absorption_AOD[500nm]']
                
            
            X2.plot(color = colors[i], label = labels[i])
            
            temp.loc[labels[i]] = RMSE(X1, X2), MAE(X1, X2), X1.corr(X2), (abs(X1 - X2) <= mm.AAOD_err).sum() / len(X1) * 1e2, iROI, labels[i]
            print(RMSE(X1, X2), X1.corr(X2), (abs(X1 - X2) <= mm.AAOD_err).sum() / len(X1) * 1e2)

        stats_roi = stats_roi.append(temp)
        X1.plot(color = 'k', linestyle = '--', linewidth = 1, label = 'AERONET(550)')
        X1_2.plot(color = 'gray', linestyle = '--', marker = '', linewidth = 1, label = 'AERONET(500)')
        plt.title('(%s) %s time series' % (plotidx[j], iROI))
        if j == 5:
            plt.legend(ncol = 2)
        plt.ylabel('AAOD')
        # plt.ylim(0.5e-2, 3e-2)
        ax.yaxis.get_major_formatter().set_powerlimits((0,1))
    plt.savefig(figdir + 'Temporal_vld_residue.png', dpi = 300, transparent = True)  
    
    
    
    stats_roi = stats_roi.groupby(['ROI', 'Data']).min()
    # save 

    for ipara in stats_roi.columns:
        if ipara in ['R^2']: 
            stats_roi[ipara] = stats_roi[ipara].map(lambda x: '%1.2f' % x)
        if ipara in ['RMSE', 'MAE']:
            stats_roi[ipara] = stats_roi[ipara].map(lambda x: '%1.2e' % x)
        if ipara in ['P']:
            stats_roi[ipara] = stats_roi[ipara].map(lambda x: '%i' % x)
            
    stats_roi.to_csv('Regional_stats.csv')

    #%% select site

ROIs = {
        'SE Asia': {'E': 120, 'W': 95, 'N': 25, 'S': -0},
        # 'E Asia': {'E': 130, 'W': 100, 'N': 60, 'S': 25},
        # 'N Africa': {'E': 30, 'W': -20, 'N': 35, 'S': 0},
        # 'Saudi': {'E': 60, 'W': 30, 'N': 40, 'S': 20},
        # 'S Africa': {'E': 40, 'W': -10, 'N': -10, 'S': -40},
        # 'W America': {'E': -100, 'W': -130, 'N': 60, 'S': 30},
        # 'E America': {'E': -60, 'W': -80, 'N': 45, 'S': 30},
        # 'S America': {'E': -30, 'W': -80, 'N': 5, 'S': -30},
        # 'Austrialia': {'E': 150, 'W': 100, 'N': -10, 'S': -45}
        }
for iROI in ROIs:
    ROI = ROIs[iROI]
    x, y = dataVal['Longitude(Degrees)'], dataVal['Latitude(Degrees)']
    ROImask = (x <= ROI['E']) & (x >= ROI['W']) & (y <= ROI['N']) & (y >= ROI['S'])
    
    mask = ROImask & dataVal['diffAOD_MODIS'] & dataVal['diffAOD_OMAERUV'] &\
                    dataVal['diffAOD_MERRA2'] & dataVal['diffSSA_OMAERUV'] &\
                    (dataVal['AOD550(AERONET)'] >= 0.1)
                    
    counts = dataVal[mask].groupby(['Site']).count()['lat']
    sites = list(counts.nlargest(20).index)
    
    dataValMonth = dataVal[mask].groupby(['Site', 'YYMM']).mean()
    dataValMonth.reset_index(inplace = True)


    plt.figure(figsize = (6, 3))
    bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
                lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
    bm.drawcoastlines(color='gray',linewidth=1)
    
    for isite in sites:    
        temp = dataVal[dataVal.Site == isite]
        plt.scatter(temp['Longitude(Degrees)'].iloc[0], temp['Latitude(Degrees)'].iloc[0], 
                        s = 25, vmin = 1e-3, vmax = 1e-1, edgecolor = 'k', 
                        linewidth = 0.5, zorder = 10)
    
        plt.text(temp['Longitude(Degrees)'].iloc[0] * 1.01, temp['Latitude(Degrees)'].iloc[0]* 1.01,
                 isite, fontsize = 8)
    plt.title('AERONET')
    
    print('%s %s' % (iROI, sites))

#%%
# site_stats = pd.DataFrame(index = list(set(dataVal.Site)), columns = ['k', 'b', 'R2', 'RMSE', 'MAE', 'P'])
# for isite in list(set(dataVal.Site)):
#     temp = dataVal[dataVal.Site == isite]
    
#     # for i, idata in enumerate(list(paras.keys())):
#     #     ipara = paras[idata]
#     #     mask = temp['diffAOD_%s' % (idata)]
        
#     #     X1, X2 = temp['Absorption_AOD[550nm]'][mask], temp[ipara][mask]
#     #     if idata == 'OMAERUV':
#     #         X1 = temp['Absorption_AOD[500nm]'][mask]
#     try: 
#         mask = temp.diffAOD_MODIS
#         X1, X2 = temp['Absorption_AOD[550nm]'][mask], temp['Y_pred'][mask]
#         slope, intercept, r_value, p_value, std_err = stats.linregress(X1, X2)
#         perc = (abs(X1 - X2) <= temp.AAOD_err[mask]).sum() / len(X1) * 1e2
    
#         site_stats.loc[isite] = np.c_[slope, intercept, X1.corr(X2), RMSE(X1, X2), MAE(X1, X2), perc]
#     except:
#         pass

#%%

mask = dataVal['diffAOD_MODIS'] & dataVal['diffAOD_OMAERUV'] &\
                    dataVal['diffAOD_MERRA2'] & dataVal['diffSSA_OMAERUV'] &\
                    (dataVal['AOD550(AERONET)'] >= 0.1)
percent = pd.DataFrame()
sites = list(set(dataVal.Site))
for i, isite in enumerate(sites): 
    percent = percent.append(dataVal[mask][dataVal[mask].Site == isite].groupby('type').count().T.iloc[0:1] / len(dataVal[mask][dataVal[mask].Site == isite]))
percent.index = sites
#%%

colors = sns.color_palette("hls", 5)

paras = {
         'MODIS': 'Y_pred',
         'OMAERUV': 'AAOD500',
         'MERRA2': 'AAOD'}

sites = [
        'Ubon_Ratchathani', 'Chiang_Mai_Met_Sta', 'Doi_Ang_Khang','Tsumkwe', 'Lubango', 'Maun_Tower', 'Mongu_Inn', 'SANTA_CRUZ_UTEPSA',  'Alta_Floresta', 'Ji_Parana_SE', 'Rio_Branco',
          'XiangHe', 'Beijing',  'Beijing-CAMS', 'Beijing_PKU', 'Beijing_RADI', 'Fresno_2', 'Rimrock',
        'Banizoumbou', 'IER_Cinzana', 'Dakar', 'Tamanrasset_INM', 'Medenine-IRA', 'Zinder_Airport', 'Santa_Cruz_Tenerife',
          'Adelaide_Site_7',  'Tumbarumba', 'Hampton_University', 'La_Jolla', 'Camborne_MO', 'Arica', 'Hada_El-Sham',
      'NASA_LaRC', 'COVE_SEAPRISM', 'NEON_Harvard', 'St_Louis_University',  'Sigma_Space_Corp', 'NEON_CVALLA',
      ]


count = dataVal[mask].groupby('Site').count().lat_g
sites = list(dataVal[mask].groupby('Site').count()[(count > 20) & (count < 30)].index)

mask = dataVal['diffAOD_MODIS'] & dataVal['diffAOD_OMAERUV'] &\
                    dataVal['diffAOD_MERRA2'] & dataVal['diffSSA_OMAERUV'] &\
                    (dataVal['AOD550(AERONET)'] >= 0.1)


# sites_ = []
# for isite in sites: 
#     if (dataVal[mask].Site == isite).sum() >= 40: 
#         sites_.append(isite)

percent.loc[sites].plot(kind = 'bar', stacked = True)


sites_ = [
            'Banizoumbou', 'IER_Cinzana', 'Dakar', 'Tamanrasset_INM',
            'Maun_Tower', 'Mongu_Inn', 'SANTA_CRUZ_UTEPSA', 'Tsumkwe', 
            'Jaipur', 'XiangHe', 'Beijing', 'Kanpur',
            'Yakutsk', 'MD_Science_Center', 
            'Barcelona', 'Munich_University', 
          ]


percent.loc[sites_].plot(kind = 'bar', stacked = True)

markers = ['o', 's', 'x', '*']

types = {'BB': ['Maun_Tower', 'Mongu_Inn', 'SANTA_CRUZ_UTEPSA', 'Tsumkwe'],
         'Dust': ['Banizoumbou', 'IER_Cinzana', 'Dakar', 'Tamanrasset_INM'],
         'Mixed': ['Jaipur', 'XiangHe', 'Beijing', 'Kanpur'],
         'Other': ['Yakutsk', 'MD_Science_Center', 'Barcelona', 'Munich_University']}
fig = plt.figure(figsize = (8, 6))

for k, itype in enumerate(types.keys()):
    ax = fig.add_axes([0.1 + (k) % 2 * 0.4, 0.7 - (k) // 2 * 0.4, 0.325, 0.3])
    for j, isite in enumerate(types[itype]): 
        temp = dataVal[mask][dataVal[mask].Site == isite]
        
        for i, idata in enumerate(list(paras.keys())):
            X1 = temp['Absorption_AOD[550nm]']
            X2 = temp[paras[idata]]
            if idata == 'OMAERUV':
                X1 = temp['Absorption_AOD[500nm]']
        
            plt.scatter(X1, X2, marker = markers[j])
    
#%% validate by type
paras = {
         'MODIS': 'Y_pred',
         'OMAERUV': 'AAOD500',
         'MERRA2': 'AAOD'}             
paras_ssa = {'MODIS': 'SSA_pred',
         'OMAERUV': 'SSA500',
         'MERRA2': 'SSA'}
paras_aod = {'MODIS': 'AOD550(MODIS)',
         'OMAERUV': 'AOD500',
         'MERRA2': 'AOD'}
dataVal['type'] = aerosolType(dataVal)
stats_type = pd.DataFrame()

labels = ['DNN-F11', 'OMAERUV', 'MERRA-2']
colors = sns.color_palette("hls", 3)
alphas = [0.6, 0.4, 0.2]
markers = ['s', 'x', '.']
fig = plt.figure(figsize = (6, 6))
for k, itype in enumerate(['Smoke', 'Dust', 'Mixed', 'Other']):
    ax = fig.add_axes([0.1 + (k) % 2 * 0.5, 0.6 - (k) // 2 * 0.525, 0.35, 0.35])
    dataType = dataVal[mask & (dataVal.type == itype)]
    
    temp = pd.DataFrame(index = labels, columns = ['RMSE', 'MAE', 'k', 'b', 'R^2', 'P', 'N', 'ME_AAOD', 'ME_AOD', 'ME_SSA', 'Type', 'Data'])
        
    for i, idata in enumerate(paras.keys()):
        X1 = dataType['Absorption_AOD[550nm]']
        X2 = dataType[paras[idata]]
        X3 = dataType['AOD550(AERONET)']
        X4 = dataType[paras_aod[idata]]
        X5 = dataType['Single_Scattering_Albedo[550nm]']
        X6 = dataType[paras_ssa[idata]]
                      
                    
        if idata == 'OMAERUV':
            X1 = dataType['Absorption_AOD[500nm]']
            X3 = dataType['AOD500(AERONET)']
            X5 = dataType['Single_Scattering_Albedo[500nm]']
        slope, intercept, r_value, p_value, std_err = stats.linregress(X1, X2)
        K, b, _, _,_ = stats.linregress(X3, X4)
        perc = (abs(X4 - X3) <= 0.03).sum() / len(X1) * 1e2
        print(itype, idata, )
        plt.scatter(X1, X2, color = colors[i], label = labels[i],
                    marker = markers[i], edgecolor = None, alpha = alphas[i])
        temp.loc[labels[i]] = RMSE(X1, X2), MAE(X1, X2), slope, intercept, X1.corr(X2), (abs(X1 - X2) <= dataType.AAOD_err).sum() / len(X1) * 1e2, len(X1), (X2 - X1).mean(), (X4 - X3).mean(), (X6 - X5).mean(), itype, labels[i]
    plt.plot([0, 1], [0, 1], ':', color = 'gray', linewidth = 1)
    plt.plot([0, 0.5], [0, 1], '--', color = 'gray', linewidth = 1)
    plt.plot([0, 1], [0, 0.5], '--', color = 'gray', linewidth = 1)

    plt.title('(%s) %s' % (plotidx[k], itype))
    plt.xlabel(r'AAOD$^A$')
    plt.ylabel(r'AAOD')
    ymax = np.ceil(max(X1.max(), X2.max()) * 10) / 10
    plt.xlim([0, ymax])
    plt.ylim([0, ymax])
    # ax.xaxis.get_major_formatter().set_powerlimits((0,1))
    # ax.yaxis.get_major_formatter().set_powerlimits((0,1))
    plt.xticks(np.linspace(0, ymax, 3))
    plt.yticks(np.linspace(0, ymax, 3))
    
    
    

    stats_type = stats_type.append(temp)
plt.legend()    
plt.savefig(figdir + 'Type_vld_residue.png', dpi = 300, transparent = True)  




stats_type = stats_type.groupby(['Type', 'Data']).min()
# save 

for ipara in stats_type.columns:
    if ipara in ['k', 'b', 'R^2', 'ME_AOD', 'ME_SSA']: 
        stats_type[ipara] = stats_type[ipara].map(lambda x: '%1.2f' % x)
    if ipara in ['ME_AAOD']: 
        stats_type[ipara] = stats_type[ipara].map(lambda x: '%1.3f' % x)
    if ipara in ['RMSE', 'MAE']:
        stats_type[ipara] = stats_type[ipara].map(lambda x: '%1.2e' % x)
    if ipara in ['P']:
        stats_type[ipara] = stats_type[ipara].map(lambda x: '%i' % x)
        
stats_type.to_csv('Type_stats.csv')
#%% validate by type

    
dataVal['type'] = aerosolType(dataVal)
stats_type = pd.DataFrame()

colors = sns.color_palette("hls", 3)
alphas = [0.6, 0.4, 0.2]
markers = ['s', 'x', '.']
fig = plt.figure(figsize = (6, 6))
for k, itype in enumerate(['Smoke', 'Dust', 'Mixed', 'Other']):
    ax = fig.add_axes([0.1 + (k) % 2 * 0.5, 0.6 - (k) // 2 * 0.525, 0.35, 0.35])
    dataType = dataVal[mask & (dataVal.type == itype)]
    
    temp = pd.DataFrame(index = labels, columns = ['RMSE', 'MAE', 'k', 'b', 'R^2', 'P', 'N', 'Type', 'Data'])
        
    for i, idata in enumerate(paras.keys()):
        X1 = dataType['Single_Scattering_Albedo[550nm]']
        X2 = dataType[paras_ssa[idata]]
        if idata == 'OMAERUV':
            X1 = dataType['Single_Scattering_Albedo[500nm]']
        slope, intercept, r_value, p_value, std_err = stats.linregress(X1, X2)
        plt.scatter(X1, X2, color = colors[i], label = labels[i],
                    marker = markers[i], edgecolor = None, alpha = alphas[i])
        temp.loc[labels[i]] = RMSE(X1, X2), MAE(X1, X2), slope, intercept, X1.corr(X2), (abs(X1 - X2) <= dataType.AAOD_err).sum() / len(X1) * 1e2, len(X1), itype, labels[i]
    plt.plot([0.03, 1], [0, 0.97], '--', color = 'gray', linewidth = 1)
    plt.plot([0, 0.97], [0.03, 1], '--', color = 'gray', linewidth = 1)

    plt.title('(%s) %s' % (plotidx[k], itype))
    plt.xlabel(r'AAOD$^A$')
    plt.ylabel(r'AAOD')
    ymax = np.ceil(max(X1.max(), X2.max()) * 10) / 10
    plt.xlim([0.8, 1])
    plt.ylim([0.8, 1])
    ax.xaxis.get_major_formatter().set_powerlimits((0,1))
    ax.yaxis.get_major_formatter().set_powerlimits((0,1))
    # plt.xticks(np.linspace(0, ymax, 3))
    # plt.yticks(np.linspace(0, ymax, 3))

    stats_type = stats_type.append(temp)
plt.legend()    
plt.savefig(figdir + 'Type_vld.png', dpi = 300, transparent = True)  

#%%     
stats_roi = pd.DataFrame()
fig = plt.figure(figsize = (12, 8))
for j, isite in enumerate(sites_[:]): 
    ROI = ROIs[iROI]
    x, y = dataVal['Longitude(Degrees)'], dataVal['Latitude(Degrees)']
    ROImask = (dataVal.Site == isite)
    
    colors = sns.color_palette("hls", 3)
    labels = ['DNN-F11', 'OMAERUV', 'MERRA-2']
    
    
    ax = fig.add_axes([0.1 + (j) % 2 * 0.4, 0.7 - (j ) // 2 * 0.3, 0.325, 0.2])
    
    temp = pd.DataFrame(index = labels, columns = ['RMSE', 'MAE', 'R^2', 'P', 'N', 'Site', 'Data'])
    for i, idata in enumerate(list(paras.keys())):
        ipara = paras[idata]
        mask = ROImask & dataVal['diffAOD_MODIS'] & dataVal['diffAOD_OMAERUV'] &\
                dataVal['diffAOD_MERRA2'] & dataVal['diffSSA_OMAERUV'] &\
                (dataVal['AOD550(AERONET)'] >= 0.1)
    
        mm = dataVal[mask]
        mstd = dataVal[mask]
        
        mm.index = pd.to_datetime(mm.timeStamp * 1e9)
        X1, X2 = mm['Absorption_AOD[550nm]'], mm[ipara]
        X1_std = mstd['Absorption_AOD[550nm]']
        if idata == 'OMAERUV':
            X1 = mm['Absorption_AOD[500nm]']
            X1_2 = mm['Absorption_AOD[500nm]']
            X1_2_std = mstd['Absorption_AOD[500nm]']
            
        
        X2.plot(color = colors[i], label = labels[i])
        temp.loc[labels[i]] = RMSE(X1, X2), MAE(X1, X2), X1.corr(X2), (abs(X1 - X2) <= mm.AAOD_err).sum() / len(X1) * 1e2, len(X1), isite, labels[i]

    print(isite, len(X1))
    stats_roi = stats_roi.append(temp)
    X1.plot(color = 'k', linestyle = '--', linewidth = 1, label = 'AERONET(550)')
    X1_2.plot(color = 'gray', linestyle = '--', marker = '', linewidth = 1, label = 'AERONET(500)')
    plt.title('(%s) %s time series' % (plotidx[j], isite))
    if j == 5:
        plt.legend(ncol = 2)
    plt.ylabel('AAOD')
    # plt.ylim(0.5e-2, 3e-2)
    ax.yaxis.get_major_formatter().set_powerlimits((0,1))
plt.savefig(figdir + 'Site_vld.png', dpi = 300, transparent = True)  



stats_roi = stats_roi.groupby(['Site', 'Data']).min()
# save 

for ipara in stats_roi.columns:
    if ipara in ['R^2']: 
        stats_roi[ipara] = stats_roi[ipara].map(lambda x: '%1.2f' % x)
    if ipara in ['RMSE', 'MAE']:
        stats_roi[ipara] = stats_roi[ipara].map(lambda x: '%1.2e' % x)
    if ipara in ['P']:
        stats_roi[ipara] = stats_roi[ipara].map(lambda x: '%i' % x)
        
stats_roi.to_csv('Site_stats.csv')