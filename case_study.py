#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  1 21:54:48 2021

@author: kanonyui
"""
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
cmap = matplotlib.cm.coolwarm
cmap3 = shiftedColorMap(cmap, start=0, midpoint = 0.5, stop=1, name='shifted')
cmap = matplotlib.cm.cubehelix
cmap1 = shiftedColorMap(cmap, start=0, midpoint = 0.25, stop=1, name='shifted')


t1 = time.time()

plt.close('all')
#%% 
ROIs = {
        'SAF2': {'E': 50, 'W': -30, 'N': 40, 'S': -40},
        'SAF1': {'E': 50, 'W': -30, 'N': 40, 'S': -40},
        'NAF2': {'E': 50, 'W': -30, 'N': 40, 'S': -40},
        'NAF1': {'E': 50, 'W': -30, 'N': 40, 'S': -40},
        

        # 'GOBI': {'E': 100, 'W':70, 'N': 60, 'S': 20},
        # 'CAN': {'E': -70, 'W': -120, 'N': 70, 'S': 30},
        # 'global': {'E': 180, 'W': -180, 'N': 90, 'S': -90},
        # 'AFC': {'E': 50, 'W': -30, 'N': 40, 'S': -40},
        # 'E America': {'E': -60, 'W': -80, 'N': 45, 'S': 30},
        # 'S America': {'E': -30, 'W': -80, 'N': 5, 'S': -30},
        }

dates = {
        'NAF1': pd.to_datetime('2019-08-07'),
        'NAF2': pd.to_datetime('2019-07-10'),
        'GOBI': pd.to_datetime('2019-05-19'),
        'CAN': pd.to_datetime('2019-05-31'),
        'SAF1': pd.to_datetime('2019-02-18'),
        'SAF2': pd.to_datetime('2019-02-14'),
        'global': pd.to_datetime('2019-10-28'),
        'S America': pd.to_datetime('2019-09-25'),
        
        }


for iROI in ROIs:
    ROI = ROIs[iROI]
    date = dates[iROI]
    temp = pd.read_pickle(dataOutputDir + '%s_output/%s_output_%04i-%02i-%02i.pickle'  
                                  % (expName, expName, date.year, date.month, date.day))



    # OMAERUV at 550 nm
    EAE = Angstorm(388, temp['AOD388'], 500, temp['AOD500'])
    temp['AOD550'] = wvldepAOD(500, temp['AOD500'], 550, EAE)
    
    AAE = Angstorm(388, temp['AAOD388'], 500, temp['AAOD500'])
    temp['AAOD550'] = wvldepAOD(500, temp['AAOD500'], 550, AAE)
    
    temp['SSA550'] = 1 - temp['AAOD550'] / temp['AOD550']  
    
    temp['SSA_pred'] = 1- temp['Y_pred'] / temp['AOD550(MODIS)']
    temp['SSA'] = 1- temp['AAOD'] / temp['AOD']
    temp = temp[temp['SSA_pred'] >= 0]
     
    temp.reset_index(inplace = True)
    temp.loc[np.isnan(temp.AAOD550), 'AAOD550'] = np.nan
# =============================================================================
# mask invalid samples
# =============================================================================
    mask = (~np.isinf(temp['SSA_pred'])) & (temp['SSA_pred']<=1) & \
           (~np.isinf(temp['SSA_pred'])) & (temp['AOD550(MODIS)'] >= temp['Y_pred']) & \
           (temp['As'] <= 0.3) & (temp['AAOD550'] >= 0) & (temp['AOD550'] >= 0) & \
           (temp['AOD550(MODIS)'] >= 0.1)
    
    temp = temp[mask]

    lat = temp['lat_g']
    lon = temp['lon_g']

    
    fig = plt.figure(figsize = (8, 6.5))
    temp['diffAAODmm'] = temp['Y_pred'] - temp['AAOD']
    temp['diffAAODmo'] = temp['Y_pred'] - temp['AAOD550']
    titles = ['$AAOD^{pred}$', '$AAOD^{pred}$ - $AAOD^O$', '$AAOD^{pred}$ - $AAOD^M$']

    for i, ipara in enumerate(['Y_pred', 'diffAAODmo', 'diffAAODmm']):
        # plt.subplot(4, 3, c + 1)
        ax = fig.add_axes([0.0 + i * 0.3, 0.7, 0.3, 0.2])
        X = temp[ipara] 
        bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
                    lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
        bm.drawcoastlines(color='gray',linewidth=1)
        bm.drawparallels(np.arange(-45, 46, 30), labels=[True,False,False,False], linewidth = 0, fontsize = 8)
        bm.drawmeridians(np.arange(-90, 91, 30), labels=[False,False,False,True], linewidth = 0, fontsize = 8)
        XX, YY = np.meshgrid(np.arange(-180, 181, 10), np.arange(-90, 91, 10))
        plt.scatter(XX, YY, c = 'lightgray', s = 1e4)
        if i == 0: 
            cb = plt.scatter(lon, lat, c = X, cmap = cmap2, norm=matplotlib.colors.LogNorm(),
                    s = 1, vmin = 1e-2, vmax = 1e-1, )
            plt.colorbar(fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10, extend = 'both')
            plt.title('(%s) %s' % (plotidx[i], titles[i]), fontsize = 9)
    
        else:
            cb = plt.scatter(lon, lat, c = X, cmap = cmap3, 
                        s = 1, vmin = -5e-2, vmax = 5e-2, )
            plt.colorbar(fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10, extend = 'both', ticks = [-5e-2, 0, 5e-2])

            plt.title('(%s) %s' % (plotidx[i], titles[i]), fontsize = 9)


    temp['diffSSAmm'] = temp['SSA_pred'] - temp['SSA']
    temp['diffSSAmo'] = temp['SSA_pred'] - temp['SSA550']
    titles = ['$SSA^{pred}$', '$SSA^{pred}$ - $SSA^O$', '$SSA^{pred}$ - $SSA^M$']

    for i, ipara in enumerate(['SSA_pred', 'diffSSAmo', 'diffSSAmm']):
        # plt.subplot(4, 3, c + 1)
        ax = fig.add_axes([0.0 + i * 0.3, 0.4, 0.3, 0.2])
        X = temp[ipara] 
        bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
                    lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
        bm.drawcoastlines(color='gray',linewidth=1)
        bm.drawparallels(np.arange(-45, 46, 30), labels=[True,False,False,False], linewidth = 0, fontsize = 8)
        bm.drawmeridians(np.arange(-90, 91, 30), labels=[False,False,False,True], linewidth = 0, fontsize = 8)
        XX, YY = np.meshgrid(np.arange(-180, 181, 10), np.arange(-90, 91, 10))
        plt.scatter(XX, YY, c = 'lightgray', s = 1e3)
        if i == 0: 
            cb = plt.scatter(lon, lat, c = X, cmap = cmap1,
                        s = 1, vmin = 0.8, vmax = 1, )
            plt.colorbar(fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10, extend = 'both', ticks = [0.8, 0.9, 1.0])
            plt.title('(%s) %s' % (plotidx[i + 3], titles[i]), fontsize = 9)
        else:
            cb = plt.scatter(lon, lat, c = X, cmap = cmap3, 
                        s = 1, vmin = -5e-2, vmax = 5e-2, )
            cbar = plt.colorbar(fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10, extend = 'both', ticks = [-5e-2, 0, 5e-2])
            plt.title('(%s) %s' % (plotidx[i + 3], titles[i]), fontsize = 9)

    
   
        
    temp['diffAODmm'] = temp['AOD550(MODIS)'] - temp['AOD']
    temp['diffAODmo'] = temp['AOD550(MODIS)'] - temp['AOD550']
    titles = ['$AOD^{MODIS}$', '$AOD^{MODIS}$ - $AOD^O$', '$AOD^{MODIS}$ - $AOD^M$']

    for i, ipara in enumerate(['AOD550(MODIS)', 'diffAODmo', 'diffAODmm']):
        # plt.subplot(4, 3, c + 1)
        ax = fig.add_axes([0.0 + i * 0.3, 0.1, 0.3, 0.2])
        X = temp[ipara] 
        bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
                    lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
        bm.drawcoastlines(color='gray',linewidth=1)
        bm.drawparallels(np.arange(-45, 46, 30), labels=[True,False,False,False], linewidth = 0, fontsize = 8)
        bm.drawmeridians(np.arange(-90, 91, 30), labels=[False,False,False,True], linewidth = 0, fontsize = 8)
        XX, YY = np.meshgrid(np.arange(-180, 181, 10), np.arange(-90, 91, 10))
        plt.scatter(XX, YY, c = 'lightgray', s = 1e4)
        if i == 0: 
            cb = plt.scatter(lon, lat, c = X, cmap = cmap2, 
                        s = 1, vmin = 0, vmax = 1, )
            plt.colorbar(fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10,  extend = 'both', ticks = [0, 0.5, 1])
            plt.title('(%s) %s' % (plotidx[i + 6], titles[i]), fontsize = 9)
        else:
            cb = plt.scatter(lon, lat, c = X, cmap = cmap3, 
                             s = 1, vmin = -0.5, vmax = 0.5, )
            plt.colorbar(fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10,  extend = 'both', ticks = [-5e-1, 0, 5e-1])
            plt.title('(%s) %s' % (plotidx[i + 6], titles[i]), fontsize = 9)

    # for i, ipara in enumerate(['SSA_pred', 'SSA550', 'SSA']):
    #     # plt.subplot(4, 3, c + 1)
    #     ax = fig.add_axes([0.0 + i * 0.3, 0.4, 0.3, 0.2])
    #     X = temp[ipara] 
    #     bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
    #                 lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
    #     bm.drawcoastlines(color='gray',linewidth=1)
    #     bm.drawparallels(np.arange(-45, 46, 30), labels=[True,False,False,False], linewidth = 0, fontsize = 8)
    #     bm.drawmeridians(np.arange(-90, 91, 30), labels=[False,False,False,True], linewidth = 0, fontsize = 8)
    #     XX, YY = np.meshgrid(np.arange(-180, 181, 10), np.arange(-90, 91, 10))
    #     plt.scatter(XX, YY, c = 'lightgray', s = 1e3)
    #     cb = plt.scatter(lon, lat, c = X, cmap = cmap1,
    #                 s = 1, vmin = 0.8, vmax = 1, )
    #     plt.colorbar(fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10, extend = 'both', ticks = [0.8, 0.9, 1.0])
    #     plt.title('(%s) %s' % (plotidx[i + 3], titles[i]), fontsize = 9)
    
            
    plt.savefig(figdir + 'case_study_%s.png' % (iROI), dpi = 300, transparent = True)  

#%%    
fig = plt.figure(figsize = (6, 6))
for i, iROI in enumerate(ROIs):
    ROI = ROIs[iROI]
    date = dates[iROI]
    temp = pd.read_pickle(dataOutputDir + '%s_output/%s_output_%04i-%02i-%02i.pickle'  
                                  % (expName, expName, date.year, date.month, date.day))

    # OMAERUV at 550 nm
    EAE = Angstorm(388, temp['AOD388'], 500, temp['AOD500'])
    temp['AOD550'] = wvldepAOD(500, temp['AOD500'], 550, EAE)
    
    AAE = Angstorm(388, temp['AAOD388'], 500, temp['AAOD500'])
    temp['AAOD550'] = wvldepAOD(500, temp['AAOD500'], 550, AAE)
    
    temp['SSA550'] = 1 - temp['AAOD550'] / temp['AOD550']  
    
    temp['SSA_pred'] = 1- temp['Y_pred'] / temp['AOD550(MODIS)']
    temp['SSA'] = 1- temp['AAOD'] / temp['AOD']
    temp = temp[temp['SSA_pred'] >= 0]
     
    temp.reset_index(inplace = True)
    temp.loc[np.isnan(temp.AAOD550), 'AAOD550'] = np.nan
# =============================================================================
# mask invalid samples
# =============================================================================
    mask = (~np.isinf(temp['SSA_pred'])) & (temp['SSA_pred']<=1) & \
           (~np.isinf(temp['SSA_pred'])) & (temp['AOD550(MODIS)'] >= temp['Y_pred']) & \
           (temp['As'] <= 0.3) & (temp['AAOD550'] >= 0) & (temp['AOD550'] >= 0) & \
           (temp['AOD550(MODIS)'] >= 0.1)
    
    temp = temp[mask]

    lat = temp['lat_g']
    lon = temp['lon_g']


    ax = fig.add_axes([0.1 + (i) % 2 * 0.5, 0.6 - (i) // 2 * 0.525, 0.35, 0.35])
    X = temp['residue'] 
    bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
                lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
    bm.drawcoastlines(color='gray',linewidth=1)
    bm.drawparallels(np.arange(-45, 46, 30), labels=[True,False,False,False], linewidth = 0, fontsize = 8)
    bm.drawmeridians(np.arange(-90, 91, 30), labels=[False,False,False,True], linewidth = 0, fontsize = 8)
    XX, YY = np.meshgrid(np.arange(-180, 181, 10), np.arange(-90, 91, 10))
    plt.scatter(XX, YY, c = 'lightgray', s = 1e4)
    cb = plt.scatter(lon, lat, c = X, cmap = cmap2, 
            s = 1, vmin = 0, vmax = 4, )
    plt.colorbar(fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10, extend = 'both', ticks = [0, 2, 4, 6])
    plt.title('(%s) %04i-%02i-%02i' % (plotidx[i], dates[iROI].year, dates[iROI].month, dates[iROI].day), fontsize = 9)

plt.savefig(figdir + 'case_study_UVAI.png', dpi = 300, transparent = True)  
