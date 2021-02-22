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
import shutil
import time
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import glob
from scipy import ndimage
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors
from pylab import cm
import seaborn as sns
import pandas as pd
import random
from pandas import Series, DataFrame, Panel
import string
import pickle
from scipy import spatial
from datetime import datetime
from scipy import stats
from otherFunctions import *
import h5py
import netCDF4
from MODIS import *
from MultiSensor import *
from AERONETtimeSeries_v3 import *
from pyOMI import *
from SVRfunc import *
from sklearn.model_selection import train_test_split
from supportFunctions import *

plt.close('all')
dataOutputDir = '/nobackup/users/sunj/'
dataInputDir = '/nobackup_1/users/sunj/'
figdir = '/usr/people/sunj/Dropbox/Paper_Figure/ML_AAOD/'

dataOutputDir = '/Users/kanonyui/PhD_program/Data/'
dataInputDir = '/Users/kanonyui/PhD_program/Data/'
figdir = '/Users/kanonyui/Dropbox/Paper_Figure/ML_AAOD/'

ROI = {'S': -90, 'N': 90, 'W': -180, 'E': 180}
startdate = '%4i-%02i-%02i' % (2014, 1, 1)
enddate   = '%4i-%02i-%02i' % (2019, 12, 31)
dates = pd.date_range(startdate, enddate)

def func(lat, lon):
    return round(lat * 2) / 2, round(lon * 1.6) / 1.6

t1 = time.time()



#%%
plt.close('all')
figidx = string.ascii_lowercase
# =============================================================================
# load training data
# =============================================================================
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
#  quality control
# =============================================================================
# temp = pd.read_pickle(dataOutputDir + 'MERRA-2_OMAERUV_MODIS_AERONET_collocation/MERRA-2_OMAERUV_MODIS_AERONET_collocation_2006-2019.pickle')
# # pre-processing
# temp['doy'] = temp['dateTimeLocal'].dt.dayofyear + (temp['dateTimeLocal'].dt.hour + temp['dateTimeLocal'].dt.minute / 60 + temp['dateTimeLocal'].dt.second / 3600) / 24

# temp = temp[temp['Absorption_AOD[550nm]'] > 0]
# temp['SSA'] = 1 - temp['AAOD'] / temp['AOD']
# temp['AOD550(AERONET)'] = temp['Absorption_AOD[550nm]'] / (1 - temp['Single_Scattering_Albedo[550nm]'])
# # quality control of AOD and SSA
# temp['diffSSA'] = abs(temp['Single_Scattering_Albedo[500nm]'] - temp['SSA500']) <= 0.03
# temp['diffAOD'] = np.nan
# land = temp['landoceanMask'] >= 0.5
# temp.loc[temp['landoceanMask'] >= 0.5, 'diffAOD'] = abs(temp['AOD550(AERONET)'][land] - temp['AOD550(MODIS)'][land]) <= (0.05 + 0.15 * temp['AOD550(AERONET)'][land])
# temp.loc[temp['landoceanMask'] < 0.5, 'diffAOD'] = abs(temp['AOD550(AERONET)'][~land] - temp['AOD550(MODIS)'][~land]) <= (0.03 + 0.05 * temp['AOD550(AERONET)'][~land])

# temp = temp[temp.diffSSA & temp.diffAOD & (temp.AI388std <= 0.5)]
# data = temp.copy()

from trainingData import *
features = ['AI388', 'AOD550(MODIS)', 'Haer_t1',\
            'vza', 'raa', 'As', 'Ps',\
            'lat', 'lon', 'doy']
data = dataTrain(features)
data = data.data.copy()


features = ['AI388', 'AOD550(MODIS)', 'Haer_t1',
            'lat', 'lon', 'doy',
            'sza', 'vza', 'raa', 'As', 'Ps', 'R354obs', 'R388obs']
parameters = features + ['Absorption_AOD[550nm]']
labels = [r'UVAI$_{388}$', r'AOD$_{550}^{M}$', r'H$_{aer}^{t}$', \
          'Lat', 'Lon', 'DOY', \
          'SZA', 'VZA', 'RAA', r'a$_s$', r'P$_s$', 'R354', 'R388', r'AAOD$_{550}^{A}$']

data = data.dropna(how = 'all', axis = 1)
f = 1
# allfeatures = set(data.groupby(['lat', 'lon']).mean().keys()) - set('Absorption_AOD[550nm]')
X = data[features][::f]
Y = data['Absorption_AOD[550nm]'][::f]

Scores = pd.DataFrame(index = features, columns = ['R2', 'MI', 'MIC', 'FI', 'PFI', 'DF', 'RFE'])
#%%
# =============================================================================
# histogram
# =============================================================================
fig = plt.figure(figsize = (12, 8))
for i, ipara in enumerate(features):
    ax = fig.add_axes([0.1 + i % 3 * 0.275 , 0.65 - i//3 * 0.275, 0.225, 0.2])
    # ax = plt.subplot(3, len(features) // 3 + 1, i + 1)
    sns.distplot(X[ipara], bins = 12, kde = False, label = 'mean:%1.2f''\n''std:%1.2f''\n''median:%1.2f''\n''max:%1.2f''\n''min:%1.2f' 
             % (X[ipara].mean(), X[ipara].std(), X[ipara].median(), X[ipara].max(), X[ipara].min()))
    ax.yaxis.get_major_formatter().set_powerlimits((0,1))
    plt.legend()
    # plt.xlabel('(%s) %s' % (figidx[i], labels[i]))
plt.savefig(figdir + 'Feature_space_histogram.png', dpi = 300, transparent = True)
#%% 
# =============================================================================
# correlation matrix
# =============================================================================
fig = plt.figure(figsize = (7, 6))
ax1 = fig.add_axes([0.2, 0.2, 0.7, 0.7])
corr = data[parameters].corr('spearman')
mask = np.triu(np.ones_like(corr, dtype=np.bool))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
cmap = shiftedColorMap(cmap, midpoint=0.38, stop=1, name='shifted')
ax = sns.heatmap(corr, mask = mask, cmap = cmap, vmin = -0.4, vmax = 0.6,
            annot = True, fmt = '.2f', annot_kws={'size':7}, linewidth = 0.5, cbar_kws = {'shrink': 0.5},
            xticklabels = labels, yticklabels = labels)
ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 8, rotation = 45)
ax.set_yticklabels(ax.get_xmajorticklabels(), fontsize = 8)
plt.savefig(figdir + 'Feature_space_CorrMatrix.png', dpi = 300, transparent = True)

Scores['R2'] = corr.loc['Absorption_AOD[550nm]']

#%%
#%% 
# =============================================================================
# mutural information
# =============================================================================
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
selector = SelectKBest(mutual_info_regression, k=10)
selector.fit(X, Y)
Scores['MI'] = selector.scores_.reshape(-1) 


fig = plt.figure(figsize = (8, 6))
ax = fig.add_axes([0.3, 0.2, 0.65, 0.7])
Scores.nlargest(20, 'MI')['MI'].plot(kind='barh', color = 'k', fig = fig, fontsize = 8, legend = False)
plt.xlabel('Mutural Information')
ax.set_yticklabels(labels, fontsize = 8)
plt.savefig(figdir + 'Feature_selection_mi.png', dpi = 300, transparent = True)



# =============================================================================
# maximum information coefficient
# =============================================================================
from minepy import MINE
mic = []

m = MINE()
for i in range(len(features)):
    m.compute_score(X.values[:, i], Y)
    mic.append(m.mic())
Scores['MIC'] = mic

fig = plt.figure(figsize = (4, 3))
ax = fig.add_axes([0.3, 0.2, 0.65, 0.7])
Scores.nlargest(20, 'MIC')['MIC'].plot(kind='barh', color = 'k', fig = fig, fontsize = 8, legend = False)
# plt.xscale('log')
plt.xlabel('Maximum Information Coefficient')
ax.set_yticklabels(labels, fontsize = 8)
# plt.savefig(figdir + 'Feature_selection_mic.png', dpi = 300, transparent = True)

#%% 
# =============================================================================
# feature importance
# =============================================================================
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


X_train, X_vld, Y_train, Y_valid = train_test_split(X, Y, test_size = 0.95, random_state = 42)
X_train_mean, X_train_std = X_train.mean(axis = 0), X_train.std(axis = 0)
Y_train_mean, Y_train_std = Y_train.mean(axis = 0), Y_train.std(axis = 0)

X_train_norm = standardization(X_train, X_train_mean, X_train_std)
X_vld_norm = standardization(X_vld, X_train_mean, X_train_std)

model = RandomForestRegressor()

model.fit(X_train_norm, Y_train)        
print(model.feature_importances_)
print(model.score(X_train_norm, Y_train), model.score(X_vld_norm, Y_valid))#use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
Scores['FI'] = model.feature_importances_
fig = plt.figure(figsize = (6, 6))
ax = fig.add_axes([0.25, 0.2, 0.65, 0.7])
Scores.nlargest(20, 'FI')['FI'].plot(kind='barh', color = 'k')
plt.xscale('log')
plt.xlabel('Feature Importance')
# ax.set_yticklabels(labels)
ax.set_yticklabels(features, fontsize = 10)
plt.savefig(figdir + 'Feature_selection_importance.png', dpi = 300, transparent = True)
#%% Permutation of feature importance
from sklearn.metrics import r2_score
from rfpimp import permutation_importances
import eli5
from eli5.sklearn import PermutationImportance

def r2(rf, X_train, y_train):
    return r2_score(y_train, rf.predict(X_train))
model = RandomForestRegressor()
perm = PermutationImportance(model, random_state=1).fit(X_train_norm, Y_train)
eli5.show_weights(perm, feature_names = X.columns.tolist())

perm_imp_rfpimp = permutation_importances(model, X_train_norm, Y_train, r2)
Scores['PFI'] = perm_imp_rfpimp['Importance']

fig = plt.figure(figsize = (6, 6))
ax = fig.add_axes([0.25, 0.2, 0.65, 0.7])
Scores.nlargest(20, 'PFI')['PFI'].plot(kind='barh', color = 'k')
plt.xscale('log')
plt.xlabel('Permutation Feature Importance')
# ax.set_yticklabels(labels)
plt.savefig(figdir + 'Feature_selection_importance_permutation.png', dpi = 300, transparent = True)


#%% drop one feature off
from sklearn.base import clone 

def imp_df(column_names, importances):
    df = pd.DataFrame({'feature': column_names,
                       'feature_importance': importances}).sort_values('feature_importance', ascending = False).reset_index(drop = True)
    return df

def drop_col_feat_imp(model, X_train, y_train, random_state = 42):
    
    # clone the model to have the exact same specification as the one initially trained
    model_clone = clone(model)
    # set random_state for comparability
    model_clone.random_state = random_state
    # training and scoring the benchmark model
    model_clone.fit(X_train, y_train)
    benchmark_score = model_clone.score(X_train, y_train)
    # list for storing feature importances
    importances = []
    
    # iterating over all columns and storing feature importance (difference between benchmark and new model)
    for col in X_train.columns:
        model_clone = clone(model)
        model_clone.random_state = random_state
        model_clone.fit(X_train.drop(col, axis = 1), y_train)
        drop_col_score = model_clone.score(X_train.drop(col, axis = 1), y_train)
        importances.append(benchmark_score - drop_col_score)
    
    importances_df = imp_df(X_train.columns, importances)
    return importances_df

importances_df = drop_col_feat_imp(model, X_train_norm, Y_train, random_state = 42)
importances_df.index = importances_df['feature']

Scores['DOF'] = importances_df['feature_importance']

fig = plt.figure(figsize = (6, 6))
ax1 = fig.add_axes([0.25, 0.2, 0.65, 0.7])
importances_df.feature_importance.nlargest(len(features)).plot(kind='barh', color = 'k')
plt.xlabel('Drop one off')
plt.show()
plt.savefig(figdir + 'Feature_selection_drop_one_off.png', dpi = 300, transparent = True)

#%%
# from sklearn.feature_selection import RFE
# import matplotlib.pyplot as plt
# from sklearn.neural_network import MLPRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import LassoCV
# #model = RandomForestRegressor()
# score_list = []
# high_score = 0
# for numf in np.arange(1, len(features) + 1): 
#     # Create the RFE object and rank each pixel
#     rfe = RFE(estimator=model, n_features_to_select=numf, step=1)
#     rfe.fit(X_train_norm, Y_train)
#     X_train_rfe = rfe.fit_transform(X_train_norm,Y_train)
#     X_valid_rfe = rfe.transform(X_vld_norm)  
#     model.fit(X_train_rfe,Y_train)
#     score = model.score(X_valid_rfe, Y_valid)
#     score_list.append(score)
#     if(score>high_score):
#         high_score = score
#         nbest = numf
        
        
# print("Optimum number of features: %d" %nbest)
# print("Score with %d features: %f" % (nbest, high_score)) 

# rfe = RFE(estimator=model, n_features_to_select=nbest, step=1)
# rfe.fit(X_train_norm, Y_train)

# ranking = pd.Series(rfe.ranking_, index = features)
# support = pd.Series(rfe.support_, index = features)
   
# fig = plt.figure(figsize = (6, 6))
# ax1 = fig.add_axes([0.25, 0.2, 0.65, 0.7])
# ranking.nlargest(len(features)).plot(kind='barh', color = 'k')
# plt.xlabel('Rank')
# plt.savefig(figdir + 'Feature_selection_RFE.png', dpi = 300, transparent = True)

# print(support)



from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

t3 = time.time()
RFEscores = pd.DataFrame(index = Scores.index, columns = np.arange(10))
RFEranks = pd.DataFrame(index = Scores.index, columns = np.arange(10))
for i in range(100): 
    model = RandomForestRegressor(50)
    rfecv = RFECV(estimator=model, cv=10, scoring = 'neg_mean_squared_error')
    rfecv.fit(X_train_norm, Y_train)
    RFEscores[i] = rfecv.grid_scores_
    RFEranks[i] = rfecv.ranking_
    

    print("Optimal number of features : %d" % rfecv.n_features_)

t4 = time.time() 
print('Time: %1.2f' % (t4 - t3)) 

RFErank = pd.DataFrame(rfecv.ranking_, index = labels[:-1], columns = ['rank'])

fig = plt.figure(figsize = (6, 4))
ax1 = fig.add_axes([0.25, 0.2, 0.65, 0.6])
RFErank.nlargest(len(features), 'rank').plot(kind='barh', color = 'k', legend = False)
plt.xlabel('Rank')
plt.savefig(figdir + 'Feature_selection_RFECV.png', dpi = 300, transparent = True)

#%%
# Plot number of features VS. cross-validation scores
fig = plt.figure(figsize = (4, 3))
ax = fig.add_axes([0.15, 0.15, 0.8, 0.6])
plt.xlabel("Number of features")
plt.ylabel("RMSE")
plt.errorbar(range(1, RFEscores.shape[0] + 1), np.sqrt(-RFEscores).mean(axis = 1), 
         yerr = np.sqrt(-RFEscores).std(axis = 1),
         color = 'royalblue', marker = 's')
ax.yaxis.get_major_formatter().set_powerlimits((0,1))
plt.title('RFE model performance')
plt.savefig(figdir + 'Score_vs_num_of_features_wrapper.png', dpi = 300, transparent = True)

    
Scores['RFE_rank'] = rfecv.ranking_
Scores['RFE_score'] = rfecv.grid_scores_


#%%
# =============================================================================
# SelectFromModel
# =============================================================================
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor	
model = RandomForestRegressor()
clf = RandomForestRegressor()

# Set a minimum threshold of 0.25
sfm = SelectFromModel(clf, threshold=0.0)
sfm.fit(X_train_norm, Y_train)
n_features = sfm.transform(X).shape[1]



def GetCVScore(estimator,X,y):
    from sklearn.model_selection import  cross_val_score
    nested_score = cross_val_score(clf, X=X, y=y, cv=10, scoring = 'neg_mean_squared_error')
    nested_score_mean = nested_score.mean()
    return nested_score_mean

# Reset the threshold till the number of features equals two.
# Note that the attribute can be set directly instead of repeatedly
# fitting the metatransformer.

nested_scores = []
n_features_list = []
while n_features > 2:
    sfm.threshold += 0.01
    X_transform = sfm.transform(X_train_norm)
    n_features = X_transform.shape[1]

    nested_score = GetCVScore(estimator=clf, X=X_transform, y=Y_train)
    nested_scores.append(nested_score)
    n_features_list.append(n_features)
    # print("nested_score: %s"%nested_score)
    # print("n_features: %s"%n_features)
    # print("threshold: %s"%sfm.threshold)

fig = plt.figure(figsize = (4, 3))
ax = fig.add_axes([0.2, 0.2, 0.65, 0.6])
plt.xlabel("Number of features")
plt.ylabel("RMSE")
plt.plot(n_features_list, -np.array(nested_scores), marker = 's', color = 'k', label = 'Selected')
# plt.scatter(X.shape[1],GetCVScore(estimator=clf, X=X_train_norm, y=Y_train), c=u'r',marker=u'*',label = 'old feature')
plt.title("Embedded")
plt.ylim(0.3e-4, 0.6e-4)
ax.yaxis.get_major_formatter().set_powerlimits((0,1))
plt.savefig(figdir + 'Score_vs_num_of_features_embedded.png', dpi = 300, transparent = True)



#%%

X1, X2, X3 = 'AI388', 'Absorption_AOD[550nm]', 'AOD550(MODIS)'


fig = plt.figure(figsize = (6, 4))
regime = [0.05, 0.1, 0.2, 0.5, 1]
for i in range(len(regime)):
    ax = fig.add_axes([0.1 + i%3 * 0.275, 0.55 - i//3 * 0.45 , 0.225, 0.325])
    if i == 0:
        mask = data[X3] < regime[i]
        plt.title(labels[1]+ '<%1.2f' % (regime[i]))
    if i >= len(regime) - 1:
        mask = data[X3] > regime[i]
        plt.title(labels[1]+ '>%1.1f' % (regime[i]))
    if (i > 0) & (i < len(regime) - 1):
        mask = (data[X3] >= regime[i]) & (data[X3] <= regime[i + 1])
        plt.title(r'%1.1f$\leq$' % (regime[i]) + labels[1]+ '$\leq$%1.1f' % (regime[i + 1]))
        
    plt.scatter(data[X1][mask], data[X2][mask], c = data[X3][mask], 
                vmin =1e-4, vmax = 1, marker = 's', alpha = 0.5, s = 20, edgecolor = 'k', linewidth = 0.25,
                 cmap = 'magma_r')
    plt.ylim(0, 0.2)
    
    if i % 3 == 0:
        plt.yticks(np.arange(0, 0.21, 0.1))
        plt.ylabel(labels[-1])
    else:
        plt.yticks([])

    if i  >= 3:
        plt.xlabel(labels[0])
    else:
        plt.xticks([])

ax = fig.add_axes([0.1 + 5%3 * 0.275, 0.55 - 5//3 * 0.45 , 0.225, 0.325])
plt.scatter(data[X1], data[X2], c = data[X3], 
            marker = 's', alpha = 0.5, s = 20, edgecolor = 'k', linewidth = 0.1,
            vmin = 1e-1, vmax = 1, cmap = 'magma_r')
plt.xlabel(labels[0])
plt.title('All AOD regimes')
plt.ylim(0, 0.2)
plt.yticks([])
cax = fig.add_axes([0.9, 0.2, 0.025, 0.6])
plt.colorbar(cax = cax, label = labels[1], ticks = np.arange(0, 1.1, 0.25), extend = 'both')
plt.savefig(figdir + 'AAOD_vs_UVAI.png', dpi = 300, transparent = True)




Scores.index = labels[:-1]
Scores_norm = Scores[['R2', 'MI', 'MIC', 'FI', 'PFI']].copy()
Scores_norm['RFE'] = Scores['RFE_score'].copy()

Scores_norm['R2'] = abs(Scores_norm['R2'])
Scores_norm['RFE'] = abs(Scores_norm['RFE'])
Scores_norm = (Scores_norm - Scores_norm.min()) / (Scores_norm.max() - Scores_norm.min())
plt.figure(figsize = (4,4))
sns.heatmap(Scores_norm, cmap = sns.color_palette("RdBu_r", len(features)), vmin = 1e-3, vmax = 1, 
            norm=matplotlib.colors.LogNorm(), cbar_kws = {'extend': 'both'})
plt.savefig(figdir + 'Score_norm_matrix.png', dpi = 300, transparent = True)

#%%
# plt.figure()
# sns.heatmap(Scores, annot = True, color = 'w')

old_width = pd.get_option('display.max_colwidth')
pd.set_option('display.max_colwidth', -1)
Scores.to_html('files.html',escape=False,index=False,sparsify=True,border=0,index_names=False,header=False)


#%%
Ranks = pd.DataFrame()
Ranks['R2'] = abs(Scores.R2).sort_values(ascending = True).rank()
Ranks['MI'] = Scores.MI.sort_values(ascending = True).rank()
Ranks['MIC'] = Scores.MIC.sort_values(ascending = True).rank()
Ranks['RFE'] = ( - Scores.RFE_score).sort_values(ascending = True).rank()
Ranks['FI'] = Scores.FI.sort_values(ascending = True).rank()
# Ranks['PFI'] = Scores.PFI.sort_values(ascending = True).rank()

Ranks = len(features) + 1 - Ranks
import seaborn as sns
ax = plt.figure(figsize = (5,4))
sns.heatmap(Ranks, cmap = sns.color_palette("magma_r", len(features)), vmin = 1, vmax = 11, 
            linewidth = 0.1, annot = True, 
            cbar_kws = {'label': 'Rank', 'ticks': np.arange(1, 12, 1)})
plt.savefig(figdir + 'Rank_matrix.png', dpi = 300, transparent = True)

t2 = time.time()

#%%
Scores['RFEranks'] = RFEranks.median(axis = 1).values
fig = plt.figure(figsize =  (4, 3))
ax1 = fig.add_axes([0.15, 0.25, 0.75, 0.6])
ax2 = ax1.twinx()
Scores['MIC'].plot(kind = 'bar', color = 'royalblue', ax = ax1,
                   position = 1, width = 0.3, edgecolor = 'w')
Scores['RFEranks'].plot(kind = 'bar', color = 'tomato', ax = ax2,
                        position = 0, width = 0.3, edgecolor = 'w')
ax1.set_ylabel('MIC', color = 'royalblue')
ax1.tick_params(axis='y', colors='royalblue')
ax2.set_ylabel('RFE rank (median)', color  = 'tomato')
ax2.tick_params(axis='y', colors='tomato')
ax2.set_xticklabels(labels, fontsize = 8)
plt.savefig(figdir + 'Feature_selection_summary.png', dpi = 300, transparent = True)


#%%
Scores['RFE_mean'] = (np.sqrt(-RFEscores)).mean(axis = 1).values
Scores['RFE_std'] = (np.sqrt(-RFEscores)).std(axis = 1).values

fig = plt.figure(figsize =  (4, 2.5))
ax1 = fig.add_axes([0.15, 0.25, 0.75, 0.6])
ax2 = ax1.twinx()
Scores['MIC'].plot(kind = 'bar', ax = ax1, color = 'royalblue', hatch = '//',
                   position = 1, width = 0.3, edgecolor = 'royalblue')
Scores['RFE_mean'].plot(kind = 'bar', ax = ax2, color = 'none',
                        position = 0, width = 0.3, edgecolor = 'k')
ax1.set_ylabel('MIC', color = 'royalblue')
ax1.tick_params(axis='y', colors='royalblue')

ax2.set_ylabel('RFE RMSE (mean)', color  = 'k')
ax2.tick_params(axis='y', colors='k')
ax2.set_xticklabels(labels, fontsize = 8)

ax1.yaxis.get_major_formatter().set_powerlimits((0,1))
ax2.yaxis.get_major_formatter().set_powerlimits((0,1))
plt.savefig(figdir + 'Feature_selection_summary.png', dpi = 300, transparent = True)

# print('Time: %1.2f' % (t2 - t1))