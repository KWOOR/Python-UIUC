# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:03:28 2019

@author: kur7
"""

import pandas as pd
import scipy.stats as sps
import statsmodels as stm
import numpy as np
import matplotlib.pyplot as plt

crsp= pd.read_csv('CRSP.csv', index_col = 'date')
retail = pd.read_csv('retail_frac.csv', index_col = 'date')
svi = pd.read_csv('SVI.csv', index_col = 'date')
svi_AGL = pd.read_csv('multiTimeline.csv', skiprows=1, index_col = '일')

gm = pd.read_csv('geoMap (2).csv', skiprows=1, index_col = 'Country').dropna(axis=0)

#%%
''' #1 '''

for i in range(len(gm)):
    if gm.iloc[i,:].values[0] == '<1' :
        gm.iloc[i,:] = np.nan
gm = gm.dropna(axis=0)
gm = gm.astype('int')
gm[gm == np.max(gm).values].dropna().index.values[0]

print('The city of the HeadQuater of AGL is Sydney, ', gm[gm == np.max(gm).values].dropna().index.values[0])
print('The name of AGL is AGL Energy')
print('The mean is ', np.mean(svi_AGL.values) )
print('The std is', np.std(svi_AGL.values))
print('The skewness is ', sps.skew(svi_AGL.values))
plt.hist(svi_AGL.values)

#%%
''' #2 '''

retail[retail['ticker'] == 'AGL'] , crsp[crsp['TICKER'] == 'AGL'], svi['AGL']
pd.concat([ retail[retail['ticker'] == 'AGL'] , crsp[crsp['TICKER'] == 'AGL'], svi['AGL'] ] , axis=1)




































