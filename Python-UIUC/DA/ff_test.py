# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 22:04:45 2019

@author: kur7
"""
import numpy as np
import statsmodels.formula.api as smf
import pandas as pd

ff_test = pd.read_excel('FF_Test.xlsx')
ff_data = pd.read_csv('F-F_Research_Data_5_Factors_2x3_daily.csv')
mom = pd.read_csv('F-F_Momentum_Factor_daily.csv')
del mom['Date']


aaa = pd.concat([ff_data/100, ff_test, mom/100], axis=1).dropna()

del aaa['Date']

aaa= aaa.rename(columns = {'Mkt-RF':'Mkr_RF'})


aa = aaa.copy()
del aa['RMW'], aa['CMA']


before_crisis = aaa.iloc[:500]
after_crisis = aaa.iloc[500:]

#%%


print("==================================== ML Result ====================================")


model = smf.ols(formula='Portfolio ~ Mkr_RF + SMB + HML + RMW + CMA + RF',data=before_crisis)
result = model.fit()
print(result.summary())
#%%

print("==================================== ML Result ====================================")


model = smf.ols(formula='Portfolio ~ Mkr_RF + SMB + HML + RMW + CMA + RF',data=after_crisis)
result = model.fit()
print(result.summary())

#%%
print("==================================== ML Result ====================================")


model = smf.ols(formula='Portfolio ~ Mkr_RF + SMB + HML  + Mom + RF ',data=before_crisis)
result = model.fit()
print(result.summary())

#%%

print("==================================== ML Result ====================================")


model = smf.ols(formula='Portfolio ~ Mkr_RF + SMB + HML  + Mom + RF ',data=after_crisis)
result = model.fit()
print(result.summary())

