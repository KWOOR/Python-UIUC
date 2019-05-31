# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 18:18:37 2019

@author: kur7
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import datetime
import statsmodels.api as sm
from scipy import stats
from scipy.stats.stats import pearsonr
from sklearn.preprocessing import scale
import sys
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

mod = sys.modules[__name__]
def get_data(file_name):
    data = pd.read_csv(file_name,na_values=['#VALUE!', '#DIV/0!','N/A','#REF!','#NAME?'], encoding='CP949',engine='python')
    data.index = pd.to_datetime(data['Date'],format='%Y-%m-%d')    
#    data.pop('Date')     
    return data

data1=get_data('SA1.csv')
data2=get_data('SA2.csv')
data=pd.concat([data1,data2])
del data1, data2
def get_backtest_term(data):                                                         
    backtest_term = pd.DataFrame(pd.unique(data.index))
    backtest_term.index = backtest_term.iloc[:,0]
#    test_term=backtest_term.loc[test_period:] 
    return backtest_term 
   
def find_cointegrated_pairs(dataframe, critial_level = 0.05): 
    n = dataframe.shape[1] # the length of dateframe
    pvalue_matrix = np.ones((n, n)) # initialize the matrix of p
    keys = dataframe.keys() # get the column names
    pairs = [] # initilize the list for cointegration
    for i in range(n):
        for j in range(i+1, n): # for j bigger than i
            stock1 = np.log(dataframe[keys[i]]) # obtain the price of two contract
            stock2 = np.log(dataframe[keys[j]])
            result = sm.tsa.stattools.coint(stock1, stock2) # get conintegration
            pvalue = result[1] # get the pvalue
            pvalue_matrix[i, j] = pvalue
            if pvalue < critial_level: # if p-value less than the critical level 
                pairs.append((keys[i], keys[j], pvalue)) # record the contract with that p-value
             
    return pvalue_matrix, pairs    


def golden_selection(data1, data2):
    lower = -20
    upper = 20
    errlimit = 0.0001
    iteration = 1000
    g = 0.618033989
    c = 1
    log1 = np.log(data1)-np.log(data1).shift(1)
    
    log2 = np.log(data2)-np.log(data2).shift(1)
        
    spread = np.log(data1)-c*np.log(data2)
        
    longrun = np.mean(spread)
        
    spreade = spread-longrun
        
    rohs = pearsonr(spreade.iloc[1:],spreade.iloc[:-1])[0]
    
    rohr = pearsonr(log1.dropna(), log2.dropna())[0]
        

    for i in range(iteration):
        x1 = lower + g*(upper-lower)
        x2 = upper - g*(upper-lower)
        
        c = x1
        spread = spread = np.log(data1)-c*np.log(data2)
        longrun = np.mean(spread)
        spreade = spread-longrun
        rohs = pearsonr(spreade.iloc[1:],spreade.iloc[:-1])[0]
        rohr = pearsonr(log1.dropna(), log2.dropna())[0]

        fx1 = rohs
        
        c = x2
        spread = spread = np.log(data1)-c*np.log(data2)
        longrun = np.mean(spread)
        spreade = spread-longrun
        rohs = pearsonr(spreade.iloc[1:],spreade.iloc[:-1])[0]
        rohr = pearsonr(log1.dropna(), log2.dropna())[0]
        fx2 = rohs
        
        if fx1 > fx2 : 
            upper = x1
        else:
            lower = x2
        
        if upper- lower < errlimit :
            c = (upper + lower)/2
            break
    return c

def assert_normality(data):
    statistic, pvalue = stats.shapiro(data)
    return pvalue


def get_pair(data):
    returns = data.pct_change().dropna()    
    X=scale(returns)
    pca = PCA(n_components=100)
    pca.fit(X)
    N_PRIN_COMPONENTS= np.where(pca.explained_variance_ratio_.cumsum() >= explain_ratio)[0][0]
    pca = PCA(n_components=N_PRIN_COMPONENTS)
    pca.fit(returns)
    X=np.array(pca.components_.T)
    X = preprocessing.StandardScaler().fit_transform(X)
    clf = DBSCAN(eps=DBSCAN_eps, min_samples=3)
    clf.fit(X)
    clustered = clf.labels_ 
    clustered_series = pd.Series(index=returns.columns, data=clustered.flatten())
    clustered_series = clustered_series[clustered_series != -1]
    counts = clustered_series.value_counts() 
    ticker_count_reduced = counts[(counts>1) & (counts<=9999)]
    fin_pair = []
    buff = pd.DataFrame()
    for k, which_clust in enumerate(ticker_count_reduced.index):
        tickers = clustered_series[clustered_series == which_clust].index
        pvalues, pairs= find_cointegrated_pairs(data[tickers])
        p = pd.DataFrame(pairs,columns=['S1','S2','Pvalue'])
        p_sorted = p.sort_index(by='Pvalue').reset_index(drop=True)   
        
        normal_pvalue = []
        for j in range(len(p_sorted)):
            c = golden_selection(data[p_sorted.iloc[j][0]], data[p_sorted.iloc[j][1]])            
            spread = np.log(data[p_sorted.iloc[j][0]]) - c*np.log(data[p_sorted.iloc[j][1]])
            normal_pvalue.append(assert_normality(spread))
        
        p_sorted['Normal_Pvalue'] = normal_pvalue
        buff = pd.concat([buff, p_sorted])
        del p_sorted
    buff['Pvalue'] = buff['Pvalue'].rank()
    buff['Normal_Pvalue'] = buff['Normal_Pvalue'].rank(ascending =False)
    buff['Total'] = (buff['Normal_Pvalue']+buff['Pvalue'])
    buff.sort_index(by='Total').reset_index(drop=True)
    buff = buff[:25]
    for i in range(len(buff)):
        fin_pair.append((buff['S1'][i], buff['S2'][i]))
    return fin_pair
    

backtest_term = get_backtest_term(data)
backtest_term = backtest_term.dropna(axis=0,how='any')
backtest_term = backtest_term.sort_index()

Close = data.pivot(index='Date', columns='Ticker', values='Close').set_index(backtest_term.index,drop=True).dropna(how='all',axis=0)
Open = data.pivot(index='Date', columns='Ticker', values='Open').set_index(backtest_term.index,drop=True).dropna(how='all',axis=0)
High = data.pivot(index='Date', columns='Ticker', values='High').set_index(backtest_term.index,drop=True).dropna(how='all',axis=0)
Low = data.pivot(index='Date', columns='Ticker', values='Low').set_index(backtest_term.index,drop=True).dropna(how='all',axis=0)
del backtest_term

find = np.sum(np.isnan(Close)*1,axis=0)
for i in range(len(find)):
    if find[i] !=0:
       ix = find.index[i]
       Close.pop(ix)
       Open.pop(ix)
       High.pop(ix)
       Low.pop(ix)
       
Close = Close['2007-02-08':]
Open = Open['2007-02-08':]
High = High['2007-02-08':]
Low = Low['2007-02-08':]


l=locals()
rolling_window = 130
from_trade = 252

explain_ratio = 0.60  #PCA에서 몇프로 이상 설명하는 갯수를 뽑을 것인가!?!?  지금은 70%이상 설명하는 갯수를 뽑음
DBSCAN_eps = 1.9  #DBSCAN 파라미터
#trading_start = '2016-10-24'

formation_start_date = [ Close.index.values[i+1] for i in range(0, len(Close), rolling_window)]
trading_start_date = [Close.index.values[i+1] for i in range(from_trade, len(Close) , rolling_window)]


#%%


final_pair = []
for i in range(len(trading_start_date) ):
    data = Close.loc[formation_start_date[i]: trading_start_date[i]-1]
    final_pair.append(get_pair(data))
    print(i)


#%%

pair = pd.DataFrame()

for i in range(len(final_pair)):
    empty = pd.DataFrame(final_pair[i])
    pair = pd.concat([pair, empty])    

pair.to_csv("PAIR5.csv")

