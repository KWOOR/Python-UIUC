# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:47:04 2019

@author: kur7
"""
%pylab inline

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing
from statsmodels.tsa.stattools import coint
from sklearn.preprocessing import scale
import statsmodels.api as sm
from scipy import stats
from scipy.stats.stats import pearsonr
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix,roc_curve, auc,accuracy_score
import tensorflow as tf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from pykalman import KalmanFilter
import ffn
import datetime
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from scipy import poly1d
import seaborn as sns
from talib import func
import sys
import os
os.chdir('C:\\Users\\kur7\\OneDrive\\바탕 화면\\uiuc\\DA\\Daily')


#%%
def get_data(file_name):
    data = pd.read_csv(file_name,na_values=['#VALUE!', '#DIV/0!','N/A','#REF!','#NAME?'], encoding='CP949',engine='python')
    data.index = pd.to_datetime(data['Date'],format='%Y-%m-%d')    
#    data.pop('Date')     
    return data
data1=get_data('SA1.csv')
data2=get_data('SA2.csv')
data=pd.concat([data1,data2])
del data1, data2
#%%                                                      

def get_backtest_term(data):                                                         
    backtest_term = pd.DataFrame(pd.unique(data.index))
    backtest_term.index = backtest_term.iloc[:,0]
#    test_term=backtest_term.loc[test_period:] 
    return backtest_term 
   
#def test_matrix():
backtest_term = get_backtest_term(data)
backtest_term = backtest_term.dropna(axis=0,how='any')
backtest_term = backtest_term.sort_index()

#%%
Close = data.pivot(index='Date', columns='Ticker', values='Close').set_index(backtest_term.index,drop=True).dropna(how='all',axis=0)
Open = data.pivot(index='Date', columns='Ticker', values='Open').set_index(backtest_term.index,drop=True).dropna(how='all',axis=0)
High = data.pivot(index='Date', columns='Ticker', values='High').set_index(backtest_term.index,drop=True).dropna(how='all',axis=0)
Low = data.pivot(index='Date', columns='Ticker', values='Low').set_index(backtest_term.index,drop=True).dropna(how='all',axis=0)
#%%
find = np.sum(np.isnan(Close)*1,axis=0)
for i in range(len(find)):
    if find[i] !=0:
       ix = find.index[i]
       Close.pop(ix)
       Open.pop(ix)
       High.pop(ix)
       Low.pop(ix)

#%% Get Technical indicators
def get_technical_indicators(dataset):
    # Create 7 and 21 days Moving Average
    dataset['ma5'] = dataset['Close'].rolling(window=5).mean()
    dataset['ma10'] = dataset['Close'].rolling(window=10).mean()
    
    # Create MACD
    dataset['26ema'] = func.EMA(dataset['Close'],26)
    dataset['12ema'] = func.EMA(dataset['Close'],12)
    dataset['MACD'] = (dataset['12ema']-dataset['26ema'])

    # Create Bollinger Bands
    dataset['10sd'] = dataset['Close'].rolling(10).std()
    dataset['upper_band'] = dataset['ma5'] + (dataset['10sd']*2)
    dataset['lower_band'] = dataset['ma10'] - (dataset['10sd']*2)
    
    
    # Create Exponential moving average
    dataset['ema'] = dataset['Close'].ewm(com=0.5).mean()
    
    # Create Momentum
    dataset['momentum'] = func.MOM(dataset['Close'],20)
    
    # ROC
    dataset['roc'] = func.ROC(dataset['Close'],10)
    
    # Willam
    dataset['wpr'] = func.WILLR(dataset['High'], dataset['Low'], dataset['Close'], timeperiod=14)
    
    # ATR
    dataset['atr'] = func.ATR(dataset['High'], dataset['Low'], dataset['Close'], timeperiod=14)
    
    return dataset

#%% TI
ma5 = Close.rolling(window=5).mean()['2007-02-08':] 
print(sum(sum(np.isnan(ma5.values))))
ma5_open = Open.rolling(window=5).mean()['2007-02-08':]    
print(sum(sum(np.isnan(ma5_open.values))))

ma10 = Close.rolling(window=10).mean()['2007-02-08':]    
print(sum(sum(np.isnan(ma10.values))))
ma10_open = Open.rolling(window=10).mean()['2007-02-08':] 
print(sum(sum(np.isnan(ma10_open.values))))

ema_26 = Close.apply(lambda x : func.EMA(x,26),axis=0)['2007-02-08':]
print(sum(sum(np.isnan(ema_26.values))))
ema_26_open = Open.apply(lambda x : func.EMA(x,26),axis=0)['2007-02-08':]
print(sum(sum(np.isnan(ema_26_open.values))))

ema_12 = Close.apply(lambda x : func.EMA(x,12),axis=0)['2007-02-08':]
print(sum(sum(np.isnan(ema_12.values))))
ema_12_open = Open.apply(lambda x : func.EMA(x,12),axis=0)['2007-02-08':]
print(sum(sum(np.isnan(ema_12_open.values))))

sd_10 = Close.rolling(10).std()['2007-02-08':]
print(sum(sum(np.isnan(sd_10.values))))
sd_10_open = Open.rolling(10).std()['2007-02-08':]
print(sum(sum(np.isnan(sd_10_open.values))))

bb_up = (ma5+sd_10*2)['2007-02-08':]
print(sum(sum(np.isnan(bb_up.values))))
bb_up_open = (ma5_open+sd_10_open*2)['2007-02-08':]
print(sum(sum(np.isnan(bb_up_open.values))))

bb_low = (ma5-sd_10*2)['2007-02-08':]
print(sum(sum(np.isnan(bb_low.values))))
bb_low_open = (ma5_open-sd_10_open*2)['2007-02-08':]
print(sum(sum(np.isnan(bb_low_open.values))))

ema = Close.ewm(com=0.5).mean()['2007-02-08':]
print(sum(sum(np.isnan(ema.values))))
ema_open = Open.ewm(com=0.5).mean()['2007-02-08':]
print(sum(sum(np.isnan(ema_open.values))))

roc = Close.apply(lambda x : func.ROC(x,10),axis=0)['2007-02-08':]
print(sum(sum(np.isnan(roc.values))))
roc_open = Open.apply(lambda x : func.ROC(x,10),axis=0)['2007-02-08':]
print(sum(sum(np.isnan(roc_open.values))))

mom = Close.apply(lambda x : func.MOM(x,20),axis=0)['2007-02-08':]
print(sum(sum(np.isnan(mom.values))))
mom_open = Open.apply(lambda x : func.MOM(x,20),axis=0)['2007-02-08':]
print(sum(sum(np.isnan(mom_open.values))))

Close = Close['2007-02-08':]
Open = Open['2007-02-08':]
High = High['2007-02-08':]
Low = Low['2007-02-08':]

#%%
criteria = 0.8
returns = Close.pct_change()
returns = returns.iloc[1:,:]

returns_copy=returns
returns=returns.iloc[:int(len(returns)*criteria),:]  #Formation Period 동안의 수익률로 페어 찾고 분석 
returns=returns.dropna(axis=1)
#returns=returns.fillna(1)
#%% 과거 수익률을 이용하여 PCA를 통해 현재 시점의 리턴에 대한 설명력 측정 

'''100개의 component를 사용하여 확인'''

X=returns.values
X=scale(X)

pca = PCA(n_components=100)

pca.fit(X)

var= pca.explained_variance_ratio_

var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.plot(var1)
plt.title('PCA explained variance ratio')
plt.ylabel('Explained Percentage(%)')
plt.xlabel('Number of components')
plt.show()
#%% 
''' PCA , DBSCAN으로 클러스터링 (후보군 찾기) 
    70%정도의 설명력을 갖는 40개 components를 활용하여 분석
'''

N_PRIN_COMPONENTS = 50
pca = PCA(n_components=N_PRIN_COMPONENTS)

#ttt = pca.fit_transform(returns)
#
#tt = pca.components_
#loading = pca.components_.T*np.sqrt(pca.explained_variance_)
#loading = StandardScaler().fit_transform(loading)

pca.fit(returns)

pca.components_.T.shape

X=np.array(pca.components_.T)
X = preprocessing.StandardScaler().fit_transform(X)

#pca.components_.T.shape
#X=np.array(loading)
# Density-Based Spatial Clustering of Applications with Noise

clf = DBSCAN(eps=1.8, min_samples=3)
clf.fit(X)
labels = clf.labels_                                          # -1이면 구분이 안된것
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)   # 숫자 5, 총 5개의 클러스터 있음

print ("\nClusters discovered: %d" % n_clusters_)

clustered = clf.labels_   #{-1,-1,.....0....4... } 총 200개 사이즈

#%%
# clustered_series의 값이 같은 것들끼리 동일 cluster로 분류된 주식들을 의미함
clustered_series = pd.Series(index=returns.columns, data=clustered.flatten())
clustered_series_all = pd.Series(index=returns.columns, data=clustered.flatten())
clustered_series = clustered_series[clustered_series != -1]


CLUSTER_SIZE_LIMIT = 9999

#한 클러스터에 몇 개의 주식이 있는지 확인 
counts = clustered_series.value_counts() 

# ticker_count_reduced 는 cluster안에 주식이 1개 이상 9999개 이하인 경우의 종목개수를 카운트
ticker_count_reduced = counts[(counts>1) & (counts<=CLUSTER_SIZE_LIMIT)]

print("Clusters formed : %s " % len(ticker_count_reduced))
print ("Pairs to evaluate: %d" % (ticker_count_reduced*(ticker_count_reduced-1)).sum())

#%% 차원 축소하여 Cluster plot을 출

X_tsne = TSNE(learning_rate=1000, perplexity=25, random_state=1337).fit_transform(X)

plt.figure(1, facecolor='white')
plt.clf()
plt.axis('off')

plt.scatter(
    X_tsne[(labels!=-1), 0],
    X_tsne[(labels!=-1), 1],
    s=100,
    alpha=0.85,
    c=labels[labels!=-1],
    cmap=cm.Paired
)

plt.scatter(
    X_tsne[(clustered_series_all==-1).values, 0],
    X_tsne[(clustered_series_all==-1).values, 1],
    s=100,
    alpha=0.05
)

plt.title('T-SNE of all Stocks with DBSCAN Clusters Noted');

#%% Cluster Member Count 하여 bar plot 출력

plt.barh(range(len(clustered_series.value_counts())),clustered_series.value_counts())
plt.title('Cluster Member Counts')
plt.xlabel('Stocks in Cluster')
plt.ylabel('Cluster Number');

# Cluster에 존재하는 종목들의 standardized lof price plot 출력
cluster_vis_list = list(counts[(counts<20) & (counts>1)].index)[::-1]  

for clust in cluster_vis_list[0:len(cluster_vis_list)]:
    tickers = list(clustered_series[clustered_series==clust].index)
    means = np.log(Close[tickers].mean())
    data = np.log(Close[tickers]).sub(means)
    data.plot(title='Stock Time Series for Cluster %d' % clust)
#%% Cointegrate Test를 통한 Pair 선택
    
''' Cointegrate 이용한 Pair 찾기 (Mean - reverting 찾기)'''

def find_cointegrated_pairs(data, significance=0.05):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
#            S1 = data[keys[i]]
#            S2 = data[keys[j]]
            S1 = np.log(data[keys[i]])
            S2 = np.log(data[keys[j]])
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < significance:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs


cluster_dict = {}
for i, which_clust in enumerate(ticker_count_reduced.index):
    tickers = clustered_series[clustered_series == which_clust].index
    score_matrix, pvalue_matrix, pairs = find_cointegrated_pairs(
        Close[tickers]
    )
    cluster_dict[which_clust] = {}
    cluster_dict[which_clust]['score_matrix'] = score_matrix
    cluster_dict[which_clust]['pvalue_matrix'] = pvalue_matrix
    cluster_dict[which_clust]['pairs'] = pairs


pairs = []
for clust in cluster_dict.keys():
    pairs.extend(cluster_dict[clust]['pairs'])



print ("In those pairs, there are %d unique tickers.\n" % len(np.unique(pairs)))
print(pairs)


stocks = np.unique(pairs)
X_df = pd.DataFrame(index=returns.T.index, data=X)
in_pairs_series = clustered_series.loc[stocks]
stocks = list(np.unique(pairs))
X_pairs = X_df.loc[stocks]

never_erase = pairs
#%%

'''찾은 페어들의 종가 그래프와 스프레드 Plot '''

for i in range(len(pairs)):
    
     
     Close[list(pairs[i])].plot()                                   
     plt.axvline(x='2017-09-08')
     ax = (Close[pairs[i][1]]-Close[pairs[i][0]]).plot(title='Stock price of pairs and spread')
     ax.legend(['{}'.format(pairs[i][0]),'{}'.format(pairs[i][1]),'Spread'])

#%%


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
    
     
     
     
 

#%%    
     ''' 표준화시켜서 하기 '''



spread =[]
spread1 = []
for i in range(len(pairs)):
    NPI0 = Close[list(pairs[i])[0]]
    NPI0 = (NPI0 - NPI0.mean() )/ NPI0.std()
    
    NPI1 = Close[list(pairs[i])[1]]
    NPI1 = (NPI1 - NPI1.mean() )/ NPI1.std()
    c = golden_selection(Close[list(pairs[i])[0]], Close[list(pairs[i])[1]])
#    spread.append(NPI0-c*NPI1)
#    spread1.append(NPI0-NPI1)
    spread.append(np.log(Close[list(pairs[i])[0]]) - np.log(Close[list(pairs[i])[1]]))
    spread1.append(np.log(Close[list(pairs[i])[0]]) - c*np.log(Close[list(pairs[i])[1]]))
    pd.concat([NPI0, NPI1], axis=1).plot()

    plt.axvline(x='2017-09-08')
    ax = (NPI0 - NPI1).plot(title = 'Stock price of pairs and spread' )
    ax.legend(['{}'.format(pairs[i][0]),'{}'.format(pairs[i][1]),'Spread'])     


def assert_normality(data):
    """Description of assert_normality
    The Shapiro-Wilk test tests the null hypothesis that the data was drawn 
    from a normal distribution
    if test_stat and p_value close to 1 then data is normal
    """
#    print("Values " + str(data))
    statistic, pvalue = stats.shapiro(data)
#    print("Shapiro Statistic " + str(statistic) + " and p-value " + str(pvalue))
    if pvalue > 0.1:
        print("Normal")

    else:
        print("Not normal")

result = reg_m(np.log(Close[pairs[0][0]]), c*np.log(Close[pairs[0][1]]))

adfuller(np.log(Close[pairs[0][0]]) - c*np.log(Close[pairs[0][1]]))
     
for i in range(len(spread1)):
    
    print('Pair{}'.format(i+1))
    assert_normality(spread1[i][:int(len(spread1[i])/10)])

#    sm.qqplot(spread[i], line='s')  #qqplot check
    sns.kdeplot(spread1[i][:int(len(spread1[i])/10)])  #Density Function Check
    plt.show()


#%%
def KalmanFilterAverage(x):
    # Construct a Kalman filter
   
    kf = KalmanFilter(transition_matrices = [1],
                      observation_matrices = [1],
                      initial_state_mean = 0,
                      initial_state_covariance = 1,
                      observation_covariance=1,
                      transition_covariance=.01)

    # Use the observed values of the price to get a rolling mean
    state_means, _ = kf.filter(x.values)
    state_means = pd.Series(state_means.flatten(), index=x.index)
    return state_means

#  Kalman filter regression
def KalmanFilterRegression(x,y):

    delta = 1e-3
    trans_cov = delta / (1 - delta) * np.eye(2) # How much random walk wiggles
    obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)
    
    kf = KalmanFilter(n_dim_obs=1, n_dim_state=2, # y is 1-dimensional, (alpha, beta) is 2-dimensional
                      initial_state_mean=[0,0],
                      initial_state_covariance=np.ones((2, 2)),
                      transition_matrices=np.eye(2),
                      observation_matrices=obs_mat,
                      observation_covariance=2,
                      transition_covariance=trans_cov)
    
    # Use the observations y to get running estimates and errors for the state parameters
    state_means, state_covs = kf.filter(y.values)
    return state_means     
#%% Spread 모델을 이용하기 위한 회귀분석 및 Support Vector Machine 학습을 위한 T-score 계산

# Spread Model 구성을 위해 data 전처리 

''' Spread 모델을 이용하기 위한 회귀분석 및 T-score 계산'''
def reg_m(y, x):
    ones = np.ones(len(x))
    X = sm.add_constant(np.column_stack((x, ones)))
    results = sm.OLS(y, X).fit()
    return results

mod = sys.modules[__name__]
    
            
final_pairs = [('CL', 'KMB'),
 ('D', 'DUK'),
 ('DTE', 'PNW'),
 ('ED', 'PNW'),
 ('PNW', 'WEC'),
 ('CPB', 'K'),
 ('DUK', 'SRE'),
 ('ALE', 'DTE'),
 ('ALE', 'NEE'),
 ('IDA', 'NJR'),
 ('DTE', 'ES'),
 ('DTE', 'SWX'),
 ('DUK', 'SWX'),
 ('ED', 'SWX'),
 ('ES', 'PNW'),
 ('PNW', 'SWX'),
 ('SWX', 'WEC')]
     
pairs = final_pairs

''' 이제부터 모델 ㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱ '''



#%%
 #80%가 트레인셋, 20%가 테스트셋

              
def get_Tscore(data1,data2, status, absvalue=-1, Kalman=False): #data1은 종가, data2는 시가, status는 'T_price_{}' 이런식으로!!
    #status=T_price_{}, T_wma_{}, T_sma_{}, T_rsi_{}, T_mfi_{} string형식으로..
    for i in range(len(pairs)):
        d_A = data1[pairs[i][0]]-data2[pairs[i][0]]
        d_B=  data1[pairs[i][1]]-data2[pairs[i][1]]
        Y= (d_A/(data2[pairs[i][0]]+1e-10)).dropna()
        X= (d_B/(data2[pairs[i][1]]+1e-10)).dropna()
        result=reg_m(Y,X)
        X_t=result.resid
        
        resid=result.resid
        resid_1=resid.shift(1).dropna()
        resid=resid.iloc[1:]
        
        if Kalman==True:
            obs_mat = np.vstack([resid_1, np.ones(resid_1.shape)]).T[:, np.newaxis]
            delta = 1e-5
            trans_cov = delta / (1 - delta) * np.eye(2)
            kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,
                              initial_state_mean=np.zeros(2),
                              initial_state_covariance=np.ones((2, 2)),
                              transition_matrices=np.eye(2),
                              observation_matrices=obs_mat,
                              observation_covariance=1.0,
                              transition_covariance=trans_cov)
            state_means, state_covs = kf.filter(resid.values) #means 0이 slope, 1이 constant
            beta=state_means[:,0]
            const=state_means[:,1]
            error=resid- (resid_1*beta) -const
            mu=const/(1-beta)
            mu=np.hstack([mu[0], mu])
            sigma= np.sqrt( error.var()/(1-(beta**2)) )
            sigma=np.hstack([sigma[0], sigma])
            
        else:
            result=reg_m(resid, resid_1)
            const=result.params[1]
            beta=result.params[0]
            error=result.resid
            mu=const/(1-beta)
            sigma= np.sqrt( error.var()/(1-(beta**2)) )
        
        if absvalue==-1:  #방향성 맞추기
            if status=='T_price_{}':
                setattr(mod, status.format(i), pd.DataFrame( abs ( (X_t - mu) /sigma )).iloc[1:] ), 
                setattr(mod, 'label_{}'.format(i), pd.DataFrame((abs(X_t.shift(-1)) <  abs(X_t))*2-1).iloc[1:] )
            else:
                setattr(mod, status.format(i), pd.DataFrame( abs ( (X_t - mu) /sigma )) )
                
        elif absvalue==0: #PCA에 있던 ppt Score방식 사용
            if status=='T_price_{}':
                setattr(mod, status.format(i), pd.DataFrame(  (X_t - mu) /sigma ).iloc[1:] ) 
                buy_open= pd.DataFrame( ((X_t - mu) /sigma) > 1.25 )*1
                sell_open= pd.DataFrame( ((X_t - mu) /sigma) < -1.25 )*-1
                buy_open+=((buy_open+sell_open)==0)*(buy_open.shift(1)==1) * (pd.DataFrame( ((X_t-mu)/sigma) > 0.5))*1
                sell_open+=((buy_open+sell_open)==0)*(sell_open.shift(1)==-1) * (pd.DataFrame( ((X_t-mu)/sigma) <-0.75))*-1  

                setattr(mod, 'label_{}'.format(i), pd.DataFrame(buy_open+sell_open,dtype='i').iloc[1:] )
            else:
                setattr(mod, status.format(i), pd.DataFrame( (X_t - mu) /sigma ) )
                
        else: #거래 Signal 맞추기 
            if status=='T_price_{}':
                setattr(mod, status.format(i), pd.DataFrame(  (X_t - mu) /sigma ).iloc[1:] ) 
                buy_open= (pd.DataFrame(X_t > 0) & pd.DataFrame(X_t.shift(-1) < absvalue*X_t))*1
                sell_open= (pd.DataFrame(X_t < 0) & pd.DataFrame(X_t.shift(-1) > absvalue*X_t))*-1
                buy_open+=((buy_open+sell_open)==0)*(buy_open.shift(1)==1) * (pd.DataFrame(X_t > 0))*1
                sell_open+=((buy_open+sell_open)==0)*(sell_open.shift(1)==-1) * (pd.DataFrame(X_t <0))*-1  

                setattr(mod, 'label_{}'.format(i), pd.DataFrame(buy_open+sell_open,dtype='i').iloc[1:] )
            else:
                setattr(mod, status.format(i), pd.DataFrame( (X_t - mu) /sigma ) )
                
value=-1  #-1이면 방향성 맞추기, 0이면 거래 Signal 중에서 PCA ppt에 있던 S-score 방식 사용하기
kal=False #False면 Kalman Filter 안 씀 
get_Tscore(Close, Open, 'T_price_{}', value, kal)
get_Tscore(ma5, ma5_open, 'T_ma5_{}', value, kal)
get_Tscore(ma10, ma10_open, 'T_ma10_{}', value, kal)
get_Tscore(ema, ema_open, 'T_ema_{}', value, kal)
get_Tscore(ema_26, ema_26_open, 'T_ema26_{}', value, kal)
get_Tscore(ema_12, ema_12_open, 'T_ema12_{}', value, kal)
get_Tscore(sd_10, sd_10_open, 'T_sd10_{}', value, kal)
get_Tscore(bb_up, bb_up_open, 'T_bb_up_{}', value, kal)
get_Tscore(bb_low, bb_low_open, 'T_bb_low_{}', value, kal)


#%%
def MinMaxScale(data):
   
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    
    scale_data = numerator / (denominator + 1e-7)
    
    return scale_data

def makeX_train_test(data,rate):
    
    data = MinMaxScale(data)
    
    X_train = data[:int(len(data)*rate),:np.shape(data)[1]-1]
    X_test =  data[int(len(data)*rate):,:np.shape(data)[1]-1]
    
    return X_train, X_test

def makeY_train_test(data,rate):
     
    Y_train = data[:int(len(data)*rate),-1]
    Y_test = data[int(len(data)*rate):,-1]
    
    return Y_train, Y_test

#%%
    


l=locals()
pairs = final_pairs
for i in range(len(pairs)):
    setattr(mod, 'Pair{}'.format(i+1), pd.concat([l['T_price_%d'%i],l['T_ma5_%d'%i],
            l['T_ma10_%d'%i],l['T_ema_%d'%i], l['T_ema26_%d'%i], l['T_ema12_%d'%i]
            ,l['T_sd10_%d'%i],l['T_bb_up_%d'%i],l['T_bb_low_%d'%i],l['label_%d'%i]], axis=1, 
        join='inner').values)        
    
    setattr(mod, 'Pair{}X_train'.format(i+1), l['Pair%d'%(i+1)][:int(len(l['Pair%d'%(i+1)])*criteria),:-1])
    setattr(mod, 'Pair{}X_test'.format(i+1), l['Pair%d'%(i+1)][int(len(l['Pair%d'%(i+1)])*criteria):,:-1])
    setattr(mod, 'Pair{}Y_train'.format(i+1), l['Pair%d'%(i+1)][:int(len(l['Pair%d'%(i+1)])*criteria),-1])
    setattr(mod, 'Pair{}Y_test'.format(i+1), l['Pair%d'%(i+1)][int(len(l['Pair%d'%(i+1)])*criteria):,-1])


#%%

#%% Train Model for Pair

def Train_SVM(Xtrain,Xtest,Ytrain,Ytest, C, gamma,decision_function,kernel):
    
    clf = svm.SVC(C=C,gamma=gamma, decision_function_shape = decision_function,kernel = kernel)
    
    clf.fit(Xtrain,Ytrain)
    
    y_pred = clf.predict(Xtest)
    
    pred_df = pd.DataFrame(y_pred)
    
    conf_matrix=confusion_matrix(Ytest, y_pred)
      
    print("Accuracy: {}%".format(np.round(metrics.accuracy_score(Ytest, y_pred),2)*100)) 
    print(conf_matrix)
    print(classification_report(Ytest, y_pred))
  
    return pred_df

def Train_LinearSVM(Xtrain,Xtest,Ytrain,Ytest, C,multi_class,tol,max_iter):
    
    clf = svm.LinearSVC(C=C,multi_class=multi_class,tol=tol,max_iter=max_iter)
    
    clf.fit(Xtrain,Ytrain)
    
    y_pred = clf.predict(Xtest)
    
    pred_df = pd.DataFrame(y_pred)
    
    conf_matrix=confusion_matrix(Ytest, y_pred) 

    print("Accuracy: {}%".format(np.round(metrics.accuracy_score(Ytest, y_pred),2)*100))
    print(conf_matrix)
    print(classification_report(Ytest, y_pred))

    return pred_df     

def Train_GradientBoostingCalssifier(Xtrain,Xtest,Ytrain,Ytest,learning_rate,n_estimators):
    
    model = GradientBoostingClassifier(learning_rate=learning_rate,n_estimators=n_estimators)

    model.fit(Xtrain, Ytrain)
    
    y_pred = model.predict(Xtest)
    
    pred_df = pd.DataFrame(y_pred)
    
    conf_matrix=confusion_matrix(Ytest, y_pred)
    
    accuracy = accuracy_score(Ytest,y_pred)*100
    
    print("Accuracy: {}%".format(accuracy))
    print(conf_matrix)
    print(classification_report(Ytest, y_pred))
    
    return pred_df

def Train_RandomForestClassifier(Xtrain,Xtest,Ytrain,Ytest,n_estimators,criterion):
    
    model = RandomForestClassifier(n_estimators=n_estimators,criterion=criterion)

    model.fit(Xtrain, Ytrain)
    
    y_pred = model.predict(Xtest)
    
    pred_df = pd.DataFrame(y_pred)
    
    conf_matrix=confusion_matrix(Ytest, y_pred)

    accuracy = accuracy_score(Ytest,y_pred)*100
    
    print("Accuracy: {}%".format(accuracy))
    print(conf_matrix)
    print(classification_report(Ytest, y_pred))
   
    return pred_df


#%%
    

# Train result
pred_SVM_result = []
for i in range(len(final_pairs)):
    train = 'Pair{}X_train'.format(i+1)
    test = 'Pair{}X_test'.format(i+1)
    y_train = 'Pair{}Y_train'.format(i+1)
    y_test = 'Pair{}Y_test'.format(i+1)
    pred_SVM_result.append(Train_SVM(train,test,y_train,y_test,c,gamma,decision_function,kernel))

LinearSVM_result = []
for i in range(len(final_pairs)):
    train = 'Pair{}X_train'.format(i+1)
    test = 'Pair{}X_test'.format(i+1)
    y_train = 'Pair{}Y_train'.format(i+1)
    y_test = 'Pair{}Y_test'.format(i+1)
    LinearSVM_result.append(Train_LinearSVM(train,test,y_train,y_test,c,multi_class,tol,max_iter))

GBC_result = []
for i in range(len(final_pairs)):
    train = 'Pair{}X_train'.format(i+1)
    test = 'Pair{}X_test'.format(i+1)
    y_train = 'Pair{}Y_train'.format(i+1)
    y_test = 'Pair{}Y_test'.format(i+1)
    GBC_result.append(Train_GradientBoostingCalssifier(train,test,y_train,y_test,learning_rate_gb,n_estimators_GB))

RF_result = []
for i in range(len(final_pairs)):
    train = 'Pair{}X_train'.format(i+1)
    test = 'Pair{}X_test'.format(i+1)
    y_train = 'Pair{}Y_train'.format(i+1)
    y_test = 'Pair{}Y_test'.format(i+1)
    RF_result.append(Train_RandomForestClassifier(train,test,y_train,y_test,n_estimators_RF,criterion))












#%%
    


def get_ret(Y_train, Y_test, y_pred, num_of_pairs):
    answer=[]
    total=[]
    payoff=-1*Y_train*(Close[pairs[num_of_pairs][0]].pct_change().shift(-1).loc[T_ma5_0.index][:int(len(T_ma5_0)*criteria)])+\
    Y_train*(Close[pairs[num_of_pairs][1]].pct_change().shift(-1).loc[T_ma5_0.index][:int(len(T_ma5_0)*criteria)])
    print("Training 기간 동안 누적 수익률은:",(((payoff)+1).cumprod().iloc[-1]-1)*100,"%")
    payoff=-1*Y_test*(Close[pairs[num_of_pairs][0]].pct_change().shift(-1).loc[T_ma5_0.index][int(len(T_ma5_0)*criteria):])+\
    Y_test*(Close[pairs[num_of_pairs][1]].pct_change().shift(-1).loc[T_ma5_0.index][int(len(T_ma5_0)*criteria):])
    answer.append(((payoff)+1).cumprod())
    print("Test기간 다 맞췄을 때, 누적수익률은:", (((payoff)+1).cumprod().iloc[-2]-1)*100,"%")
    payoff=-1*y_pred*(Close[pairs[num_of_pairs][0]].pct_change().shift(-1).loc[T_ma5_0.index][int(len(T_ma5_0)*criteria):])+\
    y_pred*(Close[pairs[num_of_pairs][1]].pct_change().shift(-1).loc[T_ma5_0.index][int(len(T_ma5_0)*criteria):])
    total.append(((payoff)))
    print("실제 거래 누적 수익률은:", (((payoff)+1).cumprod().iloc[-2]-1)*100,"%")
    print("거래비용은:", (pd.DataFrame(y_pred).replace(-1,1).sum() - \
                     (pd.DataFrame(y_pred).replace(0,np.nan) == pd.DataFrame(y_pred).replace(0,np.nan).shift(1)).sum()).values*0.3,"%" )
    print("Sharpe Ratio:", (((payoff)).mean())/(payoff).std())
    print("="*60, "\n")
    return pd.DataFrame(answer), pd.DataFrame(total)

answer = []
total = []
for i in range(len(final_pairs)):
    train = 'Pair{}Y_train'.format(i+1)
    test = 'Pair{}Y_train'.format(i+1)
    y_pred_ = 'y_pred{}'.format(i+1)
    answer_, total_ = get_ret(train, test, y_pred_, i)
    answer.append(answer_)
    total.append(total_)
#answer1, total1 =get_ret(Pair1Y_train, Pair1Y_test, y_pred1, 0)
#answer2, total2 =get_ret(Pair2Y_train, Pair2Y_test, y_pred2, 1)
#answer3, total3 =get_ret(Pair3Y_train, Pair3Y_test, y_pred3, 2)
#answer4, total4 =get_ret(Pair4Y_train, Pair4Y_test, y_pred4, 3)
#answer5, total5 =get_ret(Pair5Y_train, Pair5Y_test, y_pred5, 4)
































