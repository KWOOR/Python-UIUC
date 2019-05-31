# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:41:31 2019

@author: kur7
"""
#%pylab inline
import matplotlib.pyplot as plt
#%pylab automatic
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
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix,roc_curve, auc,accuracy_score
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
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

#%%
    

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
     
final_pairs = [('AEE', 'PEG'), ('AEP', 'ALE'), ('ALE', 'NEE'), ('AVA', 'IDA'), ('AVA', 'NI'),
               ('D', 'DUK'), ('DTE', 'DUK'), ('DTE', 'IDA'),
 ('DTE', 'NI'), ('DTE', 'PNW'), ('DTE', 'SWX'), ('DUK', 'NI'), ('DUK', 'SWX'), ('ED', 'PNW'), ('ED', 'SWX'), ('IDA', 'NI'),
 ('IDA', 'PNW'), ('IDA', 'SWX'), ('NI', 'PNW'), ('NI', 'SWX'), ('PNW', 'SWX'), ('PNW', 'WEC'), ('SWX', 'WEC'), ('CL', 'KMB'),
 ('CLX', 'JNJ'), ('CLX', 'MKC'), ('CPB', 'K')]

pairs = final_pairs

criteria=0.8 #80%가 트레인셋, 20%가 테스트셋

         #%%
         
         
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
        
#        c = golden_selection(Close[pairs[i][0]], Close[pairs[i][1]])
#        X_t = np.log(data1[pairs[i][0]]) - c*np.log(data1[pairs[i][1]])
#        X_t = MinMaxScale(X_t)
#        resid = np.log(data1[pairs[i][0]]) - c*np.log(data1[pairs[i][1]])
#        resid = MinMaxScale(resid)
#        resid_1 = resid.shift(1).dropna()
#        resid = resid.iloc[1:]
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
                
                
#        if absvalue==-1:  #방향성 맞추기
#            if status=='T_price_{}':
#                setattr(mod, status.format(i), pd.DataFrame( abs ( (X- mu) /sigma )).iloc[1:] ), 
#                setattr(mod, 'label_{}'.format(i), pd.DataFrame((abs(X.shift(-1)) <  abs(X))*2-1).iloc[1:] )
#            else:
#                setattr(mod, status.format(i), pd.DataFrame( abs ( (X - mu) /sigma )) )
#                
        
                
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

        elif absvalue == 7:
            sig = X.std()
            avg = X.mean()
            if status =='T_price_{}':
                setattr(mod, status.format(i), pd.DataFrame(  (X_t - mu) /sigma ).iloc[1:] )
                buy_open = pd.DataFrame(  (X >= avg+ 1.5*sig)*-1  )
                sell_open = pd.DataFrame(  (X <= avg-1.5*sig)*1  )
                for j in range(100):
                    buy_open+=((buy_open+sell_open)==0)*(buy_open.shift(1)==-1) * (pd.DataFrame(  X >= avg +0.5*sig))*-1
                    sell_open+=((buy_open+sell_open)==0)*(sell_open.shift(1)==1) * (pd.DataFrame( X <=avg-0.5*sig))*1  

                setattr(mod, 'label_{}'.format(i), pd.DataFrame(buy_open+sell_open,dtype='i').iloc[1:] )
            else:
                setattr(mod, status.format(i), pd.DataFrame( (X_t - mu) /sigma ) )
            
        
        
        else: #거래 Signal 맞추기 
            if status=='T_price_{}':
               df = pd.DataFrame({'y':Y,'x':X})
               state_means = KalmanFilterRegression(KalmanFilterAverage(X),KalmanFilterAverage(Y))    
               df['hr'] = state_means[:,0]     
               df['intercept'] = state_means[:,1]            
               df['spread'] = df.y - (df.x * df.hr)- df.intercept
               meanSpread = df.spread.mean()
               stdSpread = df.spread.std()
               df['zScore'] = ((df.spread-meanSpread)/stdSpread).rolling(window=10).mean()
               l_entryZscore = df.zScore.quantile(0.3)
               s_entryZscore = df.zScore.quantile(0.7)
               l_exitZscore = 0
               s_exitZscore = 0
               
               #set up num units long
               df['long entry'] = ((df.zScore < l_entryZscore) & ( df.zScore.shift(1) > l_entryZscore))
               df['long exit'] = ((df.zScore > l_exitZscore) & (df.zScore.shift(1) < l_exitZscore) )
               df['num units long'] = np.nan 
               df.loc[df['long entry'],'num units long'] = 1 
               df.loc[df['long exit'],'num units long'] = 0 
               df['num units long'][0] = 0 
               df['num units long'] = df['num units long'].fillna(method='pad')
               
               #set up num units short 
               df['short entry'] = ((df.zScore > s_entryZscore) & ( df.zScore.shift(1) < s_entryZscore))
               df['short exit'] = ((df.zScore < s_exitZscore) & (df.zScore.shift(1) > s_exitZscore))
               df.loc[df['short entry'],'num units short'] = -1
               df.loc[df['short exit'],'num units short'] = 0
               df['num units short'][0] = 0
               df['num units short'] = df['num units short'].fillna(method='pad')
               
               df['numUnits'] = df['num units long'] + df['num units short']
                               
#               setattr(mod, status.format(i), df['zScore'].iloc[1:] ) 
               signal = df['numUnits']
               

               setattr(mod, 'label_{}'.format(i), signal.iloc[1:] )
               
            elif status == 'spread{}':
                 setattr(mod, status.format(i), pd.DataFrame( X_t ) )
            
            else:
               zScore = (X_t-mu)/sigma

               setattr(mod, status.format(i), pd.DataFrame( zScore ))
                
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
    
#trading_start = '2016-10-20'
trading_start = '2016-10-24'
cost = 0.005


l=locals()
for i in range(len(pairs)):
    setattr(mod, 'Pair{}'.format(i+1), pd.concat([l['T_price_%d'%i],l['T_ma5_%d'%i],
            l['T_ma10_%d'%i],l['T_ema_%d'%i], l['T_ema26_%d'%i], l['T_ema12_%d'%i]
            ,l['T_sd10_%d'%i],l['T_bb_up_%d'%i],l['T_bb_low_%d'%i],l['label_%d'%i]], axis=1, 
        join='inner').values)        
    
    setattr(mod, 'Pair{}X_train'.format(i+1), l['Pair%d'%(i+1)][:int(len(l['Pair%d'%(i+1)])*criteria),:-1])
    setattr(mod, 'Pair{}X_test'.format(i+1), l['Pair%d'%(i+1)][int(len(l['Pair%d'%(i+1)])*criteria):,:-1])
    setattr(mod, 'Pair{}Y_train'.format(i+1), l['Pair%d'%(i+1)][:int(len(l['Pair%d'%(i+1)])*criteria),-1])
    setattr(mod, 'Pair{}Y_test'.format(i+1), l['Pair%d'%(i+1)][int(len(l['Pair%d'%(i+1)])*criteria):,-1])

#l=locals()
#for i in range(len(pairs)):
#    setattr(mod, 'Pair{}'.format(i+1), pd.concat([l['T_price_%d'%i],l['T_ma5_%d'%i],
#           l['T_ema_%d'%i],
#            l['T_sd10_%d'%i],l['T_bb_up_%d'%i],l['T_bb_low_%d'%i],l['label_%d'%i]], axis=1, 
#        join='inner').values)        
#    
#    setattr(mod, 'Pair{}X_train'.format(i+1), l['Pair%d'%(i+1)][:int(len(l['Pair%d'%(i+1)])*criteria),:-1])
#    setattr(mod, 'Pair{}X_test'.format(i+1), l['Pair%d'%(i+1)][int(len(l['Pair%d'%(i+1)])*criteria):,:-1])
#    setattr(mod, 'Pair{}Y_train'.format(i+1), l['Pair%d'%(i+1)][:int(len(l['Pair%d'%(i+1)])*criteria),-1])
#    setattr(mod, 'Pair{}Y_test'.format(i+1), l['Pair%d'%(i+1)][int(len(l['Pair%d'%(i+1)])*criteria):,-1])



#%%
    

def Train_SVM(Xtrain,Xtest,Ytrain,Ytest, C, gamma,decision_function,kernel):
    
    clf = svm.SVC(C=C,gamma=gamma, decision_function_shape = decision_function,kernel = kernel)
    
    clf.fit(Xtrain,Ytrain)
    
    y_pred = clf.predict(Xtest)
    
    pred_df = pd.DataFrame(y_pred)
    
    conf_matrix=confusion_matrix(Ytest, y_pred)
    accuracy = accuracy_score(Ytest, y_pred)*100
      
    print("Accuracy: {}%".format(np.round(metrics.accuracy_score(Ytest, y_pred),2)*100)) 
#    print(conf_matrix)
#    print(classification_report(Ytest, y_pred))
  
    return pred_df, conf_matrix, accuracy , C, gamma

def Train_LinearSVM(Xtrain,Xtest,Ytrain,Ytest, C,multi_class,tol,max_iter):
    
    clf = svm.LinearSVC(C=C,multi_class=multi_class,tol=tol,max_iter=max_iter)
    
    clf.fit(Xtrain,Ytrain)
    
    y_pred = clf.predict(Xtest)
    
    pred_df = pd.DataFrame(y_pred)
    
    conf_matrix=confusion_matrix(Ytest, y_pred) 
    accuracy = accuracy_score(Ytest, y_pred)*100

    print("Accuracy: {}%".format(np.round(metrics.accuracy_score(Ytest, y_pred),2)*100))
#    print(conf_matrix)
#    print(classification_report(Ytest, y_pred))

    return pred_df, conf_matrix, accuracy ,C

def Train_GradientBoostingCalssifier(Xtrain,Xtest,Ytrain,Ytest,learning_rate,n_estimators):
    
    model = GradientBoostingClassifier(learning_rate=learning_rate,n_estimators=n_estimators)

    model.fit(Xtrain, Ytrain)
    
    y_pred = model.predict(Xtest)
    
    pred_df = pd.DataFrame(y_pred)
    
    conf_matrix=confusion_matrix(Ytest, y_pred)
    
    accuracy = accuracy_score(Ytest,y_pred)*100
    
    print("Accuracy: {}%".format(accuracy))
#    print(conf_matrix)
#    print(classification_report(Ytest, y_pred))
    
    return pred_df, conf_matrix, accuracy

def Train_RandomForestClassifier(Xtrain,Xtest,Ytrain,Ytest,n_estimators,criterion):
    
    model = RandomForestClassifier(n_estimators=n_estimators,criterion=criterion)

    model.fit(Xtrain, Ytrain)
    
    y_pred = model.predict(Xtest)
    
    pred_df = pd.DataFrame(y_pred)
    
    conf_matrix=confusion_matrix(Ytest, y_pred)

    accuracy = accuracy_score(Ytest,y_pred)*100
    
    print("Accuracy: {}%".format(accuracy))
#    print(conf_matrix)
#    print(classification_report(Ytest, y_pred))
   
    return pred_df, conf_matrix, accuracy


def cnn_model_fn(features, labels, mode):
        """Model function for CNN."""
        # Input Layer
        input_layer = tf.reshape(features["x"], [-1, 3, 3, 1])
    
        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
          inputs=input_layer,
          filters=32,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)
    
        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    
        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=64,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[1, 1], strides=2)
    
        # Dense Layer
        pool2_flat = tf.reshape(pool2, [-1, 1 * 1 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
          inputs=dense, rate=0.8, training=mode == tf.estimator.ModeKeys.TRAIN)
    
        # Logits Layer
        logits = tf.layers.dense(inputs=dropout, units=3)
    
        predictions = {
          # Generate predictions (for PREDICT and EVAL mode)
          "classes": tf.argmax(input=logits, axis=1),
          # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
          # `logging_hook`.
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
    
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(
                                    loss=loss,
                                    global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
                      "accuracy": tf.metrics.accuracy(
                      labels=labels, predictions=predictions["classes"])
                      }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    
def Train_CNN(Xtrain,Xtest,Ytrain,Ytest):
    Xtrain = MinMaxScale(Xtrain)
    Xtest = MinMaxScale(Xtest)
    Ytrain = Ytrain.astype(np.int32)
    Ytest = Ytest.astype(np.int32)

    for n, i in enumerate(Ytrain):
        if i == -1:
            Ytrain[n] = 0
        elif i== 0:
            Ytrain[n] = 1
        else:
            Ytrain[n] = 2
    for n, i in enumerate(Ytest):
        if i == -1:
            Ytest[n] = 0
        elif i== 0:
            Ytest[n] = 1
        else:
            Ytest[n] = 2
    
    cnn_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/cnn_convnet_model")
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
                                                    x={"x": Xtrain.reshape(-1,3,3)},
                                                    y=Ytrain,
                                                    num_epochs=None,
                                                    shuffle=True)

    # train one step and display the probabilties
    cnn_classifier.train(input_fn=train_input_fn,steps=1000)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                                                    x={"x": Xtest.reshape(-1,3,3)},
                                                    y=Ytest,
                                                    num_epochs=1,
                                                    shuffle=False)
    
    eval_results = cnn_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    return eval_results

def logreg(Xtrain,Xtest,Ytrain,Ytest):
    logreg = LogisticRegression()
    logreg.fit(Xtrain, Ytrain.reshape(-1,1))   
    
    logreg_y_pred = logreg.predict(Xtest)
    
    logreg_pred_df = pd.DataFrame(logreg_y_pred)
    
    logreg_conf_matrix=confusion_matrix(Ytest.reshape(-1,1), logreg_y_pred)
    
    logreg_accuracy = accuracy_score(Ytest.reshape(-1,1),logreg_y_pred)*100
    
    return logreg_pred_df,logreg_conf_matrix,logreg_accuracy

def knn(Xtrain,Xtest,Ytrain,Ytest):
    knn = KNeighborsClassifier()
    knn.fit(Xtrain, Ytrain.reshape(-1,1))
    
    knn_y_pred = knn.predict(Xtest)
    
    knn_pred_df = pd.DataFrame(knn_y_pred)
    
    knn_conf_matrix=confusion_matrix(Ytest.reshape(-1,1), knn_y_pred)
    
    knn_accuracy = accuracy_score(Ytest.reshape(-1,1),knn_y_pred)*100
    
    return knn_pred_df,knn_conf_matrix,knn_accuracy

#%%
# Train result
# Hyper parameters for SVM & LinearSVM
c = 1
gamma = 0.1
decision_function ='ovo'
kernel = 'rbf'
multi_class = 'ovr'
tol = 0.0001
max_iter = 10000*5
#
# Hyper parameters for GradientBoostingClassifier
learning_rate_gb = 1
n_estimators_GB = 20

# Hyper parameters for Randomforest
n_estimators_RF = 30
criterion = 'entropy'

final_pairs = pairs
#
logreg_result = []
for i in range(len(final_pairs)):
   train = 'Pair{}X_train'.format(i+1)
   test = 'Pair{}X_test'.format(i+1)
   y_train = 'Pair{}Y_train'.format(i+1)
   y_test = 'Pair{}Y_test'.format(i+1)
   exec("logreg_result.append(logreg(%s,%s,%s,%s))" % (train,test,y_train,y_test))

knn_result = []
for i in range(len(final_pairs)):
   train = 'Pair{}X_train'.format(i+1)
   test = 'Pair{}X_test'.format(i+1)
   y_train = 'Pair{}Y_train'.format(i+1)
   y_test = 'Pair{}Y_test'.format(i+1)
   exec("knn_result.append(logreg(%s,%s,%s,%s))" % (train,test,y_train,y_test))


pred_SVM_result = []
for i in range(len(pairs)):
    train = 'Pair{}X_train'.format(i+1)
    test = 'Pair{}X_test'.format(i+1)
    y_train = 'Pair{}Y_train'.format(i+1)
    y_test = 'Pair{}Y_test'.format(i+1)
    exec("pred_SVM_result.append(Train_SVM(%s,%s,%s,%s,c,gamma,decision_function,kernel))" % (train,test,y_train,y_test))
    

LinearSVM_result = []
for i in range(len(pairs)):
    train = 'Pair{}X_train'.format(i+1)
    test = 'Pair{}X_test'.format(i+1)
    y_train = 'Pair{}Y_train'.format(i+1)
    y_test = 'Pair{}Y_test'.format(i+1) 
    exec("LinearSVM_result.append(Train_LinearSVM(%s,%s,%s,%s,c,multi_class,tol,max_iter))" % (train,test,y_train,y_test))

GBC_result = []
for i in range(len(pairs)):
    train = 'Pair{}X_train'.format(i+1)
    test = 'Pair{}X_test'.format(i+1)
    y_train = 'Pair{}Y_train'.format(i+1)
    y_test = 'Pair{}Y_test'.format(i+1)
    exec("GBC_result.append(Train_GradientBoostingCalssifier(%s,%s,%s,%s,learning_rate_gb,n_estimators_GB))" % (train,test,y_train,y_test))


RF_result = []
for i in range(len(pairs)):
    train = 'Pair{}X_train'.format(i+1)
    test = 'Pair{}X_test'.format(i+1)
    y_train = 'Pair{}Y_train'.format(i+1)
    y_test = 'Pair{}Y_test'.format(i+1)
    exec("RF_result.append(Train_RandomForestClassifier(%s,%s,%s,%s,n_estimators_RF,criterion))" % (train,test,y_train,y_test))
   

#CNN_result = []
#for i in range(len(pairs)):
#    train = 'Pair{}X_train'.format(i+1)
#    test = 'Pair{}X_test'.format(i+1)
#    y_train = 'Pair{}Y_train'.format(i+1)
#    y_test = 'Pair{}Y_test'.format(i+1)
#    exec("RF_result.append(Train_RandomForestClassifier(%s,%s,%s,%s,n_estimators_RF,criterion))" % (train,test,y_train,y_test))
#       
    
#%%
# Hyper parameters for SVM & LinearSVM
c = [10,15,20,30]
gamma = [0.01,0.10,0.25]
decision_function ='ovo'
kernel = ['rbf']
multi_class = 'ovr'
tol = 0.00001
max_iter = 20000
#
pred_SVM_result = []
for i in range(len(final_pairs)):
    train = 'Pair{}X_train'.format(i+1)
    test = 'Pair{}X_test'.format(i+1)
    y_train = 'Pair{}Y_train'.format(i+1)
    y_test = 'Pair{}Y_test'.format(i+1)
    for j in c:
        for k in gamma:
            for l in range(len(kernel)):
                exec("pred_SVM_result.append(Train_SVM(%s,%s,%s,%s,%d,%f,decision_function,kernel[%d]))" % (train,test,y_train,y_test,j,k,l))




#%%
#cost = 0.005
#trading_start = '2016-10-24'

def SSIBA(Y_train, num_of_pairs, trade = True):
    if trade == True:
        open = Open[trading_start:]
        close = Close[trading_start:]
    else:
        open = Open
        close = Close
    result = {}
    buff = {}
    a_list=[]
    b_list=[]
    if Y_train[0] == -1:
        buy_price = (open[pairs[num_of_pairs][1]]*(1+cost))[1]
        buy_stock = pairs[num_of_pairs][1]
        sell_price = (open[pairs[num_of_pairs][0]]*(1-cost))[1]
        sell_stock = pairs[num_of_pairs][0]
        date = open[pairs[num_of_pairs][1]].index.values[1]
        
        result[0] =  [date, buy_stock, sell_stock, buy_price, sell_price]
        a_list.append(sell_price)
        b_list.append(-buy_price)
        
    if Y_train[0] == 1:
        buy_price = (open[pairs[num_of_pairs][0]]*(1+cost))[1]
        buy_stock = pairs[num_of_pairs][0]
        sell_price = (open[pairs[num_of_pairs][1]]*(1-cost))[1]
        sell_stock = pairs[num_of_pairs][1]
        date = open[pairs[num_of_pairs][1]].index.values[1]
        result[0] =  [date, buy_stock, sell_stock, buy_price, sell_price]
        a_list.append(-buy_price)
        b_list.append(sell_price)
    
    difference = Y_train[1:]  - Y_train[:-1]    
    for i in range(len(difference)):
        if i == len(difference)-1:
            if Y_train[i]  == -1:
                buy_price = close[pairs[num_of_pairs][1]][i+1]
                sell_price = close[pairs[num_of_pairs][0]][i+1]
                buy_stock = pairs[num_of_pairs][1]
                sell_stock = pairs[num_of_pairs][0]
                date = open[pairs[num_of_pairs][0]].index.values[i+1]
                result[i+1] = [date, buy_stock, sell_stock, buy_price, sell_price]
                a_list.append(sell_price)
                b_list.append(-buy_price)
                
            elif Y_train[i] == 1:
                buy_price = close[pairs[num_of_pairs][0]][i+1]
                sell_price = close[pairs[num_of_pairs][1]][i+1]
                buy_stock = pairs[num_of_pairs][0]
                sell_stock = pairs[num_of_pairs][1]
                date = open[pairs[num_of_pairs][0]].index.values[i+1]
                result[i+1] = [date, buy_stock, sell_stock, buy_price, sell_price]
                a_list.append(-buy_price)
                b_list.append(sell_price)         
                
        else:
            if difference[i] == -1:
                buy_price = (open[pairs[num_of_pairs][1]]*(1+cost))[i+2]
                sell_price = (open[pairs[num_of_pairs][0]]*(1-cost))[i+2]
                buy_stock = pairs[num_of_pairs][1]
                sell_stock = pairs[num_of_pairs][0]
                date = open[pairs[num_of_pairs][0]].index.values[i+2]
                result[i+1] = [date, buy_stock, sell_stock, buy_price, sell_price]
                a_list.append(sell_price)
                b_list.append(-buy_price)
                
            elif difference[i] == 2:
#                buy_price = 2*(open[pairs[num_of_pairs][0]]*(1+cost))[i+2]
#                sell_price = 2*(open[pairs[num_of_pairs][1]]*(1-cost))[i+2]
                
                buy_price = (open[pairs[num_of_pairs][0]]*(1+cost))[i+2]
                sell_price = (open[pairs[num_of_pairs][1]]*(1-cost))[i+2]
                buy_stock = pairs[num_of_pairs][0]
                sell_stock = pairs[num_of_pairs][1]
                date = open[pairs[num_of_pairs][0]].index.values[i+2]
                result[i+1] = [date, buy_stock, sell_stock, buy_price, sell_price]
                buff[i+1] = [date, buy_stock, sell_stock, buy_price, sell_price]
                a_list.append(-buy_price)
                b_list.append(sell_price)
                
            elif difference[i] == 1:
                buy_price = (open[pairs[num_of_pairs][0]]*(1+cost))[i+2]
                sell_price = (open[pairs[num_of_pairs][1]]*(1-cost))[i+2]
                buy_stock = pairs[num_of_pairs][0]
                sell_stock = pairs[num_of_pairs][1]
                date = open[pairs[num_of_pairs][0]].index.values[i+2]
                result[i+1] = [date, buy_stock, sell_stock, buy_price, sell_price]
                a_list.append(-buy_price)
                b_list.append(sell_price)
            elif difference[i] == -2:
#                buy_price = 2*(open[pairs[num_of_pairs][1]]*(1+cost))[i+2]
#                sell_price = 2*(open[pairs[num_of_pairs][0]]*(1-cost))[i+2]
                
                buy_price = (open[pairs[num_of_pairs][1]]*(1+cost))[i+2]
                sell_price = (open[pairs[num_of_pairs][0]]*(1-cost))[i+2]
                buy_stock = pairs[num_of_pairs][1]
                sell_stock = pairs[num_of_pairs][0]
                date = open[pairs[num_of_pairs][0]].index.values[i+2]
                result[i+1] = [date, buy_stock, sell_stock, buy_price, sell_price]
                buff[i+1] = [date, buy_stock, sell_stock, buy_price, sell_price]
                a_list.append(sell_price)
                b_list.append(-buy_price)
    buff = pd.DataFrame.from_dict(buff, orient = 'index', columns = ['Date','BuyStock', 'SellStock', 'BuyPrice', 'SellPrice']  )
    return buff, pd.DataFrame.from_dict(result, orient = 'index', 
                                  columns = ['Date','BuyStock', 'SellStock', 'BuyPrice', 'SellPrice']), a_list, b_list



#%%

csv_name = ['Pair1_pred.csv','Pair2_pred.csv','Pair3_pred.csv','Pair4_pred.csv','Pair5_pred.csv','Pair6_pred.csv',
            'Pair7_pred.csv','Pair8_pred.csv','Pair9_pred.csv','Pair10_pred.csv','Pair11_pred.csv','Pair12_pred.csv',
            'Pair13_pred.csv','Pair14_pred.csv','Pair15_pred.csv','Pair16_pred.csv','Pair17_pred.csv']


#
#csv_name = ['Pair1Y_train.csv','Pair2Y_train.csv','Pair3Y_train.csv','Pair4Y_train.csv','Pair5Y_train.csv','Pair6Y_train.csv',
#            'Pair7Y_train.csv','Pair8Y_train.csv','Pair9Y_train.csv','Pair10Y_train.csv','Pair11Y_train.csv','Pair12Y_train.csv',
#            'Pair13Y_train.csv','Pair14Y_train.csv','Pair15Y_train.csv','Pair16Y_train.csv','Pair17Y_train.csv']
#


def calculate_ret(data, num, trade = True):
    buff,result, a, b = SSIBA(data, num, trade)    
    empty = pd.concat([result, buff], axis=0)
    empty = empty.sort_index()
    
    returnnnn = []
    if len(empty) != 0:
        for i in range(0,int(len(empty)/2), 2):
            vv = empty.iloc[i:i+2,-2:]
            bret = (vv.iloc[1,1] - vv.iloc[0,0]) / vv.iloc[0,0]
            sret = -(vv.iloc[1,0] - vv.iloc[0,1]) / vv.iloc[0,1]
            returnnnn.append((bret+sret)/2)
    
        print((pd.DataFrame(returnnnn)+1).cumprod().iloc[-1,:])
    else:
        print("NO TRADE")
    print('='*20)
    return returnnnn

#%%
print("="*5, 'SVM', "="*5)
for i in range(len(pred_SVM_result)):
    setattr(mod, 'y_pred{}'.format(i+1), pred_SVM_result[i][0].values)

l=locals()
for i in range(len(pred_SVM_result)):
    aaaa = calculate_ret(l['y_pred%d'%(i+1)], i)
    
print("="*5, 'LSVM', "="*5)
for i in range(len(LinearSVM_result)):
    setattr(mod, 'y_pred{}'.format(i+1), LinearSVM_result[i][0].values)

l=locals()
for i in range(len(LinearSVM_result)):
    aaaa = calculate_ret(l['y_pred%d'%(i+1)], i)

print("="*5, 'GBC', "="*5)

for i in range(len(GBC_result)):
    setattr(mod, 'y_pred{}'.format(i+1), GBC_result[i][0].values)

l=locals()
for i in range(len(GBC_result)):
    aaaa = calculate_ret(l['y_pred%d'%(i+1)], i)
  
    
print("="*5, 'RF', "="*5)

for i in range(len(RF_result)):
    setattr(mod, 'y_pred{}'.format(i+1), RF_result[i][0].values)

l=locals()
for i in range(len(RF_result)):
    aaaa = calculate_ret(l['y_pred%d'%(i+1)], i)




#%%

aaaa = calculate_ret(Pair1Y_train, 0, False)
aaaa = calculate_ret(Pair2Y_train, 1, False)
aaaa = calculate_ret(Pair15Y_train, 14, False)



aaaa = calculate_ret(Pair1Y_test, 0)
aaaa = calculate_ret(Pair5Y_test, 4)

    
    
    
#%%
    
for k in csv_name:
    number1 = pd.read_csv(k).iloc[:,1:]
    column_name = number1.columns
    for j in column_name:
        buff,result, a, b = SSIBA(number1[j].values, csv_name.index(k))    
        empty = pd.concat([result, buff], axis=0)
        empty = empty.sort_index()
        
        returnnnn = []
        for i in range(0,int(len(empty)/2), 2):
            vv = empty.iloc[i:i+2,-2:]
            bret = (vv.iloc[1,1] - vv.iloc[0,0]) / vv.iloc[0,0]
            sret = -(vv.iloc[1,0] - vv.iloc[0,1]) / vv.iloc[0,1]
            returnnnn.append((bret+sret)/2)
        
        print(k,j,(pd.DataFrame(returnnnn)+1).cumprod().iloc[-1,:])
        print('='*20)



