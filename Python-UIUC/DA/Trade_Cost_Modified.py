# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 03:53:16 2019

@author: kur7
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 02:03:16 2019

@author: kur7
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 20:29:49 2019

@author: kur7
"""
import datetime
import sys
from datetime import datetime, timedelta
mod = sys.modules[__name__]

l=locals()
rolling_window = 130
from_trade = 252

value=-1  #-1이면 방향성 맞추기, 0이면 거래 Signal 중에서 PCA ppt에 있던 S-score 방식 사용하기
kal=False #False면 Kalman Filter 안 씀 
trading_start = '2016-10-24'

c = 1
gamma = 0.1
decision_function ='ovo'
kernel = 'rbf'
multi_class = 'ovr'
tol = 0.0001
max_iter = 10000

formation_start_date = [ Close.index.values[i+1] for i in range(0, len(Close), rolling_window)]
trading_start_date = [Close.index.values[i+1] for i in range(from_trade, len(Close) , rolling_window)]
trading_end_date = [trading_start_date[i+1]-1 for i in range(len(trading_start_date)-1)]
trading_end_date.append(Close.index.values[-1])
#pairs = [('LMT','TRV'),('NOC','TRV'),('LMT','MMC'), ('PNC','BBT'),('HON', 'DHR')]

#pairs =[('CL', 'D'), ('CLX', 'CMS'),('DUK', 'PG'), ('CLX', 'IDA'), ('D', 'KMB'),
#       ('AEE', 'PEG'), ('CPB', 'K'), ('PNW', 'SWX'), ('DTE', 'PNW'), ('CMS', 'PNW'),
#       ('ES', 'PNW'), ('CLX', 'ES'), ('CLX', 'WEC'),('DTE', 'SWX'),
#       ('ED', 'SWX'),('IDA', 'NJR'), ('ES', 'SWX'), ('CLX', 'NJR'),
#       ('CL', 'DUK'), ('CMS', 'ES'), ('CLX', 'PNW'), ('ALE', 'NEE'),
#       ('D', 'PG'), ('ES', 'WEC')]




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

def reg_m(y, x):
    ones = np.ones(len(x))
    X = sm.add_constant(np.column_stack((x, ones)))
    results = sm.OLS(y, X).fit()
    return results

mod = sys.modules[__name__]

#%%         
         
def get_Tscore(data1,data2, status, absvalue=-1, Kalman=False): #data1은 종가, data2는 시가, status는 'T_price_{}' 이런식으로!!
    #status=T_price_{}, T_wma_{}, T_sma_{}, T_rsi_{}, T_mfi_{} string형식으로..
    for k in range(len(pairs[i])):
        d_A = data1[pairs[i][k][0]]-data2[pairs[i][k][0]]
        d_B=  data1[pairs[i][k][1]]-data2[pairs[i][k][1]]
        Y= (d_A/(data2[pairs[i][k][0]]+1e-10)).dropna()
        X= (d_B/(data2[pairs[i][k][1]]+1e-10)).dropna()
        result=reg_m(Y,X)
        X_t=result.resid
        
        resid=result.resid
        resid_1=resid.shift(1).dropna()
        resid=resid.iloc[1:]
        c = golden_selection(Close[pairs[i][k][0]], Close[pairs[i][k][1]])
        X = np.log(data1[pairs[i][k][0]]) - c*np.log(data1[pairs[i][k][1]])
        
        
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
                setattr(mod, status.format(k), pd.DataFrame( abs ( (X_t - mu) /sigma )).iloc[1:] ), 
                setattr(mod, 'label_{}'.format(k), pd.DataFrame((abs(X_t.shift(-1)) <  abs(X_t))*2-1).iloc[1:] )
            else:
                setattr(mod, status.format(k), pd.DataFrame( abs ( (X_t - mu) /sigma )) )

#%%

def plus_trading_signals(first, second, pred, trading_data , formation_data ):
    #choose 2-sigma as the trading signal
    c = golden_selection(formation_data[first], formation_data[second])
    X = np.log(formation_data[first]) - c*np.log(formation_data[second])
#    KalmanFilterRegression( x,  y)  #x는 페어 앞에꺼 종가, y는 뒤에꺼 종가... mean에다가 결과를 넣어라
    #그 결과를 c에다가 대입
    signal = 1.5* np.std(X)
    result_dict = {}
    
    #there should be no trading initially
    trading = False
    whether = 0
    #create a time series of the spread between the two stocks
    differences = np.log(trading_data[first]) - c*np.log(trading_data[second])
    for p in range(len(differences)):
        #if there is no trading, OPEN it if the spread is greater than the signal
        #AND the spread is less than the stop-loss of 4-sigma
        #if not, move onto the next day
        if trading == False:
#            if (differences.iloc[i] >= X.mean()+signal or differences.iloc[i] <= X.mean()-signal) and (differences.iloc[i] < X.mean()+4*np.std(X) or differences.iloc[i] > X.mean()-4*np.std(X) ):
            if (differences.iloc[p] >= X.mean()+signal or differences.iloc[p] <= X.mean()-signal) and (differences.iloc[p] < X.mean()+4*np.std(X) or differences.iloc[p] > X.mean()-4*np.std(X) ) and (pred[p] ==1):
                if differences.iloc[p] >= X.mean()+signal:
                    whether = 1
                elif differences.iloc[p] <= X.mean()-signal:
                    whether = -1
                trading = True
                start_date = differences.index.values[p]
                
        #if the trade is already open, we check to see if the spread has crossed OR exceeded the 4-sigma stoploss
        #we close the trade and record the start and end date of the trade
        #we also record the return from the short and long position of the trade
        else:
#            if (differences.iloc[i-1] * differences.iloc[i] < 0) or (i == len(differences)-1) or abs(differences.iloc[i] > 4*np.std(X)):
#            if (whether == 1) and (differences.iloc[i] < X.mean() +0.5*np.std(X)) :
            if (whether == 1) and ((differences.iloc[p] < X.mean() +0.5*np.std(X)) or p== (len(differences)-1) or differences.iloc[p] > X.mean()+4*np.std(X)):

                trading = False
                whether = 0
                end_date = differences.index.values[p]
                s_ret = (trading_data[first][start_date] - trading_data[first][end_date])/trading_data[first][start_date]
                l_ret = (trading_data[second][end_date] - trading_data[second][start_date])/trading_data[second][start_date]
                result_dict[start_date] = [first, second, start_date, end_date, s_ret,l_ret]
            
            elif (whether == -1) and ((differences.iloc[p] > X.mean()- 0.5*np.std(X))or p== (len(differences)-1) or differences.iloc[p] <X.mean()-4*np.std(X)): #or (i == len(differences)-1) or abs(differences.iloc[i] > 4*np.std(X)):
                trading = False
                whether = 0
                end_date = differences.index.values[p]
                s_ret = (trading_data[second][start_date] - trading_data[second][end_date])/trading_data[second][start_date]
                l_ret = (trading_data[first][end_date] - trading_data[first][start_date])/trading_data[first][start_date]
                result_dict[start_date] = [second, first, start_date, end_date, s_ret,l_ret]
    
    #formatting the final dataframe to be returned
    df = pd.DataFrame.from_dict(result_dict, orient = 'index', columns = ['Short','Long','Start','End', 'SReturn','LReturn'])
    df.index = list(range(len(df)))
    df['Total'] = df['SReturn'] + df['LReturn']
    if len(df['Total']) != 0:
        df['Length'] = (df['End'] - df['Start']).dt.days
    return (df, len(df))

def plus_build_portfolio(trade_list, trading_data , trade_cost=0.005):
    #create a index_list of dates
    index_list = trading_data.index.tolist()
    
    #initialize dataframe
    portfolio = pd.DataFrame(index = trading_data.index.values, columns = ['Short','Long','ShortR','LongR','Trading'])
    l = trade_list[1]
    trade_list = trade_list[0]
    
    #for each trade, find the start and end dates, and which stocks to long/short
    for qq in range(len(trade_list)):
        start = trade_list['Start'][qq]
        end = trade_list['End'][qq]
        short = trade_list['Short'][qq]
        lon = trade_list['Long'][qq]
        di = index_list.index(start)
        di2 = index_list.index(end)
        
        #from the start to end date, add the value of the position from that day for that stock
        #also take away trade cost (for long) or add it for shorts
        for aa in range(di2 - di + 1):
            date_index = di + aa
            dt = index_list[date_index]
            if aa ==0:
                portfolio['Short'][dt] = trading_data[short][dt]/trading_data[short][index_list[di]] - trade_cost
                portfolio['Long'][dt] = trading_data[lon][dt]/trading_data[lon][index_list[di]] + trade_cost
            else:    
                portfolio['Short'][dt] = trading_data[short][dt]/trading_data[short][index_list[di]]
                portfolio['Long'][dt]  = trading_data[lon][dt]/trading_data[lon][index_list[di]]
            portfolio['Trading'][dt] = 1
            if aa == (di2 - di):
                portfolio['Short'][dt] = portfolio['Short'][dt] 
                portfolio['Long'][dt] = portfolio['Long'][dt]
                ''' 위에 거래비용 수정했다!! '''
    #fill non-trading days
    portfolio.fillna(value = 0, axis = 0)
      
    #adding columns for returns from the short and long portions of the portfolio
    for aa in range(1, len(portfolio)):
        if portfolio.iloc[aa-1]['Short'] > 0:
            portfolio.iloc[aa]['ShortR'] = -(portfolio.iloc[aa]['Short'] /portfolio.iloc[aa-1]['Short'])+1
            portfolio.iloc[aa]['LongR'] = portfolio.iloc[aa]['Long']/ portfolio.iloc[aa-1]['Long']-1
        else:
            portfolio.iloc[aa]['ShortR'] = 0
            portfolio.iloc[aa]['LongR']= 0
            
    #total return is teh sum of both returns
    portfolio['Total'] = portfolio['ShortR'] + portfolio['LongR']
    portfolio.fillna(0, inplace = True)
    return (portfolio, l)

#
#def plus_analyze_portfolio(pairs):
#    i = 0
#    df = (plus_build_portfolio(plus_trading_signals(pairs[i][0], pairs[i][1], l['y_pred%d'%(i+1)]))[0])
#    trade_count = plus_build_portfolio(plus_trading_signals(pairs[i][0], pairs[i][1],  l['y_pred%d'%(i+1)]))[1]
#    for i in range(1, len(pairs)):
#        df = df + (plus_build_portfolio(plus_trading_signals(pairs[i][0], pairs[i][1],  l['y_pred%d'%(i+1)])))[0]
#        trade_count += plus_build_portfolio(plus_trading_signals(pairs[i][0], pairs[i][1],  l['y_pred%d'%(i+1)]))[1]
#    df_short = df['ShortR']/len(pairs)
#    df_long = df['LongR']/len(pairs)
#    df_final = pd.concat([df_short, df_long], axis=1)
#    df_final.columns = ['Short Return','Long Return']
#    df_final.index.name = 'Date'
#    df_final['Total'] = df_final['Short Return'] + df_final['Long Return']
#    df_final.fillna(0, inplace = True)
#    arithemtic_daily_mean = np.mean(df_final['Total'])
#    annualized_return = (1+arithemtic_daily_mean)**250 - 1
#    annualized_std = np.std(df_final['Total'])*np.sqrt(250)
#    sharpe_ratio = annualized_return/annualized_std
#    return [annualized_return, annualized_std, sharpe_ratio, trade_count]
#

              
#%%                

roll_pair = pd.read_csv('rolling_pair.csv')
pairs = []
for i in range(int(len(roll_pair)/25)):
    buff = []
    period_pair = roll_pair[roll_pair['period']==i]
    for j in range(len(period_pair)):    
        buff.append( ( period_pair['pairA'].values[j], period_pair['pairB'].values[j]))
    pairs.append(buff)
    
    
    #%%
bindata = pd.DataFrame()
for i in range(len(trading_start_date)):
    Close_b = Close.loc[formation_start_date[i]:trading_end_date[i]]
    Open_b = Open.loc[formation_start_date[i]:trading_end_date[i]]
    ma5_open_b = ma5_open.loc[formation_start_date[i]:trading_end_date[i]]
    ma5_b = ma5.loc[formation_start_date[i]:trading_end_date[i]]
    ma10_open_b = ma10_open.loc[formation_start_date[i]:trading_end_date[i]]
    ema_b = ema.loc[formation_start_date[i]:trading_end_date[i]]
    ma10_b = ma10.loc[formation_start_date[i]:trading_end_date[i]]
    ema_26_b = ema_26.loc[formation_start_date[i]:trading_end_date[i]]
    ema_12_b = ema_12.loc[formation_start_date[i]:trading_end_date[i]]
    ema_26_open_b = ema_26_open.loc[formation_start_date[i]:trading_end_date[i]]
    ema_12_open_b = ema_12_open.loc[formation_start_date[i]:trading_end_date[i]]
    sd_10_open_b = sd_10_open.loc[formation_start_date[i]:trading_end_date[i]]
    sd_10_b = sd_10.loc[formation_start_date[i]:trading_end_date[i]]
    ema_open_b = ema_open.loc[formation_start_date[i]:trading_end_date[i]]
    bb_up_open_b = bb_up_open.loc[formation_start_date[i]:trading_end_date[i]]
    bb_up_b = bb_up.loc[formation_start_date[i]:trading_end_date[i]]
    bb_low_open_b = bb_low_open.loc[formation_start_date[i]:trading_end_date[i]]
    bb_low_b = bb_low.loc[formation_start_date[i]:trading_end_date[i]]
    
    vlaue = -1
    kal = False
    get_Tscore(Close_b, Open_b, 'T_price_{}', value, kal)
    get_Tscore(ma5_b, ma5_open_b, 'T_ma5_{}', value, kal)
    get_Tscore(ma10_b, ma10_open_b, 'T_ma10_{}', value, kal)
    get_Tscore(ema_b, ema_open_b, 'T_ema_{}', value, kal)
    get_Tscore(ema_26_b, ema_26_open_b, 'T_ema26_{}', value, kal)
    get_Tscore(ema_12_b, ema_12_open_b, 'T_ema12_{}', value, kal)
    get_Tscore(sd_10_b, sd_10_open_b, 'T_sd10_{}', value, kal)
    get_Tscore(bb_up_b, bb_up_open_b, 'T_bb_up_{}', value, kal)
    get_Tscore(bb_low_b, bb_low_open_b, 'T_bb_low_{}', value, kal)
    l=locals()
    for j in range(len(pairs[i])):
        setattr(mod, 'Pair{}'.format(j+1), pd.concat([l['T_price_%d'%j],l['T_ma5_%d'%j],
            l['T_ma10_%d'%j],l['T_ema_%d'%j], l['T_ema26_%d'%j], l['T_ema12_%d'%j]
            ,l['T_sd10_%d'%j],l['T_bb_up_%d'%j],l['T_bb_low_%d'%j],l['label_%d'%j]], axis=1, 
        join='inner').values)        
    
        setattr(mod, 'Pair{}X_train'.format(j+1), 
                l['Pair%d'%(j+1)][:int(len(l['Pair%d'%(j+1)])*from_trade/len(Close_b)),:-1])
        setattr(mod, 'Pair{}X_test'.format(j+1), 
                l['Pair%d'%(j+1)][int(len(l['Pair%d'%(j+1)])*from_trade/len(Close_b)):,:-1])
        setattr(mod, 'Pair{}Y_train'.format(j+1), 
                l['Pair%d'%(j+1)][:int(len(l['Pair%d'%(j+1)])*from_trade/len(Close_b)),-1])
        setattr(mod, 'Pair{}Y_test'.format(j+1), 
                l['Pair%d'%(j+1)][int(len(l['Pair%d'%(j+1)])*from_trade/len(Close_b)):,-1])
    
    pred_SVM_result = []
    for j in range(len(pairs[i])):
        train = 'Pair{}X_train'.format(j+1)
        test = 'Pair{}X_test'.format(j+1)
        y_train = 'Pair{}Y_train'.format(j+1)
        y_test = 'Pair{}Y_test'.format(j+1)
        exec("pred_SVM_result.append(Train_SVM(%s,%s,%s,%s,c,gamma,decision_function,kernel))" % (train,test,y_train,y_test))
    for j in range(len(pred_SVM_result)):
        setattr(mod, 'y_pred{}'.format(j+1), pred_SVM_result[j][0].values)
    
    data = Close_b
    trading_data_ = data.loc[data.index >= trading_start_date[i]]
    #formation_data=data['2013-01-24':'2017-09-13']
    formation_data_=data[data.index<trading_start_date[i]].iloc[1:,:]
    for j in range(len(pairs[i])):
        empty1=plus_trading_signals(first = pairs[i][j][0], second= pairs[i][j][1], pred = l['y_pred%d'%(j+1)],
                                    trading_data=trading_data_, formation_data=formation_data_)[0]
        print("{}번째 Pair 총 거래 횟수는".format(j) , len(empty1), "번 이다.")
        buff1=plus_build_portfolio(plus_trading_signals(first = pairs[i][j][0], second= pairs[i][j][1], pred=l['y_pred%d'%(j+1)],
                                                        trading_data=trading_data_, formation_data=formation_data_),trading_data=trading_data_ )[0]
        print("{}번째 Pair 누적 수익률은".format(j), ((buff1['Total']+1).cumprod()[-1]-1)*100, "%"  )
#    a_r, a_std, sh, tc = plus_analyze_portfolio(pairs[i])
#    print("총 포트폴리오의 연간 수익률은" , a_r*100, "%")
#    print("총 포트폴리오의 연간 변동성은" , a_std*100, "%")
#    print("총 포트폴리오의 Shapre Ratio는" , sh)
#    print("총 포트폴리오 거래 횟수는" , tc)

    j = 0
    df = (plus_build_portfolio(plus_trading_signals(pairs[i][j][0], pairs[i][j][1], l['y_pred%d'%(j+1)], trading_data=trading_data_,
                                                    formation_data=formation_data_), trading_data=trading_data_)[0])
    trade_count = plus_build_portfolio(plus_trading_signals(pairs[i][j][0], pairs[i][j][1], l['y_pred%d'%(j+1)], trading_data= trading_data_,
                                                            formation_data=formation_data_), trading_data=trading_data_)[1]
    for j in range(1, len(pairs[i])):
        df = df + (plus_build_portfolio(plus_trading_signals(pairs[i][j][0], pairs[i][j][1], l['y_pred%d'%(j+1)],
                                                             trading_data=trading_data_, formation_data=formation_data_),trading_data=trading_data_))[0]
        trade_count += plus_build_portfolio(plus_trading_signals(pairs[i][j][0], pairs[i][j][1],l['y_pred%d'%(j+1)], trading_data=trading_data_,
                                                                 formation_data=formation_data_),trading_data=trading_data_)[1]
    df_short = df['ShortR']/len(pairs[i])
    df_long = df['LongR']/len(pairs[i])
    df_final = pd.concat([df_short, df_long], axis=1)
    df_final.columns = ['Short Return','Long Return']
    df_final.index.name = 'Date'
    df_final['Total'] = df_final['Short Return'] + df_final['Long Return']
    df_final.fillna(0, inplace = True)
    bindata = pd.concat([bindata, df_final['Total']])

#%%

bindata.to_csv('FINNNNNNNNN.csv')
((bindata+1).cumprod()-1).plot()
bindata['TotalRet'] = ((bindata+1).cumprod())

idk = bindata['TotalRet'].calc_stats()
print(idk.display())

idk.display_monthly_returns()
idk.plot_histogram(title = 'Return histogram', color = 'DarkGreen', alpha=0.5)
ax = bindata['TotalRet'].hist(figsize=(12,5))
bindata.plot_corr_heatmap(title = 'Return Correlation', cmap='GnBu')

ffn.core.calc_stats(bindata['TotalRet']).to_csv(sep=',', path = 'C:\\Users\\kur7\\OneDrive\\바탕 화면\\uiuc\\DA\\Daily\\perf_each.csv')
#%%








