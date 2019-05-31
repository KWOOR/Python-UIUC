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
mod = sys.modules[__name__]

l=locals()
rolling_window = 130
from_trade = 252
trading_start = '2016-10-24'

formation_start_date = [ Close.index.values[i+1] for i in range(0, len(Close), rolling_window)]
trading_start_date = [Close.index.values[i+1] for i in range(from_trade, len(Close) , rolling_window)]
#pairs = [('LMT','TRV'),('NOC','TRV'),('LMT','MMC'), ('PNC','BBT'),('HON', 'DHR')]

pairs =[('CL', 'D'), ('CLX', 'CMS'),('DUK', 'PG'), ('CLX', 'IDA'), ('D', 'KMB'),
       ('AEE', 'PEG'), ('CPB', 'K'), ('PNW', 'SWX'), ('DTE', 'PNW'), ('CMS', 'PNW'),
       ('ES', 'PNW'), ('CLX', 'ES'), ('CLX', 'WEC'),('DTE', 'SWX'),
       ('ED', 'SWX'),('IDA', 'NJR'), ('ES', 'SWX'), ('CLX', 'NJR'),
       ('CL', 'DUK'), ('CMS', 'ES'), ('CLX', 'PNW'), ('ALE', 'NEE'),
       ('D', 'PG'), ('ES', 'WEC')]
#%%
''' Distance Approach '''
data = Close
trading_data = data.loc[data.index >= trading_start]
#formation_data=data['2013-01-24':'2017-09-13']
formation_data=data[data.index<trading_start]
pairs=np.array(pairs)

#for i in range(len(pred_SVM_result)):
#    setattr(mod, 'y_pred{}'.format(i+1), pred_SVM_result[i][0].values)
#    #6.24%  3,18
#for i in range(len(LinearSVM_result)):
#    setattr(mod, 'y_pred{}'.format(i+1), LinearSVM_result[i][0].values)
    
#for i in range(len(GBC_result)):
#    setattr(mod, 'y_pred{}'.format(i+1), GBC_result[i][0].values)

#for i in range(len(RF_result)):
#    setattr(mod, 'y_pred{}'.format(i+1), RF_result[i][0].values)
#    
for i in range(len(logreg_result)):
    setattr(mod, 'y_pred{}'.format(i+1), logreg_result[i][0].values)
    #6.23%    3.10"
    
#for i in range(len(knn_result)):
#    setattr(mod, 'y_pred{}'.format(i+1), knn_result[i][0].values)

    
#%%

''' DA + ML'''

def plus_trading_signals(first, second, pred, trading_data = trading_data, formation_data = formation_data):
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
    for i in range(len(differences)):
        
        #if there is no trading, OPEN it if the spread is greater than the signal
        #AND the spread is less than the stop-loss of 4-sigma
        #if not, move onto the next day
        if trading == False:
#            if (differences.iloc[i] >= X.mean()+signal or differences.iloc[i] <= X.mean()-signal) and (differences.iloc[i] < X.mean()+4*np.std(X) or differences.iloc[i] > X.mean()-4*np.std(X) ):
            if (differences.iloc[i] >= X.mean()+signal or differences.iloc[i] <= X.mean()-signal) and (differences.iloc[i] < X.mean()+4*np.std(X) or differences.iloc[i] > X.mean()-4*np.std(X) ) and (pred[i] ==1):
                if differences.iloc[i] >= X.mean()+signal:
                    whether = 1
                elif differences.iloc[i] <= X.mean()-signal:
                    whether = -1
                trading = True
                start_date = differences.index.values[i]
                
        #if the trade is already open, we check to see if the spread has crossed OR exceeded the 4-sigma stoploss
        #we close the trade and record the start and end date of the trade
        #we also record the return from the short and long position of the trade
        else:
#            if (differences.iloc[i-1] * differences.iloc[i] < 0) or (i == len(differences)-1) or abs(differences.iloc[i] > 4*np.std(X)):
            if (whether == 1) and (differences.iloc[i] < X.mean() +0.5*np.std(X)) :
                trading = False
                whether = 0
                end_date = differences.index.values[i]
                s_ret = (trading_data[first][start_date] - trading_data[first][end_date])/trading_data[first][start_date]
                l_ret = (trading_data[second][end_date] - trading_data[second][start_date])/trading_data[second][start_date]
                result_dict[start_date] = [first, second, start_date, end_date, s_ret,l_ret]
            
            elif (whether == -1) and (differences.iloc[i] > X.mean()- 0.5*np.std(X)): #or (i == len(differences)-1) or abs(differences.iloc[i] > 4*np.std(X)):
                trading = False
                whether = 0
                end_date = differences.index.values[i]
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



#%%
    
trade_cost=0.005
def plus_build_portfolio(trade_list, trading_data = trading_data, trade_cost=0.005/10):
    #create a index_list of dates
    index_list = trading_data.index.tolist()
    
    #initialize dataframe
    portfolio = pd.DataFrame(index = trading_data.index.values, columns = ['Short','Long','ShortR','LongR','Trading'])
    l = trade_list[1]
    trade_list = trade_list[0]
    
    #for each trade, find the start and end dates, and which stocks to long/short
    for i in range(len(trade_list)):
        start = trade_list['Start'][i]
        end = trade_list['End'][i]
        short = trade_list['Short'][i]
        lon = trade_list['Long'][i]
        di = index_list.index(start)
        di2 = index_list.index(end)
        
        #from the start to end date, add the value of the position from that day for that stock
        #also take away trade cost (for long) or add it for shorts
        for j in range(di2 - di + 1):
            date_index = di + j
            dt = index_list[date_index]
            portfolio['Short'][dt] = (trading_data[short][dt]/trading_data[short][index_list[di]]) + trade_cost
            portfolio['Long'][dt] = trading_data[lon][dt]/trading_data[lon][index_list[di]] - trade_cost
            portfolio['Trading'][dt] = 1
            if j == (di2 - di):
                portfolio['Short'][dt] = portfolio['Short'][dt] + trade_cost
                portfolio['Long'][dt] = portfolio['Long'][dt] - trade_cost

    #fill non-trading days
    portfolio.fillna(value = 0, axis = 0)
      
    #adding columns for returns from the short and long portions of the portfolio
    for j in range(1, len(portfolio)):
        if portfolio.iloc[j-1]['Short'] > 0:
            portfolio.iloc[j]['ShortR'] = -(portfolio.iloc[j]['Short'] - portfolio.iloc[j-1]['Short'])/portfolio.iloc[j-1]['Short']
            portfolio.iloc[j]['LongR'] = (portfolio.iloc[j]['Long'] - portfolio.iloc[j-1]['Long'])/portfolio.iloc[j-1]['Long']
        else:
            portfolio.iloc[j]['ShortR'] = 0
            portfolio.iloc[j]['LongR']= 0
            
    #total return is teh sum of both returns
    portfolio['Total'] = portfolio['ShortR'] + portfolio['LongR']
    portfolio.fillna(0, inplace = True)
    return (portfolio, l)


def plus_analyze_portfolio(pairs):
    i = 0
    df = (plus_build_portfolio(plus_trading_signals(pairs[i][0], pairs[i][1], l['y_pred%d'%(i+1)]))[0])
    trade_count = plus_build_portfolio(plus_trading_signals(pairs[i][0], pairs[i][1],  l['y_pred%d'%(i+1)]))[1]
    for i in range(1, len(pairs)):
        df = df + (plus_build_portfolio(plus_trading_signals(pairs[i][0], pairs[i][1],  l['y_pred%d'%(i+1)])))[0]
        trade_count += plus_build_portfolio(plus_trading_signals(pairs[i][0], pairs[i][1],  l['y_pred%d'%(i+1)]))[1]
    df_short = df['ShortR']/len(pairs)
    df_long = df['LongR']/len(pairs)
    df_final = pd.concat([df_short, df_long], axis=1)
    df_final.columns = ['Short Return','Long Return']
    df_final.index.name = 'Date'
    df_final['Total'] = df_final['Short Return'] + df_final['Long Return']
    df_final.fillna(0, inplace = True)
    arithemtic_daily_mean = np.mean(df_final['Total'])
    annualized_return = (1+arithemtic_daily_mean)**250 - 1
    annualized_std = np.std(df_final['Total'])*np.sqrt(250)
    sharpe_ratio = annualized_return/annualized_std
    return [annualized_return, annualized_std, sharpe_ratio, trade_count]

bindata = pd.DataFrame()
for i in range(len(pairs)):
    empty1=plus_trading_signals(pairs[i][0], pairs[i][1], l['y_pred%d'%(i+1)])[0]
    print("{}번째 Pair 총 거래 횟수는".format(i) , len(empty1), "번 이다.")
    buff1=plus_build_portfolio(plus_trading_signals(pairs[i][0], pairs[i][1], l['y_pred%d'%(i+1)]))[0]
    bindata = pd.concat([bindata , buff1['Total']], axis=1)
    print("{}번째 Pair 누적 수익률은".format(i), ((buff1['Total']+1).cumprod()[-1]-1)*100, "%"  )

binnn = []
for i in range(len(bindata.columns)):
    binnn.append('Pair{}'.format(i+1))
bindata.columns  = binnn

a_r, a_std, sh, tc = plus_analyze_portfolio(pairs)
print("총 포트폴리오의 연간 수익률은" , a_r*100, "%")
print("총 포트폴리오의 연간 변동성은" , a_std*100, "%")
print("총 포트폴리오의 Shapre Ratio는" , sh)
print("총 포트폴리오 거래 횟수는" , tc)

i = 0
df = (plus_build_portfolio(plus_trading_signals(pairs[i][0], pairs[i][1], l['y_pred%d'%(i+1)]))[0])
trade_count = plus_build_portfolio(plus_trading_signals(pairs[i][0], pairs[i][1], l['y_pred%d'%(i+1)]))[1]
for i in range(1, len(pairs)):
    df = df + (plus_build_portfolio(plus_trading_signals(pairs[i][0], pairs[i][1], l['y_pred%d'%(i+1)])))[0]
    trade_count += plus_build_portfolio(plus_trading_signals(pairs[i][0], pairs[i][1],l['y_pred%d'%(i+1)]))[1]
df_short = df['ShortR']/len(pairs)
df_long = df['LongR']/len(pairs)
df_final = pd.concat([df_short, df_long], axis=1)
df_final.columns = ['Short Return','Long Return']
df_final.index.name = 'Date'
df_final['Total'] = df_final['Short Return'] + df_final['Long Return']
df_final.fillna(0, inplace = True)
    
((df_final['Total']+1).cumprod()-1).plot()
    


#%%

idk = (df_final['Total']+1).cumprod()
idk = idk.calc_stats()
print(idk.display())

idk.display_monthly_returns()
idk.plot_histogram(title ='Return histogram',color='DarkGreen',alpha=0.5)
ax = bindata.hist(figsize=(12, 5))   
bindata.plot_corr_heatmap(title ='Return Correlation',cmap='GnBu')

#%%

perf_each = bindata.cumsum()+1
perf_each_result = perf_each.calc_stats()
print(perf_each_result.display())

#%%
ffn.core.calc_stats(perf_each).to_csv(sep=',', path ='C:\\Users\\kur7\\OneDrive\\바탕 화면\\uiuc\\DA\\Daily\\perf_each.csv') #ffn data 저장



