# -*- coding: utf-8 -*-
"""
Created on Fri May  3 17:43:40 2019

@author: kur7
"""

import pandas as pd
import datetime as dt
import numpy as np
import statsmodels.api as sm
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
#import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

def model(excel_data):
    
#    excel_data = 'training_sample.csv'
#    
#    training_size = 0.7
    mlp_learning_rate = 0.001
    lasso_feature_count = 10
    
    training_sample = pd.read_csv(excel_data).iloc[:,1:]
    
    index=[]
    for j in range(len(training_sample)):
        date = str(training_sample['yyyy'].values[j])+'-'+str(training_sample['mm'].values[j])
        index.append(dt.datetime.strptime(date,'%Y-%m'))
    training_sample.index = index
    
    #%% SMB
    training_sample_smb = training_sample.sort_values(by = ['yyyy','mm'])
    #training_sample_smb=dict(list(training_sample_smb.groupby(training_sample['PERMNO'])))
    
    #def make_name(data):
    #    name=dict(zip(range(len(data)),data.keys()))
    #    name_buff=dict(zip(data.keys(),range(len(data))))
    #    return name, name_buff
    #
    #name,_ = make_name(training_sample)
    
    #sort = pd.DataFrame()
    #for i in range(len(name)):
    empty = pd.DataFrame(training_sample_smb[['lag_ME','PERMNO']])
    sort = empty.pivot(columns='PERMNO', values='lag_ME')
    #sort = pd.concat([empty['lag_ME']], axis=1)
    
    #sort_ret = pd.DataFrame()
    #for i in range(len(name)):
    empty = pd.DataFrame(training_sample_smb[['RET','PERMNO']])
    sort_ret = empty.pivot(columns='PERMNO', values='RET')
    
    sort_ret = sort_ret.replace(np.nan,0)
    sort = sort.replace(np.nan,0)
    
    
    B = sort.rank(axis=1,ascending=False, method='dense').values - pd.DataFrame(np.max(sort.rank(axis=1,ascending=False, method='dense'),axis=1)*0.3).values
    B = (sort_ret*(B<0)).sum(axis=1) / (B<0).sum(axis=1)
    
    S = sort.rank(axis=1,ascending=True, method='dense').values - pd.DataFrame(np.max(sort.rank(axis=1,ascending=False, method='dense'),axis=1)*0.7).values
    S= (sort_ret*(S>0)).sum(axis=1)/(S>0).sum(axis=1)
    
    SMB = pd.DataFrame(S-B, columns = ['SMB'])
    
    
    
    training_sample_hml = training_sample_smb
    
    empty = pd.DataFrame(training_sample_hml[['FFbm','PERMNO']])
    sort = empty.pivot(columns='PERMNO', values ='FFbm')
    sort = sort.replace(np.nan,0)
    
    H = sort.rank(axis=1,ascending=False, method='dense').values - pd.DataFrame(np.max(sort.rank(axis=1,ascending=False, method='dense'),axis=1)*0.5).values
    H = (sort_ret*(H<0)).sum(axis=1) / (H<0).sum(axis=1)
    
    L = sort.rank(axis=1,ascending=True, method='dense').values - pd.DataFrame(np.max(sort.rank(axis=1,ascending=False, method='dense'),axis=1)*0.5).values
    L= (sort_ret*(L>0)).sum(axis=1)/(L>0).sum(axis=1)
    
    HML = pd.DataFrame(H-L, columns=['HML'])
    
    #%%Mom
    training_sample_Mom = training_sample_smb
    
    empty = pd.DataFrame(training_sample_Mom[['FF_Momentum','PERMNO']])
    sort = empty.pivot(columns='PERMNO', values ='FF_Momentum')
    sort = sort.replace(np.nan,0)
    
    M = sort.rank(axis=1,ascending=False, method='dense').values - pd.DataFrame(np.max(sort.rank(axis=1,ascending=False, method='dense'),axis=1)*0.5).values
    M = (sort_ret*(H<0)).sum(axis=1) / (M<0).sum(axis=1)
    
    m = sort.rank(axis=1,ascending=True, method='dense').values - pd.DataFrame(np.max(sort.rank(axis=1,ascending=False, method='dense'),axis=1)*0.5).values
    m= (sort_ret*(m>0)).sum(axis=1)/(m>0).sum(axis=1)
    
    Mom = pd.DataFrame(M-m, columns=['Mom'])


    from sklearn.preprocessing import Imputer
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputer = imputer.fit(training_sample.loc[:,'PRC':'Amq'])
    training_sample.loc[:,'PRC':'Amq'] = imputer.transform(training_sample.loc[:,'PRC':'Amq'])
    training_sample = pd.merge(training_sample, SMB, left_index=True, right_index=True)
    training_sample = pd.merge(training_sample, HML, left_index=True, right_index=True)
    training_sample = pd.merge(training_sample, Mom, left_index=True, right_index=True)
    ##################################################################################
    # Cross sectional Analysis
#    all_df_date = dict(list(training_sample.groupby(training_sample.index)))
#    for k,i in zip(range(len(SMB)), list(SMB.index)):
#        all_df_date[i]['SMB'] = float(SMB.iloc[k])
    
    # Time Series Analysis
    all_df_date = dict(list(training_sample.groupby(training_sample['PERMNO'])))
#    for i in all_df_date.keys():
##        i = 10923
#        j = all_df_date[i].index.intersection(SMB.index)
#        all_df_date[i]['SMB'] = SMB.loc[j]
    ##################################################################################
    
    #training_sample = dict(list(training_sample.groupby(training_sample['PERMNO'])))
    
    def lasso_picking_feature(data):
#        data = all_df_date[list(all_df_date.keys())[0]]
        buff = data.dropna()
        index = list(buff.loc[:,'PRC':'Amq'].columns)
        index.append('SMB')
        index.append('HML')
        index.append('Mom')
        X= buff[index].values
        ones = np.ones(len(X))
        X = sm.add_constant(np.column_stack((X, ones)))
        Y = buff['RET'].values
        model = sm.OLS(Y, X)
        result = model.fit_regularized(alpha=0.000001, L1_wt=1)
        index.append('const')
        buff = pd.concat([pd.DataFrame(index, columns=['index']),pd.DataFrame(result.params, columns = ['params'])], axis=1)
        features = buff[buff['params']!=0]['index'].values
        return features
    
    def MinMaxScale(data):
        numerator = data - np.min(data, 0)
        denominator = np.max(data, 0) - np.min(data, 0)
        scale_data = numerator / (denominator + 1e-7)
        return scale_data
    
    final_result = pd.DataFrame()
    # Run Part
    for key in tqdm(all_df_date.keys()):
        
        # Lasso feature selection
        buff = all_df_date[key].dropna()
        if len(buff) == 0: continue
        df_new0_feature = lasso_picking_feature(all_df_date[key])
        
        X = all_df_date[key][df_new0_feature[:lasso_feature_count]]
        Y = all_df_date[key]['RET'].fillna(0)
        
        Xtrain, Xtest  = X, X
        Ytrain, Ytest = Y, Y
        
        
#        Xtrain = X[:int(len(X)*training_size)]
#        Ytrain = Y[:int(len(X)*training_size)]
#        Xtest = X[int(len(X)*training_size):]
#        Ytest = Y[int(len(X)*training_size):]
        
        # MinMaxScaling
        Xtrain = MinMaxScale(Xtrain)
        Xtest = MinMaxScale(Xtest)
        
        # FF Regressor
        index_FF = ['SMB','FFbm','FF_Momentum']
    #    index.append('SMB')
        X_FF= all_df_date[key][index_FF].values
        Xtrain_FF = X_FF
        Xtest_FF = X_FF
#        Xtrain_FF = X_FF[:int(len(X_FF)*training_size)]
#        Xtest_FF = X_FF[int(len(X_FF)*training_size):]
        model = LinearRegression()
        model.fit(Xtrain_FF, Ytrain)
        y_ff = model.predict(Xtest_FF)
#        print('FF Prediction:', y_ff)
        
        r2_score_ff = r2_score(Ytest,y_ff)
        print('FF R^2:', r2_score_ff)
    #    print('intercept:', model.intercept_)
    #    print('slope:', model.coef_)
    
        print("LASSO:",key,"===>",df_new0_feature[:lasso_feature_count])
        # MLP Regressor
        mlp = MLPRegressor(hidden_layer_sizes=(500,), activation='relu',verbose=0, random_state=0,
                             batch_size='auto',max_iter=1000, solver='adam', learning_rate_init=mlp_learning_rate)
        
        mlp.fit(Xtrain, Ytrain)
        y_pred =mlp.predict(Xtest)
#        print('MLP Prediction:', y_pred)
        
        mse_mlp = mean_squared_error(Ytest,y_pred)
        r2_score_mlp = r2_score(Ytest,y_pred)
        print ("MLP mse == >" , mse_mlp)
        print ("MLP R^2 == >" , r2_score_mlp)
        
        # Decision Tree
        regr_1 = DecisionTreeRegressor(max_depth=2)
        regr_2 = DecisionTreeRegressor(max_depth=5)
        regr_1.fit(Xtrain, Ytrain)
        regr_2.fit(Xtrain, Ytrain)
        y_1 = regr_1.predict(Xtest)
        y_2 = regr_2.predict(Xtest)
#        print('DT (max_depth=2) Prediction:', y_1)
#        print('DT (max_depth=5) Prediction:', y_2)
        
        mse_dt2 = mean_squared_error(Ytest,y_1)
        r2_score_dt2 = r2_score(Ytest,y_1)
        print ("DT (max_depth=2) mse == >" , mse_dt2)
        print ("DT (max_depth=2) R^2 == >" , r2_score_dt2)
        
        mse_dt5 = mean_squared_error(Ytest,y_2)
        r2_score_dt5 = r2_score(Ytest,y_2)
        print ("DT (max_depth=5) mse == >" , mse_dt5)
        print ("DT (max_depth=5) R^2 == >" , r2_score_dt5)
        
        # SVM
        clf = svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
                        gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
                        tol=0.001, verbose=False)
        clf.fit(Xtrain, Ytrain) 
        svm_result = clf.predict(Xtest)
#        print('SVM Prediction:', svm_result)
        
        mse_svm = mean_squared_error(Ytest,svm_result)
        r2_score_svm = r2_score(Ytest,svm_result)
        print ("SVM mse == >" , mse_svm)
        print ("SVM R^2 == >" , r2_score_svm)
        print ("=" * 60)
        
        
        final = pd.DataFrame({"PERMNO":key
                            ,"MLP mse":[mse_mlp]
                            , "MLP R^2":[r2_score_mlp]
                            , "DT (max_depth=2) mse":[mse_dt2]
                            , "DT (max_depth=2) R^2":[r2_score_dt2]
                            , "DT (max_depth=5) mse":[mse_dt5]
                            , "DT (max_depth=5) R^2":[r2_score_dt5]
                            , "SVM mse" : [mse_svm]
                            , "SVM R^2" : [r2_score_svm]
                            , "FF R^2" : [r2_score_ff]
                            , "MLP Prediction":[y_pred]
                            , "DT (max_depth=2) prediction": [y_1]
                            , "DT (max_depth=5) prediction": [y_2]
                            , "SVM Prediction": [svm_result]
                            }) 
        final_result = pd.concat([final_result,final])
    
    final_result.to_csv("final_result.csv")

#%%
# RUN MODEL
model('training_sample.csv')