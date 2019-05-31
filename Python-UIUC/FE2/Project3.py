# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:43:16 2019

@author: kur7
"""

import numpy as np
import pandas as pd

sig1=0.2079 #버퍼가 있으니까 95% 내재변동성 
sig2=0.2262 #버퍼가 있으니까 95% 내재변동성
rho=0.96
r=0.0273
K0=[2614.45,1512.155] #SPY, RTY
T= 65/12 #만기
Nt = np.busday_count('2018-04-04','2023-09-06')
avg_date = np.busday_count('2023-06-06','2023-09-06')+1 #6월 6일부터 9월 6일까지의 기간을 포함한 3개월 영업일 수
#np.busdat_count함수가 시작일을 빼고 반환해주는데.. 우리는 시작일도 필요하니까 1을 더함
F=1000
max_payoff = 2116.4

Mt = np.busday_count('2023-09-06','2023-09-11')
issue = np.busday_count('2018-04-04','2018-04-09')
q1 =0.0195 #배당..
q2 =0.0132


#%%

count_simulation=100000
timestep=Nt+1
t=np.linspace(0,T,timestep)
W1=np.random.normal(0,1,(count_simulation,int(timestep)))*np.sqrt(t[1]-t[0])
W2=W1*rho+np.sqrt(1-rho**2)*np.random.normal(0,1,(count_simulation,int(timestep)))*np.sqrt(t[1]-t[0])

W1[:,0]=0
W1=W1.cumsum(axis=1)
W2[:,0]=0
W2=W2.cumsum(axis=1)
stock1=K0[0]*np.exp((r-q1-(sig1**2)/2)*t+sig1*W1)
stock2=K0[1]*np.exp((r-q2-(sig2**2)/2)*t+sig2*W2)

#
#
#def project3_sim(stx0,rty0,r,stx_div,rty_div,stx_sigma,rty_sigma,rho,T,FV,n_steps,n_trials):
#    d_t = T/n_steps
##    avg_start_node = avg_node[0] ; avg_end_node = avg_node[1]
#    z_matrix1 = np.random.standard_normal((n_trials,n_steps))
#    z_matrix2 = np.random.standard_normal((n_trials,n_steps))
#    z_matrix_stx = z_matrix1
#    z_matrix_rty = rho*z_matrix1 + np.sqrt(1 - rho**2)*z_matrix2
#    stx_matrix = np.zeros((n_trials,n_steps))
#    rty_matrix = np.zeros((n_trials,n_steps))
#    stx_matrix[:,0] = stx0
#    rty_matrix[:,0] = rty0
#    for j in range(n_steps-1):
#        stx_matrix[:,j+1] = stx_matrix[:,j]*np.exp((r-stx_div-0.5*stx_sigma**2)*d_t+stx_sigma*np.sqrt(d_t)*z_matrix_stx[:,j])
#        rty_matrix[:,j+1] = rty_matrix[:,j]*np.exp((r-rty_div-0.5*rty_sigma**2)*d_t + rty_sigma*np.sqrt(d_t)*z_matrix_rty[:,j])
#    
#    return stx_matrix, rty_matrix
#
#stock1, stock2 = project3_sim(K0[0], K0[1], r, q1, q2, sig1, sig2, rho, T, F, Nt, count_simulation )





#%%
stock1_avg = np.mean(stock1[:,-avg_date:], axis=1) #avg_date 갯수만큼의 셀을 불러와야하므로 다시 +1을 인덱싱 해줌
stock2_avg = np.mean(stock2[:,-avg_date:], axis=1)

stock1_rate = stock1_avg/K0[0]
stock2_rate = stock2_avg/K0[1]

avg_rate = np.min(pd.concat([pd.Series(stock1_rate), pd.Series(stock2_rate)],axis=1), axis=1)

buffer = avg_rate[avg_rate>=1.21]
num_case1 = len(buffer)
payoff_case1 = F*(buffer - 1.21)*3.34+F+415
payoff_case1 -= (payoff_case1>max_payoff)*(payoff_case1-max_payoff) #최대 페이오프가 2116.4니까 조정해주기
#print(num_case1)
print('계산 잘 됐는지 확인하기!, case1의 최소값은: ',np.min(payoff_case1), '(1415보다 커야함)')
print('계산 잘 됐는지 확인하기!, case1의 최대값은: ',np.max(payoff_case1), '(2116.4보다 작거나 같아야함)')

buffer = avg_rate[(avg_rate<1.21) * (avg_rate>=1)]
num_case2 = len(buffer)
payoff_case2 = F*(buffer-1)*1.5+F+100
#print(num_case2)
print('계산 잘 됐는지 확인하기!, case2의 최소값은: ',np.min(payoff_case2), '(1100보다 커야함)')
print('계산 잘 됐는지 확인하기!, case2의 최대값은: ',np.max(payoff_case2), '(1415보다 작거나 같아야함)')

buffer = avg_rate[(avg_rate<1) * (avg_rate>=0.95)]
num_case3 = len(buffer)
payoff_case3 = F*(buffer-0.95)*2+F
#print(num_case3)
print('계산 잘 됐는지 확인하기!, case3의 최소값은: ',np.min(payoff_case3), '(1000보다 커야함)')
print('계산 잘 됐는지 확인하기!, case3의 최대값은: ',np.max(payoff_case3), '(1100보다 작거나 같아야함)')

buffer = avg_rate[(avg_rate<0.95)]
num_case4 = len(buffer)
print(num_case1+num_case2+num_case3+num_case4, '=', count_simulation, '?')
payoff_case4 = F*buffer+50
#print(num_case4)
print('계산 잘 됐는지 확인하기!, case4의 최소값은: ',np.min(payoff_case4), '(50보다 커야함)')
print('계산 잘 됐는지 확인하기!, case4의 최대값은: ',np.max(payoff_case4), '(1000보다 작거나 같아야함)')

Payoff = ((sum(payoff_case1)+sum(payoff_case2)+sum(payoff_case3)+sum(payoff_case4))/count_simulation)
Maturity_payoff = Payoff*np.exp(r*t[Mt])
Pricingdate_Price = np.exp(-r*(T+t[Mt]))*Maturity_payoff
ELS_Price = Pricingdate_Price*np.exp(r*t[issue])
print('ELS Price is ', ELS_Price)


#%%

def calc_Price(count,sigma1,sigma2,interest,rhoo):
    r= interest
    sig1 = sigma1
    sig2 = sigma2
    rho=rhoo
    count_simulation=count
    timestep=Nt+1
    t=np.linspace(0,T,timestep)
    W1=np.random.normal(0,1,(count_simulation,int(timestep)))*np.sqrt(t[1]-t[0])
    W2=W1*rho+np.sqrt(1-rho**2)*np.random.normal(0,1,(count_simulation,int(timestep)))*np.sqrt(t[1]-t[0])
    
    W1[:,0]=0
    W1=W1.cumsum(axis=1)
    W2[:,0]=0
    W2=W2.cumsum(axis=1)
    stock1=K0[0]*np.exp((r-q1-(sig1**2)/2)*t+sig1*W1)
    stock2=K0[1]*np.exp((r-q2-(sig2**2)/2)*t+sig2*W2)
    
    stock1_avg = np.mean(stock1[:,-avg_date:], axis=1) #avg_date 갯수만큼의 셀을 불러와야하므로 다시 +1을 인덱싱 해줌
    stock2_avg = np.mean(stock2[:,-avg_date:], axis=1)
    
    stock1_rate = stock1_avg/K0[0]
    stock2_rate = stock2_avg/K0[1]
    
    avg_rate = np.min(pd.concat([pd.Series(stock1_rate), pd.Series(stock2_rate)],axis=1), axis=1)
    
    buffer = avg_rate[avg_rate>=1.21]
    num_case1 = len(buffer)
    payoff_case1 = F*(buffer - 1.21)*3.34+F+415
    payoff_case1 -= (payoff_case1>max_payoff)*(payoff_case1-max_payoff) #최대 페이오프가 2116.4니까 조정해주기

    buffer = avg_rate[(avg_rate<1.21) * (avg_rate>=1)]
    num_case2 = len(buffer)
    payoff_case2 = F*(buffer-1)*1.5+F+100

    buffer = avg_rate[(avg_rate<1) * (avg_rate>=0.95)]
    num_case3 = len(buffer)
    payoff_case3 = F*(buffer-0.95)*2+F

    buffer = avg_rate[(avg_rate<0.95)]
    num_case4 = len(buffer)
    print(num_case1+num_case2+num_case3+num_case4, '=', count_simulation, '?')
    payoff_case4 = F*buffer+50

    Payoff = ((sum(payoff_case1)+sum(payoff_case2)+sum(payoff_case3)+sum(payoff_case4))/count_simulation)
    Maturity_payoff = Payoff*np.exp(r*t[Mt])
    Pricingdate_Price = np.exp(-r*(T+t[Mt]))*Maturity_payoff
    ELS_Price = Pricingdate_Price*np.exp(r*t[issue])
    return ELS_Price


cs = np.arange(5000, 100001,1000)
price = []

for i in (cs):
    price.append(calc_Price(count=i,sigma1=sig1,sigma2=sig2,interest=r,rhoo=rho))


pd.Series(price).plot()
#%%

s1 = np.arange(0, 0.51, 0.1)
price = []
for i in (s1):
    price.append(calc_Price(count=count_simulation,sigma1=i,sigma2=sig2,interest=r,rhoo=rho))
pd.Series(price).plot()

#%%
s2 = np.arange(0, 0.51, 0.1)
price = []
for i in (s2):
    price.append(calc_Price(count=count_simulation,sigma1=sig1,sigma2=i,interest=r,rhoo=rho))
pd.Series(price).plot()

#%%

rate = np.arange(0, 0.051, 0.01)
price = []
for i in (rate):
    price.append(calc_Price(count=count_simulation,sigma1=sig1,sigma2=sig2,interest=i,rhoo=rho))
pd.Series(price).plot()

#%%

ro = np.arange(-1,1.1, 0.5)
price = []
for i in ro:
    price.append(calc_Price(count= count_simulation, sigma1=sig1, sigma2 = sig2, interest=r, rhoo=i))
pd.Series(price).plot()






