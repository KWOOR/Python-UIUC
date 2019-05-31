# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:36:31 2019

@author: kur7
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 17:00:24 2019

@author: 우람
"""


from scipy.stats import norm
from scipy.optimize import root, fsolve, newton
import scipy.interpolate as spi
import statsmodels.api as sm
from scipy import stats
import copy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import scipy as sp
import math
import os
import time
from datetime import date
from datetime import timedelta
from datetime import datetime


T=5 #만기는 5년
Nt= 260 #52주씩 5년이면 260이지
Nx=100 #100개의 노드로 찍음
Nx0=round(Nx/2)
h=T/Nt # dt!! 1/52이다!! 문제에서 1/52로 맞추라고 함
S0 =145.5
Smax=S0*2#떨어지는 시나리오 반절, 올라가는 시나리오 반절로 맞추기 위함 
Smin=0
k=(Smax-Smin)/Nx #dx
B=250 #250달러 이상으로 주가가 넘어가면 Call을 한다! callable date 이후로
F=100 #Face Value
coupon = 0.05 #일 년 쿠폰이자율
convert = 0.59735 #conversion ratio 4월까지는 아무때나 가능!
callable_date = Nt*3/5 # 3년 뒤부터 callable 하니까 노드에 찍으면 이때 이후부터 callable 가능
coupon_date=[int((Nt/T)/2)*i for i in range(2*T)] #반기에 한번씩 주고, dt= 1/52니까 26주에 한 번씩 주지!! 
r=0.024
y=0.03
w=8.4

#conversion_date = Nt*9/10 #conversion은 아무때나 가능하지만, 만기 6개월전부터는 안 됨!! 
whether = True # 만약.. 전환이 아무때나 가능하지 않다면... False로 바꿔라!! 


#%%
def a_n(n,h,k, y=0, sig=0.27):
    return -h*(sig**2)*((n*k)**2)/(2*(k**2)) + h*(r-y)*(n*k)/(2*k)
def b_n(n,h,k,y=0, sig=0.27):
    return 1+ r*h + (sig**2)*((n*k)**2)*h/(k**2)
def c_n(n,h,k,y=0, sig=0.27):
    return -h*(sig**2)*((n*k)**2)/(2*(k**2)) - h*(r-y)*(n*k)/(2*k)
def d_n(u,n, m=0):
    x= u[m,n]
    return x

def TDMAsolver(a, b, c, d):

    nf = len(d) # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d)) # copy arrays
    for it in range(1, nf):
        mc = ac[it-1]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1] 
        dc[it] = dc[it] - mc*dc[it-1]
               
    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    return xc    


#%%
    
def make_price(Nt, Nx):
    h=T/Nt 
    k=(Smax-Smin)/Nx
    Nx0=int(S0/k)
    coupon_date=[int((Nt/T)/2)*i for i in range(2*T)]
    period = int(coupon_date[1] - coupon_date[0])
    callable_date = Nt*3/5
    
    u = np.zeros((Nt+1,Nx+1))
    u[0,math.ceil(Nx0*B /S0 ):] = F*(1+ (coupon*coupon_date[1]*h))
    for i in range(math.ceil(Nx0*B/S0)):
        u[0,i]=np.maximum(i*k*convert*whether, F*(1+ (coupon*coupon_date[1]*h)) )
    u[:,0] = 0
    
    for m in range(Nt):
        a=list()
        b=list()
        c=list()
        d=list()
        for n in range(Nx+1):
            a.append(a_n(n=n,h=h,k=k,y=y, sig= w/np.sqrt(n*k)))
            b.append(b_n(n=n,h=h,k=k,y=y, sig= w/np.sqrt(n*k)))
            c.append(c_n(n=n,h=h,k=k,y=y, sig= w/np.sqrt(n*k)))
            d.append(d_n(u,n=n, m=m))
        b[0]=1
        c[0]=0
#        d[0]=0... #이렇게 할거면 u의 열 범위를 1:Nx가 아니라 그냥 :Nx로 고쳐줘야함.. 결과값은 같으니까 노 상
        u[m+1,:]= TDMAsolver(a[1:],b,c[:-1],d)
        buff = u.copy()
        if Nt - (m+1) >=callable_date:
            for i in range(math.ceil(B/k),len(u.T)): #Call이 될 주식 구간
#                u[m+1,i] = F*(1+ (coupon*(((Nt-m-1)%period))*h))  #AI로 했을 때
                u[m+1,i] = F*(1+coupon*coupon_date[1]*h)  #AI가 아닌 그냥 102.5를 줬을 때... AI 주는게 맞는거같은데... 
                if m+1 in coupon_date:
                    u[m+1,i] = F*(1+coupon*coupon_date[1]*h)
            for i in range(1,math.ceil(B/k)):  #Call이 안 될 주식 구간 
                if m+1 in coupon_date:
                    u[m+1,i] = np.maximum(convert*i*k, buff[m+1,i] + F*coupon*coupon_date[1]*h) #전환하면 쿠폰 못 받는다고 가정함!! 
                else:
                    u[m+1,i] = np.maximum(convert*i*k, buff[m+1,i]) #buff에 AI를 더해야 하는걸까...???
        else:
             if m+1 in coupon_date:
                 for i in range(1,len(u.T)):
                     u[m+1,i] = np.maximum(convert*i*k, buff[m+1,i] + F*coupon*coupon_date[1]*h) #전환하면 쿠폰 못 받는다고 가정함!! 
             else:
                 for i in range(len(u.T)):
                     u[m+1,i] = np.maximum(convert*i*k, buff[m+1,i]) #buff에 AI를 더해야 하는걸까...???
        u[m+1,0]=0

    return u 

X_grid=100
T_grid=260*3
pv = make_price(T_grid, X_grid)
dx=(Smax-Smin)/X_grid
print("In FDM method when StockPrice-grid are",X_grid, 
      ", the Fair Value of the CB is $",make_price(Nt=T_grid,Nx=X_grid)[-1,int(S0/dx)] )

X_grid=1000
dx=(Smax-Smin)/X_grid
print("In FDM method when StockPrice-grid are",X_grid, 
      ", the Fair Value of the CB is $",make_price(T_grid,X_grid)[-1,int(S0/dx)] )

X_grid=30
dx=(Smax-Smin)/X_grid
print("In FDM method when StockPrice-grid are",X_grid, 
      ", the Fair Value of the CB is $",make_price(T_grid,X_grid)[-1,int(S0/dx)] )

#%%
price=[]
for i in range(30, 1000, 10):
    k=(Smax-Smin)/i
    price.append(make_price(Nt=T_grid,Nx=i)[-1,int(S0/k)])
    


plt.scatter(np.arange(30,1000,10),price)
plt.xlabel('# of Stock Nodes')
plt.ylabel('Value of CB')
plt.title('Value of CB')

#%%

Nx=100
k=(Smax-Smin)/Nx
price=[]
for i in np.arange(260, 780, 10):
    T_grid=i
    price.append(make_price(Nt=T_grid,Nx=100)[-1,int(S0/k)])

plt.plot(np.arange(260,780,10), price)
plt.xlabel('W')
plt.ylabel('Value of CB')
plt.title('Value of CB')






#%%
    
print('이자율이 변할 수 있다. 이자율이 만기까지 고정이라고 가정했음. 이자율이 오르면 CB 가격은 떨어진다.',
      '실제로 돌려보니까 떨어짐. 그리고 이론과도 부합한다..\n')
print('주식 grid의 수에 따라 가격이 변한다. 현실에서는 grid가 무수히 많을 것이므로 CB의 가격도 오를 것이다.',
      '실제로 돌려보면 grid가 늘어남에 따라 가격이 조오금씩 오른다.\n')
print('주식가격의 최대값, 즉 Smax는 무한대이다. 그러나 FDM을 풀기위해 2배로 제한을 뒀다. ',
      'Smax를 크게하면 할 수록 CB의 가격은 올라간다. 그러나 그만큼 주식 grid의 수도 늘려서 dx를 적절히',
      '조정해야한다. 그렇지않으면 dx가 엄청 커져서 주식가격의 미세한 변화를 잡아내지 못하고,'
      'Call 구간에 큰 영향을 준다. 또한 grid에는 주식 가격이 오르는 시나리오가 대부분일 것이므로 CB가격이 상승한다. \n' )

print('https://pdfs.semanticscholar.org/7958/000c73438122bdc6e2119f5fa6d6d781cf26.pdf' ,'참조') 


#%%


print('delta is ', (pv[-1,51]-pv[-1,49])/(2*k))
print('If I bought $100 face value, then I need to sell $', ((pv[-1,51]-pv[-1,49])/(2*k))*S0,'.', 
      'It is ',(pv[-1,51]-pv[-1,49])/(2*k), 'shares per $100 Face Value.')

#%%

''' 10million은 천만'''
week_node = len(pv)-int((datetime(2019,1,31)- datetime(2018,10,4)).days/7)
Nx=100
k=(Smax-Smin)/Nx
stock_node = math.floor(78/k)
pv = make_price(Nt,Nx)
print('The theoretical value of the bond is ', pv[week_node, stock_node] ,'\n')
print('I sold ',(10000000/100)*(pv[-1,51]-pv[-1,49])/(2*k), 'shares of the stock at $',S0,
      'and now the stock price is $78 so I make $',(S0-78)*(10000000/100)*(pv[-1,51]-pv[-1,49])/(2*k),
      'from selling the stocks.', 'I lose the money from this bond but it is less than what I make from the stocks.',
      'So, Totally I make the money \n')
print('The market price of this bond is higher than the theoretical value so, We can interpret that',
      'the market expects the price will go up.')

#%%

''' 모델이 틀렸다는건 아니고... 주가랑 시간이 변하면서 델타가 바뀌니까 계속해서 델타를 조정해주는 방법 말고는'''
'''없을것 같은데..? '''





#%%

''' C-N'''
def a_(n, y=0, sig=0.27):
    return ( (sig**2)*(n**2)-(r-y)*n )/4
def b_(n,y=0, sig=0.27):
    return -( (sig**2)*(n**2) )/2 - r/2 - 1/h
def c_(n,y=0, sig=0.27):
    return ( (sig**2)*(n**2)+(r-y)*n )/4
def d_(u,n, sig=0.27, m=0):
    return -u[m,n-1]*( (sig**2)*(n**2)-(r-y)*n )/4 -u[m,n]*( -( (sig**2)*(n**2) )/2 - r/2 + 1/h) -u[m,n+1]*( (sig**2)*(n**2)+(r-y)*n )/4


def make_price_CN(Nt, Nx):
    h=T/Nt 

    k=(Smax-Smin)/Nx
    Nx0=int(S0/k)    
    u = np.zeros((Nt+1,Nx+1))
    u[0,math.ceil(Nx0*B /S0 ):] = F*(1+ (coupon*coupon_date[1]*h))
    for i in range(math.ceil(Nx0*B/S0)):
        u[0,i]=np.maximum(i*k*convert*whether, F*(1+ (coupon*coupon_date[1]*h)) )
    u[:,0] = 0
    
    for m in range(Nt):
        a=list()
        b=list()
        c=list()
        d=list()
        for n in range(1,Nx):
            a.append(a_(n=n,y=y, sig= 8.4/np.sqrt(n*k)))
            b.append(b_(n=n,y=y, sig= 8.4/np.sqrt(n*k)))
            c.append(c_(n=n,y=y, sig= 8.4/np.sqrt(n*k)))
            d.append(d_(u, n=n,  sig= 8.4/np.sqrt(n*k), m=m))
#        b[0]=1
#        c[0]=0
#        d[0]=0... 이렇게 할거면 u의 열 범위를 1:Nx가 아니라 그냥 :Nx로 고쳐줘야함.. 결과값은 같으니까 노 상
        u[m+1,1:Nx]= TDMAsolver(a[1:],b,c[:-1],d)
        buff = u.copy()
        if Nt - (m+1) >=callable_date:
            for i in range(math.ceil(B/k),len(u.T)): #Call이 될 주식 구간
                u[m+1,i] = F*(1+ (coupon*(((Nt-m-1)%26))*h)) 
                if m+1 in coupon_date:
                    u[m+1,i] = F*(1+coupon*coupon_date[1]*h)
            for i in range(1,math.ceil(B/k)):  #Call이 안 될 주식 구간 
                if m+1 in coupon_date:
                    u[m+1,i] = np.maximum(convert*i*k, buff[m+1,i] + F*coupon*coupon_date[1]*h) #전환하면 쿠폰 못 받는다고 가정함!! 
                else:
                    u[m+1,i] = np.maximum(convert*i*k, buff[m+1,i]) #buff에 AI를 더해야 하는걸까...???
        else:
             if m+1 in coupon_date:
                 for i in range(1,len(u.T)):
                     u[m+1,i] = np.maximum(convert*i*k, buff[m+1,i] + F*coupon*coupon_date[1]*h) #전환하면 쿠폰 못 받는다고 가정함!! 
             else:
                 for i in range(len(u.T)):
                     u[m+1,i] = np.maximum(convert*i*k, buff[m+1,i]) #buff에 AI를 더해야 하는걸까...???
        u[m+1,0]=0
            
    return u 


Nx=100
pv_CN = make_price_CN(Nt,Nx)
k=(Smax-Smin)/Nx
print("In FDM method when StockPrice-grid are",Nx, 
      ", the Fair Value of the CB is $",make_price_CN(Nt,Nx)[-1,int(S0/k)] )

Nx=300
k=(Smax-Smin)/Nx
print("In FDM method when StockPrice-grid are",Nx, 
      ", the Fair Value of the CB is $",make_price_CN(Nt,Nx)[-1,int(S0/k)] )

Nx=30
k=(Smax-Smin)/Nx
print("In FDM method when StockPrice-grid are",Nx, 
      ", the Fair Value of the CB is $",make_price_CN(Nt,Nx)[-1,int(S0/k)] )


#%%

#
#count_simulation= 20000 
#timestep=Nt+1
#t=np.linspace(0,5,timestep)
#W1=np.random.normal(0,1,(int(count_simulation),int(timestep)))*np.sqrt(t[1]-t[0])
#stock = np.zeros_like(W1)
#stock[:,0] = S0
#for i in range(1,len(stock.T)):
#    stock[:,i] = stock[:,i-1]+ (r-y)*h*stock[:,i-1] + (8.4/np.sqrt(stock[:,i-1]))*stock[:,i-1]*W1[:,i]
#
#stock[np.isnan(stock)] = 0
#stock[stock<=0] = 0
#
#
#bond = np.zeros((1, Nt+1))
#for m in range(Nt+1):
#    bond[:,m] = F*(1+ (coupon*(m%26)*h))
#    if m -26 in coupon_date:
#        bond[:,m] = F*(1+coupon*coupon_date[1]*h)
#
#note = stock.copy()
#
#for i in range(len(stock)):
#    for j in range(len(stock.T)):    
#        if np.max(note[i,:])*convert >= bond[:,j]:
#            index =  np.where(note[i,:] == np.max(note[i,:]))[0]
#            note[i,index] = note[i,index]*convert*np.exp(-r*index*h)
#            note[i,int(index+1):] = 0
#            note[i,:int(index)] = 0
#        elif note[i,j] !=0 and j>=callable_date:
#            if note[i,j] >= B:
#               note[i,j] = bond[:,j]*np.exp(-r*h*j)
#               note[i,j+1:] =0
#               note[i,:j] =0
#    if note[i,-1] !=0:
#        note[i,-1] = bond[:,-1]*np.exp(-r*T)
#        note[i,:-1]=0
#    if sum(note[i,:] > 0) >=2:
#        note[i,:] =0
#    
#sum(sum(음표))/count_simulation
#
##sum(np.max(note, axis=1) == 0)
#
#
##%%
#''' 이거 진짜''' 
#
#count_simulation= 20000 
#timestep=Nt+1
#t=np.linspace(0,5,timestep)
#W1=np.random.normal(0,1,(int(count_simulation),int(timestep)))*np.sqrt(t[1]-t[0])
#stock = np.zeros_like(W1)
#stock[:,0] = S0
#for i in range(1,len(stock.T)):
#    stock[:,i] = stock[:,i-1]+ (r-y)*h*stock[:,i-1] + (8.4/np.sqrt(stock[:,i-1]))*stock[:,i-1]*W1[:,i]
#
#stock[np.isnan(stock)] = 0
#stock[stock<=0] = 0
#
#
#bond = np.zeros((1, Nt+1))
#for m in range(Nt+1):
#    bond[:,m] = F*(1+ (coupon*(m%26)*h))
#    if m -26 in coupon_date:
#        bond[:,m] = F*(1+coupon*coupon_date[1]*h)
#
#note = stock.copy()
#
#for i in range(len(stock)):
##    index =  np.where(note[i,:] == np.max(note[i,:]))[0]
##    if np.max(note[i,:])*convert >= bond[:,int(index)]:
##        note[i,index] = note[i,index]*convert*np.exp(-r*index*h)
##        note[i,int(index+1):] = 0
##        note[i,:int(index)] = 0
##        pass
#    if note[i,-1] >= B:
#        note[i,-1] = bond[:,-1]*np.exp(-r*T*h)
#        note[i,:-1] = 0
#    if note[i,-1]*convert >= bond[:,-1]:
#
#        note[i,-1] = note[i,-1]*convert*np.exp(-r*T*h)
#        note[i,:-1] = 0
##        note[i,:int(index)] = 0
#    else:
#        for j in range(len(stock.T)):    
#            if note[i,j] !=0 and j>=callable_date:
#                if note[i,j] >= B:
#                   note[i,j] = bond[:,j]*np.exp(-r*h*j)
#                   note[i,j+1:] =0
#                   note[i,:j] =0
#        if note[i,-1] !=0:
#            note[i,-1] = bond[:,-1]*np.exp(-r*T)
#            note[i,:-1]=0
#        if sum(note[i,:] > 0) >=2:
#            note[i,:] =0
#    
#sum(sum(note))/count_simulation
#
##sum(np.max(note, axis=1) == 0)
#
##%%
#
#count_simulation= 20000 
#timestep=Nt+1
#t=np.linspace(0,5,timestep)
#W1=np.random.normal(0,1,(int(count_simulation),int(timestep)))*np.sqrt(t[1]-t[0])
#stock = np.zeros_like(W1)
#stock[:,0] = S0
#for i in range(1,len(stock.T)):
#    stock[:,i] = stock[:,i-1]+ (r-y)*h*stock[:,i-1] + (8.4/np.sqrt(stock[:,i-1]))*stock[:,i-1]*W1[:,i]
#
#stock[np.isnan(stock)] = 0
#stock[stock<=0] = 0
#
#
#bond = np.zeros((1, Nt+1))
#for m in range(Nt+1):
#    bond[:,m] = F*(1+ (coupon*(m%26)*h))
#    if m -26 in coupon_date:
#        bond[:,m] = F*(1+coupon*coupon_date[1]*h)
#
#note = stock.copy()
#
#for i in range(len(stock)):
#    for j in range(len(stock.T)):    
#        if note[i,j]*convert >= bond[:,j]:
#            note[i,j] = note[i,j]*convert*np.exp(-r*j*h)
#            note[i,int(j+1):] = 0
#            note[i,:int(j)] = 0
#        elif note[i,j] !=0 and j>=callable_date:
#            if note[i,j] >= B:
#               note[i,j] = bond[:,j]*np.exp(-r*h*j)
#               note[i,j+1:] =0
#               note[i,:j] =0
#    if note[i,-1] !=0:
#        note[i,-1] = bond[:,-1]*np.exp(-r*T)
#        note[i,:-1]=0
#    if sum(note[i,:] > 0) >=2:
#        note[i,:] =0
#    
#sum(sum(음표))/count_simulation

#sum(np.max(note, axis=1) == 0)






