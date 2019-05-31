# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 16:52:53 2019

@author: kur7
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
import re
import os
os.chdir('C:\\Users\\kur7\\OneDrive\\바탕 화면')

#%%



RFP = pd.DataFrame([[16.51, 101.42,102.3],[10.53,32.15,44.22]],   ['P/E','EV/EBITDA'], 
             columns = ['Walt Disney', 'Amazon', 'Netflix'])

colormap = mpl.cm.Set2.colors

ax = RFP.plot.bar(rot = 0, color = colormap)
plt.title('Relative Financial Performance', fontsize = 30)
plt.legend(loc='upper right', fontsize=20)
plt.xticks(np.arange(0,2),('P/E', 'EV/EBITDA'), fontsize = 15)
plt.yticks(np.arange(0,101, 20), fontsize = 15)
plt.colorbar

#%%

#%%

sp500 = pd.read_csv('S&P 500 Historical Data.csv', index_col = 'Date').iloc[:,-1]
nfx = pd.read_csv('NFLX Historical Data.csv', index_col = 'Date').iloc[:,-1]
df = pd.concat([sp500, nfx], axis=1)
df.columns = ['S&P500', 'Netflix']

df = df.iloc[::-1]

df['S&P500'] = df['S&P500'].str[:-1]
df['Netflix'] = df['Netflix'].str[:-1]
df= df.astype(float)

#df['S&P500'].apply(lambda e: e.split()[-1])


df.plot.line(color = colormap)
plt.xlabel('Date', fontsize=15)
plt.ylabel('% Change', fontsize = 15)
plt.title('12 Month Performance', fontsize = 30)
plt.legend(loc='upper right', fontsize=20)
plt.xticks(np.arange(0, len(df), step =21 ),('Apr 30', 'May 31','Jun 29','Jul 31','Aug 31',
           'Sep 28', 'Oct 31' , 'Nov 30', 'Dec 31', 'Jan 31', 'Feb 28', 'Mar 29'), fontsize=15)
plt.yticks(np.arange(-10,11,2.5), fontsize = 15)

#%%

df= pd.read_excel('statistic_id616210.xlsx', index_col = 'Average internet connection speed in the U.S. 2007-2017').iloc[2:,:]
df.columns = ['Speed (Mbps)']
df.plot(color = colormap)
plt.legend(loc = 'upper left', fontsize=20)
plt.ylabel('Mbps', fontsize=15)
plt.title('Average U.S. Internet Connection Speed (Mbps)', fontsize=30)
plt.xticks(np.arange(0,len(df),4 ), ('Q3 07', 'Q3 08', 'Q3 09', 'Q3 10', 'Q3 11', 'Q3 12',
           'Q3 13' , 'Q3 14', 'Q3 15' , 'Q3 16' ), fontsize=15)
plt.xlabel('')
plt.yticks(np.arange(4,19,2), fontsize=15)

#%%
df = pd.DataFrame({'Domestic Streaming': ['1470042','1505499','1547210','1630274', '1820019','1893222',
                                          '1937314','1996092','2073555'],
                   'International Streaming':['1046199','1165228','1327435','1550329','1782086',
                                              '1921144', '1973283','2105592','2366749'],
                                              'Domestic DVD':['120394','114737','110214','105152',
                                                              '98751','92904','88777','85157','80688']})



df =df.astype(float)
df.plot(color = colormap)
plt.title('Revenue (in thousands)', fontsize=30)
plt.legend(fontsize=20)
plt.yticks(np.arange(0, 2500000, 500000), ('$0','$500,000' , '$1,000,000', '$1,500,000','$2,000,000'), fontsize=15)
plt.xticks(np.arange(0,len(df)), ('Q1 17', 'Q2 17', 'Q3 17', 'Q4 17', 'Q1 18', 'Q2 18',
           'Q3 18' , 'Q4 18', 'Q1 19' ), fontsize=15)

#%%

df = pd.DataFrame({'FCF':['-120','-170','-1085','-2000','-2400','-4000'], 'Net Income (loss)':['200','500','200','300','700','1211']})

df = df.astype(float)
df.plot()
plt.yticks(np.arange(-3000,1001,1000), ('-$3,000','-$2,000','-$1,000','$0','$1,000'), fontsize=15)
plt.xticks(np.arange(0,6), ('2013','2014','2015','2016','2017','2018'), fontsize=15)
plt.title('Segregating FCF and Net Income (thousands)', fontsize = 30)
plt.legend(fontsize=20)

























