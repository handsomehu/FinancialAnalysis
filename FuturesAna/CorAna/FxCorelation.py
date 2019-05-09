'''
Created on Apr 25, 2019

@author: I038825
'''
'''
Created on Mar 11, 2019

@author: I038825
'''
import jqdatasdk as jqd
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

#jqd.auth("18621861857", "P4ssword")

#symbolName = "RB8888.XSGE"
symbolList = ["J8888.XDCE","I8888.XDCE","CU8888.XSGE","RB8888.XSGE","CF8888.XZCE"]
sb = "J8888.XDCE"
#DF = jqd.get_price(symbolList, start_date='2019-01-01', end_date='2019-04-25', frequency='daily',fields=['open','close','high','low','volume']) # 获得IC1506.CCFX的分钟数据, 只获取open+close字段
#DF = jqd.get_price(symbolList, start_date='2017-01-01', end_date='2019-04-25', frequency='daily',fields=['close']) # 获得IC1506.CCFX的分钟数据, 只获取open+close字段

#DF.to_hdf('pn2_years.h5', 'keyyear')
DF = pd.read_hdf("pn2_years.h5",'keyyear')

#jqd.auth("18621861857", "P4ssword")
#DF = jqd.get_price(symbolList, start_date='2017-01-01', end_date='2019-04-25', frequency='daily',fields=['close']) # 获得IC1506.CCFX的分钟数据, 只获取open+close字段

#DF = pd.read_hdf("pn1.h5",'key')
DF = DF.to_frame()
DF = DF["close"].unstack()

newDF = DF.pct_change().dropna()

all_corr = newDF.corr()
all_corr.to_csv("./output/overall_correlation.csv")
print(newDF.corr())
get_year = lambda x: x.year

by_year = newDF.groupby(get_year)
print(by_year)

y2019= by_year
c2019 = y2019.corr()
print(y2019.corr())
print(c2019.index)

mi1 = c2019.index
mi1.names = ['year', 'minor']

c2019.index = mi1

#c2019.set_index(names = ['year', 'minor'])
'''
c2019.set_index(levels=[[2017, 2018, 2019], ['J8888.XDCE', 'I8888.XDCE']],
           labels=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]],
           names=['year', 'minor'])
'''

All = slice(None)
filteryear = slice('2017')
y = set()
for  x in c2019.index:

    y.add(x[0])
    

figcnt = len(y)

'''
f1 = slice(str(2018))
cf = c2019.loc[(f1,All),:]
print(cf)
print(c2019.loc[[2018]])
sys.exit(0)
'''

figure1 = plt.figure(1)
ti = 221
j = 0
#fig, ax = plt.subplots(1,figcnt)

for i  in y:
    ti = ti + j
    ax = plt.subplot(ti)
 
    filteryear = None
    filteryear = slice(str(i))
    cf = c2019.loc[[i]]
    cf.to_csv("./output/corr_"+str(i)+".csv")
    sns.heatmap(cf)
    plt.sca(ax)
    j += 1
plt.show()

                       
