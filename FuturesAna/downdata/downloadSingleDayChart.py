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


jqd.auth("18621861857", "")

slist2 = [["SR8888.XZCE"]]
for sub in slist2:

    for sb in sub:
  
        DF = jqd.get_price(sb, start_date='2009-07-01', end_date='2019-05-30', frequency='daily',fq = 'pre',fields=['open','close','high','low','volume']) # 获得IC1506.CCFX的分钟数据, 只获取open+close字段
        DF.to_csv("./data/"+sb+"_daily.csv")
    #DF.to_hdf('pn2_years.h5', 'keyyear')
    #DF = pd.read_hdf("pn1.h5",'key')
