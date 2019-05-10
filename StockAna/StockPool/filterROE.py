
import pandas as pd
import numpy as np
import sys
import jqdatasdk as jqd
from jqdatasdk import *
import talib

def preparedata():
    jqd.auth("18621861857", "123456")
    
    
    df = pd.read_csv("./data/roe_increase2y.csv")
    df = df[df["roe"]>10]
    df = df[df["pb_ratio"]<10]
    df = df[df["pe_ratio"]<100]
    
    
    
    df.sort_values(by=["roe","pe_ratio","pb_ratio"],  ascending=(False,True,True), inplace=True)
    print(df["code"])
    
    df_col = jqd.get_price("399001.XSHE", start_date='2019-01-01', end_date='2019-05-10', frequency='daily',fq = 'pre',fields=['close']) # 获得IC1506.CCFX的分钟数据, 只获取open+close字段
    df_col.rename(columns={"close":"399001.XSHE"}, inplace = True)
    print(df_col)
    #sys.exit(0)
    for l in df.itertuples():
        print(l[0])
    
        temp = jqd.get_price(l[1], start_date='2019-01-01', end_date='2019-05-10', frequency='daily',fq = 'pre',fields=['close']) # 获得IC1506.CCFX的分钟数据, 只获取open+close字段
        df_col[l[1]] = temp["close"]
        
    df_col.to_csv("./data/topclose.csv")
    
def processdata():
    
    df = pd.read_csv("./data/topclose.csv",index_col = 0)
    df13 = df.rolling(13).mean()
    df13 = df13[-11:]
    df55 = df.rolling(55).mean()
    df55 = df55[-11:]
    
    dfnew = df13-df55
    print(df13)
    print(df55)

    df_direction = dfnew * dfnew.shift(1)
    df_direction[df_direction>=0] =0
    df_direction[df_direction<0] =1 
    print(dfnew)
    print(df_direction )    
    dfnew = dfnew * df_direction
    dfnew[dfnew<0] =-1
    dfnew[dfnew>0] = 1     
    print(dfnew)  
    
    dddd = dfnew[dfnew>0]
    dddd = dddd.dropna()
    print(dddd)
    
    #dfnew.to_csv("./data/signal.csv")
    
    
processdata()