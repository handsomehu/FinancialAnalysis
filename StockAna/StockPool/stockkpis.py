'''
Created on Nov 29, 2019

@author: I038825
'''
import tushare as ts
import pandas as pd
import sys,time
import talib as tl
from datetime import datetime,timedelta


ts.set_token("e945701f493de3781f546e21895c9896cabf3c0bea1e549a3409efde")
sd = datetime.now() - timedelta(100)
sd = sd.replace(hour=0, minute=0, second=0, microsecond=0)

sdstr = sd.strftime("%Y%m%d") 

p1 = 13
p2 = 21
p3 = 55
ma1 = "ma"+str(p1)
ma2 = "ma"+str(p2)
ma3 = "ma"+str(p3)
mom1 = "mom"+str(p1)
mom2 = "mom"+str(p2)
mom3 = "mom"+str(p3)

def amscode(ts_code):
    return "0" + ts_code if ts_code[0] == "6" else "1" +ts_code

df = pd.read_csv(R"C:/app/AMS_GUI2251/out1.csv")
df["amscode"] = df["ts_code"].apply(amscode )
#print(df)
tempdf = pd.read_csv(R"C:/app/AMS_GUI2251/todaypos.csv",index_col = 0)
count = 0
dfnew = pd.DataFrame()
#df = df[:3]
for idx, row in df.iterrows():
    tscode = row["ts_code"]
    amscode = row["amscode"]
    
    
    df1 = ts.pro_bar(ts_code=tscode, start_date=sdstr,adj='qfq', freq = 'D' , ma=[p1,p2,p3])
    df1[mom1] = df1["close"].shift(p1*-1)
    df1[mom2] = df1["close"].shift(p2*-1)
    df1[mom3] = df1["close"].shift(p3*-1)
    df1["vwap"] = 10*df1["amount"]/df1["vol"]
    df1= df1[["ts_code","trade_date","close","change",ma1,ma2,ma3,mom1,mom2,mom3,"vwap"]]
    df1["change"] = abs(df1["change"])
    
    
    df1 = df1.sort_index( ascending=False)
    df1["changema"] = df1["change"].rolling(20 ).mean()
    df1 = df1.sort_index( ascending=True)
    #df1 = df1.drop(index)
    df1 = df1[:1]
    df1["amscode"] = amscode
    #print(df1)
    dfnew = dfnew.append(df1)
    if count %2 == 0:
        time.sleep(1)
    count+= 1
   
dfnew = dfnew.reset_index()
del dfnew["index"] 
dfnew = dfnew.dropna()
dffilter = dfnew[dfnew["close"]>20]
dffilter = dffilter[~dffilter["ts_code"].isin(list(tempdf["ts_code"]))]
flist = list(dffilter["ts_code"])

dfnew = dfnew[~dfnew["ts_code"].isin(flist)]
dfnew = dfnew.reset_index()
dfnew.to_csv(R"C:/app/AMS_GUI2251/dailykpi.csv")
#print(df1)