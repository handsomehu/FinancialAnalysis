# -*- coding: utf-8 -*-
#通过基本面选股，建立股票池
import tushare as ts
import pandas as pd
import sys,time

lv_pct = 0.2
pro = ts.pro_api()
lv_startdate = "20160101"
lv_enddate = "20200101"
lv_listflag = True

if not lv_listflag:
        
    df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
    
    
    df1=df.copy()  #建立一个备份
    n=500  #选择前n个数据
    filterlist = ['002680']
    print(len(df1))
    
    #删除业绩较差的ST股票
    df1['a']=[('ST' in x[0:2] )for x in df1.name.astype(str)]  #先给ST股票做标记a
    df1['b']=[('*ST' in x[0:3] )for x in df1.name.astype(str)]  #先给ST股票做标记a
    #df1['a']=[('*ST' in x )for x in df1.name.astype(str)]  #先给ST股票做标记a
    for ft in filterlist:
        df1['c']=[(ft in x )for x in df1.symbol.astype(str)]  
    df1=df1.set_index('a')  #将a设置为索引 
    df1=df1.drop(index=[True]) #删除ST股票
    df1=df1.set_index('b')  #将b设置为索引 
    df1=df1.drop(index=[True]) #删除*ST股票
    df1=df1.set_index('c')  #将c设置为索引 
    #df1=df1.drop(index=[True]) #删除filter股票
    df1 = df1[df1.index!=True]
    df1=df1.reset_index(drop=True) #重建默认索引
    print(len(df1))
    df1.to_csv('d1.csv')
    inx = df1[['industry','ts_code']].groupby("industry").count()
    inx["ts_code"] = inx["ts_code"]*lv_pct
    inx.to_csv("d2.csv")
else:
    inx = pd.read_csv("d2.csv")
    df1 = pd.read_csv("d1.csv",index_col = 0)
    
if not lv_listflag:
    dffi = pd.DataFrame()
    i =0
    for index, row in df1.iterrows():
        print(row["ts_code"])
        temp = pro.query('fina_indicator', ts_code=row["ts_code"], start_date=lv_startdate, end_date=lv_enddate)
        #temp2 = temp1[]
        dffi = dffi.append(temp)
        time.sleep(1)
        if index % 100 == 0:
            print("KPI fetched:",index)
    
    #print(dffi)
    dffi.to_csv("./data/FI_KPIs.csv")
    
else:
    dffi = pd.read_csv("./data/FI_KPIs.csv",index_col = 0)

if lv_listflag != True:
    dfind = pd.DataFrame()
    dfind = df1[["ts_code","industry"]]    
    
    dffi = dffi.merge(dfind,how = "inner", on = "ts_code")
    dffi.to_csv("./data/FI_KPIs.csv")
else:
    dffi = pd.read_csv("./data/FI_KPIs.csv",index_col = 0)
#for index, row in inx.iterrows():

dffi = dffi[["ts_code","roa","roe","capital_rese_ps","bps","industry"]] 
dffi = dffi.dropna()
dffi = dffi.groupby(["ts_code","industry"]).mean()
print(dffi)
print("test1")
dfmean = dffi.groupby("industry").mean()
dfmean = dfmean.dropna()
print(dffi.head(10))

#dffi = pd.merge(dffi,dfmean,how = "left",on = ["industry"],suffixes=["","_y"])
dffi = dffi.join(dfmean,how = "left",on = ["industry"],rsuffix="_y")
dffi = dffi.dropna()
print(dffi.head(10))
#dffi["ts_code"] = dffi["ts_code"]
dffi["roa"] = 2*(dffi["roa"]-dffi["roa_y"])/dffi["roa_y"]
dffi["roe"] = 2*(dffi["roe"]-dffi["roe_y"])/dffi["roe_y"]
dffi["capital_rese_ps"] = (dffi["capital_rese_ps"]-dffi["capital_rese_ps_y"])/dffi["capital_rese_ps_y"]
dffi["bps"] = (dffi["bps"]-dffi["bps_y"])/dffi["bps_y"]
dffi["score"] = ( dffi["roa"]+dffi["roe"] + dffi["capital_rese_ps"] + dffi["bps"] )/6
print(dffi.head(10))
dffi = dffi.sort_values(by = ["industry","score"],ascending = False)

#dfnew = dffi[["ts_code","industry","score"]]

dffi = dffi.groupby("industry")["score"].rank(pct=True)
#dffi.index = "ts_code"
dffi=pd.DataFrame(dffi)
#dffi.dropindex()
#dffi.columns = ["ts_code","industry","score"]
print(dffi)
dffi = dffi[dffi["score"]>0.8]
dffi.to_csv("./out_8.csv")
sys.exit(0)

