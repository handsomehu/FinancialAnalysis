# -*- coding: utf-8 -*-
#通过基本面选股，建立股票池
import tushare as ts
import pandas as pd
import sys



#获取最新股票数据
df=ts.get_today_all()

df1=df.copy()  #建立一个备份
n=500  #选择前n个数据
filterlist = ['002680']


#删除业绩较差的ST股票
df1['a']=[('ST' in x[0:2] )for x in df1.name.astype(str)]  #先给ST股票做标记a
df1['b']=[('*ST' in x[0:3] )for x in df1.name.astype(str)]  #先给ST股票做标记a
#df1['a']=[('*ST' in x )for x in df1.name.astype(str)]  #先给ST股票做标记a
for ft in filterlist:
    df1['c']=[(ft in x )for x in df1.code.astype(str)]  
df1=df1.set_index('a')  #将a设置为索引 
df1=df1.drop(index=[True]) #删除ST股票
df1=df1.set_index('b')  #将b设置为索引 
df1=df1.drop(index=[True]) #删除*ST股票
df1=df1.set_index('c')  #将c设置为索引 
df1=df1.drop(index=[True]) #删除filter股票
df1=df1.reset_index(drop=True) #重建默认索引

df1.to_csv('d1.csv')
#删除业绩亏损的股票
df1=df1[df1.per >0]

#删除净资产为负的股票
df1=df1[df1.pb >0]

#选取市盈率前100名股票
df2=df1.sort_values(by=['per'],ascending=True).head(n)

#选取市净率100名股票
df3=df1.sort_values(by=['pb'],ascending=True).head(n)

#生成股票代码集合，进行集合运算
g2=set(df2.code) #低市盈率股票代码
g3=set(df3.code) #低市净率股票代码
g=g2&g3 #集合交运算
dfg = pd.DataFrame(list(g),columns=['code'])
dfg.to_csv("./out.csv")

dfnew = pd.merge(df2,df3,how = 'inner', suffixes=['_l', '_r'])
dfnew.to_csv("./outnew.csv")
