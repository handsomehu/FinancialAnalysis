# -*- coding: utf-8 -*-
#通过基本面选股，建立股票池
import tushare as ts

#获取最新股票数据
df=ts.get_today_all()

df1=df.copy()  #建立一个备份
n=100  #选择前n个数据

#删除业绩较差的ST股票
df1['a']=[('ST' in x )for x in df1.name.astype(str)]  #先给ST股票做标记a
df1=df1.set_index('a')  #将a设置为索引 
df1=df1.drop(index=[True]) #删除ST股票
df1=df1.reset_index(drop=True) #重建默认索引

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
zxg1=list(g)  #把集合转为列表
print()
print('基本面选股结果：',zxg1)

