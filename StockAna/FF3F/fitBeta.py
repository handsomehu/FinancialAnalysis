'''
Created on May 15, 2019

@author: I038825
'''
import pandas as pd
import numpy as np
import jqdatasdk as jqd
from jqdatasdk import *
import sys
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols



smb=pd.read_csv("./data/smb.csv")
hml=pd.read_csv("./data/hml.csv")
pfl = ["000025.XSHE","000026.XSHE","000031.XSHE","000040.XSHE","000055.XSHE"]

'''
jqd.auth("18621861857", "P4ssword")

dfmkt = get_bars('000906.XSHG', 66, unit='1M',fields=['date','close'],include_now=False,end_dt='2019-05-01',fq_ref_date='2019-05-01')
#dfmkt["code"]='000906.XSHG'
dfmkt["return"] = dfmkt["close"].pct_change()
dfmkt = dfmkt.set_index(["date"])
dfmkt.to_csv("./data/mkt.csv")
sys.exit(0)

dfbv = get_bars(pfl[0], 66, unit='1M',fields=['date','close'],include_now=False,end_dt='2019-05-01',fq_ref_date='2019-05-01')
dfbv["code"]=pfl[0]
dfbv["return"] = dfbv["close"].pct_change()
dfbv = dfbv.set_index(["date","code"])
for i in range(1,len(pfl)):  
    temp_bv = get_bars(pfl[i], 66, unit='1M',fields=['date','close'],include_now=False,end_dt='2019-05-01',fq_ref_date='2019-05-01')
    temp_bv["code"]=pfl[i]
    temp_bv["return"] = temp_bv["close"].pct_change()
    temp_bv = temp_bv.set_index(["date","code"])
    dfbv = dfbv.append(temp_bv)
    
dfbv.to_csv("./data/pfl.csv")
'''
df = pd.read_csv("./data/pfl.csv")


del df["close"]
df = df.dropna()


df= df[df["date"]>='2014-01-01']
df["date"] = df.apply(lambda x: x['date'][:7], axis=1)

dfr = df.groupby("date").mean()

print(dfr)
smb = pd.read_csv("./data/smb.csv",index_col="date")
hml = pd.read_csv("./data/hml.csv",index_col="date")
mkt = pd.read_csv("./data/mkt.csv")
del mkt["close"]
mkt = mkt.dropna()
mkt= mkt[mkt["date"]>='2014-01-01']
mkt["date"] = mkt.apply(lambda x: x['date'][:7], axis=1)
mkt.set_index("date")
#print(mkt)
#X = (mkt["return"],smb["return"],hml["return"])
#print(X)
mkt.rename(columns = {'return': 'mkt_return'}, inplace=True)
smb.rename(columns = {'return': 'smb_return'}, inplace=True)
hml.rename(columns = {'return': 'hml_return'}, inplace=True)


X = mkt.join(smb,how="inner",on="date")
X = X.join(hml,how="inner",on="date")
dfr.rename(columns = {'return': 'my_return'}, inplace=True)
X = X.join(dfr,how="inner",on="date")
#x=np.array(X[['square_feet','bedrooms']]).reshape(len(data),2)#不管什么方法将list或DataFrame或Series转化成矩阵就行
#y=np.array(data['price']).reshape(len(data),1)

print(X)
model = ols(formula='my_return ~ mkt_return+smb_return+hml_return', data=X).fit()
print(model.summary())
sys.exit(0)
y_pred = model.predict(X[["mkt_return","smb_return","hml_return"]])
X["pred"] = y_pred
print(X)
print(model.summary())
print(model.params)


'''
#### 统计量参数
def get_lr_stats(x, y, model):
    message0 = '一元线性回归方程为: '+'\ty' + '=' + str(model.intercept_[0])+' + ' +str(model.coef_[0][0]) + '*x'
    from scipy import stats
    n     = len(x)
    y_prd = model.predict(x)
    Regression = sum((y_prd - np.mean(y))**2) # 回归
    Residual   = sum((y - y_prd)**2)          # 残差
    R_square   = Regression / (Regression + Residual) # 相关性系数R^2
    F          = (Regression / 1) / (Residual / ( n - 2 ))  # F 分布
    pf         = stats.f.sf(F, 1, n-2)
    message1 = ('相关系数(R^2)： ' + str(R_square[0]) + '；' + '\n' + 
                '回归分析(SSR)： ' + str(Regression[0]) + '；' + '\t残差(SSE)： ' +  str(Residual[0]) + '；' + '\n' + 
                '           F ： ' + str(F[0])  + '；' + '\t' + 'pf ： '  + str(pf[0])   )
    ## T
    L_xx  =  n * np.var(x)
    sigma =  np.sqrt(Residual / n) 
    t     =  model.coef_ * np.sqrt(L_xx) / sigma
    pt    =  stats.t.sf(t, n-2)
    message2 = '           t ： ' + str(t[0][0])+ '；' +  '\t' + 'pt ： '  + str(pt[0][0])
    return print(message0 +'\n' +message1 + '\n'+message2)

'''
