'''
Created on May 15, 2019

@author: I038825
'''
import jqdatasdk as jqd
from jqdatasdk import *
import pandas as pd
import numpy as np
import sys


df = pd.read_csv("./data/y2016basic.csv",index_col=0)
df["b2m"] = 1/df["pb_ratio"]

s=df.sort_values('market_cap',ascending=True)[:401]
s = s.set_index("code")
b=df.sort_values('market_cap',ascending=True)[401:]
b = b.set_index("code")



g=df.sort_values('b2m',ascending=True)[:268]
g=g.set_index("code")
n=df.sort_values('b2m',ascending=True)[268:535]
n=n.set_index("code")
v=df.sort_values('b2m',ascending=True)[535:]
v=v.set_index("code")

sg = s.join(g,how="inner",rsuffix='_r').index
sn = s.join(n,how="inner",rsuffix='_r').index
sv = s.join(v,how="inner",rsuffix='_r').index


bg = b.join(g,how="inner",rsuffix='_r').index
bn = b.join(n,how="inner",rsuffix='_r').index
bv = b.join(v,how="inner",rsuffix='_r').index

print(len(sg),len(sn),len(sv),len(bg),len(bn),len(bv))


jqd.auth("18621861857", "P4ssword")
dfsg = get_bars(sg[0], 66, unit='1M',fields=['date','close'],include_now=False,end_dt='2019-05-01',fq_ref_date='2019-05-01')
dfsg["code"]=sg[0]
dfsg["return"] = dfsg["close"].pct_change()
dfsg = dfsg.set_index(["date","code"])
for i in range(1,len(sg)):  
    temp_sg = get_bars(sg[i], 66, unit='1M',fields=['date','close'],include_now=False,end_dt='2019-05-01',fq_ref_date='2019-05-01')
    temp_sg["code"]=sg[i]
    temp_sg["return"] = temp_sg["close"].pct_change()
    temp_sg = temp_sg.set_index(["date","code"])
    dfsg = dfsg.append(temp_sg)



dfsn = get_bars(sn[0], 66, unit='1M',fields=['date','close'],include_now=False,end_dt='2019-05-01',fq_ref_date='2019-05-01')
dfsn["code"]=sn[0]
dfsn["return"] = dfsn["close"].pct_change()
dfsn = dfsn.set_index(["date","code"])
for i in range(1,len(sn)):  
    temp_sn = get_bars(sn[i], 66, unit='1M',fields=['date','close'],include_now=False,end_dt='2019-05-01',fq_ref_date='2019-05-01')
    temp_sn["code"]=sn[i]
    temp_sn["return"] = temp_sn["close"].pct_change()
    temp_sn = temp_sn.set_index(["date","code"])
    dfsn = dfsn.append(temp_sn)


dfsv = get_bars(sv[0], 66, unit='1M',fields=['date','close'],include_now=False,end_dt='2019-05-01',fq_ref_date='2019-05-01')
dfsv["code"]=sv[0]
dfsv["return"] = dfsv["close"].pct_change()
dfsv = dfsv.set_index(["date","code"])
for i in range(1,len(sv)):  
    temp_sv = get_bars(sv[i], 66, unit='1M',fields=['date','close'],include_now=False,end_dt='2019-05-01',fq_ref_date='2019-05-01')
    temp_sv["code"]=sv[i]
    temp_sv["return"] = temp_sv["close"].pct_change()
    temp_sv = temp_sv.set_index(["date","code"])
    dfsv = dfsv.append(temp_sv)




dfbg = get_bars(bg[0], 66, unit='1M',fields=['date','close'],include_now=False,end_dt='2019-05-01',fq_ref_date='2019-05-01')
dfbg["code"]=bg[0]
dfbg["return"] = dfbg["close"].pct_change()
dfbg = dfbg.set_index(["date","code"])
for i in range(1,len(bg)):  
    temp_bg = get_bars(bg[i], 66, unit='1M',fields=['date','close'],include_now=False,end_dt='2019-05-01',fq_ref_date='2019-05-01')
    temp_bg["code"]=bg[i]
    temp_bg["return"] = temp_bg["close"].pct_change()
    temp_bg = temp_bg.set_index(["date","code"])
    dfbg = dfbg.append(temp_bg)
    

dfbn = get_bars(bn[0], 66, unit='1M',fields=['date','close'],include_now=False,end_dt='2019-05-01',fq_ref_date='2019-05-01')
dfbn["code"]=bn[0]
dfbn["return"] = dfbn["close"].pct_change()
dfbn = dfbn.set_index(["date","code"])
for i in range(1,len(bn)):  
    temp_bn = get_bars(bn[i], 66, unit='1M',fields=['date','close'],include_now=False,end_dt='2019-05-01',fq_ref_date='2019-05-01')
    temp_bn["code"]=bn[i]
    temp_bn["return"] = temp_bn["close"].pct_change()
    temp_bn = temp_bn.set_index(["date","code"])
    dfbn = dfbn.append(temp_bn)
    

dfbv = get_bars(bv[0], 66, unit='1M',fields=['date','close'],include_now=False,end_dt='2019-05-01',fq_ref_date='2019-05-01')
dfbv["code"]=bv[0]
dfbv["return"] = dfbv["close"].pct_change()
dfbv = dfbv.set_index(["date","code"])
for i in range(1,len(bv)):  
    temp_bv = get_bars(bv[i], 66, unit='1M',fields=['date','close'],include_now=False,end_dt='2019-05-01',fq_ref_date='2019-05-01')
    temp_bv["code"]=bv[i]
    temp_bv["return"] = temp_bv["close"].pct_change()
    temp_bv = temp_bv.set_index(["date","code"])
    dfbv = dfbv.append(temp_bv)


dfsg.to_csv("./data/sg.csv")
dfsn.to_csv("./data/sn.csv")
dfsv.to_csv("./data/sv.csv")

dfbg.to_csv("./data/bg.csv")
dfbn.to_csv("./data/bn.csv")
dfbv.to_csv("./data/bv.csv")
sys.exit(0)


dfsg = dfsg.groupby("code").mean()
dfsn = dfsn.groupby("code").mean()
dfsv = dfsv.groupby("code").mean()

dfbg = dfbg.groupby("code").mean()
dfbn = dfbn.groupby("code").mean()
dfbv = dfbv.groupby("code").mean()




dfsg_out = dfsg.resample('M').mean()
dfsg_out = dfsg_out.dropna()
dfsn_out = dfsn.resample('M').mean()
dfsn_out = dfsn_out.dropna()
dfsv_out = dfsv.resample('M').mean()
dfsv_out = dfsv_out.dropna()



dfbg_out = dfbg.resample('M').mean()
dfbg_out = dfbg_out.dropna()
dfbn_out = dfbn.resample('M').mean()
dfbn_out = dfbn_out.dropna()
dfbv_out = dfbv.resample('M').mean()
dfbv_out = dfbv_out.dropna()


smb = (dfsv_out +dfsn_out+dfsg_out)/3 -(dfbv_out +dfbn_out+dfbg_out)/3

hml = (dfsv_out+dfbv_out)/2 - (dfsg_out+dfbg_out)/2

smb.to_csv("./data/smb.csv")
hml.to_csv("./data/hml.csv")

