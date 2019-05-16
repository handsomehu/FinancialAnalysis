'''
Created on May 15, 2019

@author: I038825
'''
import pandas as pd
import numpy as np
import sys

dfsg=pd.read_csv("./data/sg.csv")
dfsn=pd.read_csv("./data/sn.csv")
dfsv=pd.read_csv("./data/sv.csv")

dfbg=pd.read_csv("./data/bg.csv")
dfbn=pd.read_csv("./data/bn.csv")
dfbv=pd.read_csv("./data/bv.csv")

del dfsg["close"]
del dfsn["close"]
del dfsv["close"]
del dfbg["close"]
del dfbn["close"]
del dfbv["close"]

dfsg = dfsg.dropna()
dfsn = dfsn.dropna()
dfsv = dfsv.dropna()
dfbg = dfbg.dropna()
dfbn = dfbn.dropna()
dfbv = dfbv.dropna()


dfsg= dfsg[dfsg["date"]>='2014-01-01']
dfsg["date"] = dfsg.apply(lambda x: x['date'][:7], axis=1)
dfsn= dfsn[dfsn["date"]>='2014-01-01']
dfsn["date"] = dfsn.apply(lambda x: x['date'][:7], axis=1)
dfsv= dfsv[dfsv["date"]>='2014-01-01']
dfsv["date"] = dfsv.apply(lambda x: x['date'][:7], axis=1)


dfbg= dfbg[dfbg["date"]>='2014-01-01']
dfbg["date"] = dfbg.apply(lambda x: x['date'][:7], axis=1)
dfbn= dfbn[dfbn["date"]>='2014-01-01']
dfbn["date"] = dfbn.apply(lambda x: x['date'][:7], axis=1)
dfbv= dfbv[dfbv["date"]>='2014-01-01']
dfbv["date"] = dfbv.apply(lambda x: x['date'][:7], axis=1)




dfsg = dfsg.groupby("date").mean()
dfsn = dfsn.groupby("date").mean()
dfsv = dfsv.groupby("date").mean()

dfbg = dfbg.groupby("date").mean()
dfbn = dfbn.groupby("date").mean()
dfbv = dfbv.groupby("date").mean()

smb = (dfsv +dfsn+dfsg)/3 -(dfbv +dfbn+dfbg)/3

hml = (dfsv+dfbv)/2 - (dfsg+dfbg)/2

smb.to_csv("./data/smb.csv")
hml.to_csv("./data/hml.csv")
