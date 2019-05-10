
import pandas as pd
import numpy as np
import sys


df1 = pd.read_csv("./data/y2016.csv",index_col = "code")
del df1["Unnamed: 0"]
df1["fiscalyear"] = "2016"

df2 = pd.read_csv("./data/y2017.csv",index_col = "code")
del df2["Unnamed: 0"]
df2["fiscalyear"] = "2017"

df3 = pd.read_csv("./data/y2018.csv",index_col = "code")
del df3["Unnamed: 0"]
df3["fiscalyear"] = "2018"

df = pd.concat([df1,df2])
df = pd.concat([df,df3])

df = df.reset_index()


df = df.sort_values(["code","fiscalyear"])


df = df.set_index(["code","fiscalyear"])



df.sort_index()



df_temp  =  df.groupby("code").count()
#df_temp["yc"] = df_temp.count()

df_temp= df_temp[df_temp["roe"]==3]

df_temp["rc"] = df_temp["roe"]
      
del df_temp["market_cap"]
del df_temp["pe_ratio"]
del df_temp["pb_ratio"]
del df_temp["roe"]


df = df.join(df_temp, how = "inner",  rsuffix='_r')

df["roe_shift"] = df.groupby('code')["roe"].shift(1)

g=lambda x:1 if x > 0 else 0


df["roe_diff"] =  (df["roe"]-df["roe_shift"])/df["roe_shift"]
df["roe_increase"] = df["roe_diff"].apply(g)


conversion = {'market_cap' : 'last', 'pe_ratio' : 'last', 'pb_ratio' : 'last', 'roe' : 'last', 'rc' : 'mean', 'roe_shift' : 'last', 'roe_diff' : 'mean', 'roe_increase' : 'sum'}

dfout = df.groupby("code").aggregate(conversion)
#print(dfout)
dfout = dfout[dfout["roe_increase"]==2]

#df = df-df.shift(1)
#df1["nc"] = df1["roe"].pct_change()
#print(df)

dfout.to_csv("./data/roe_increase2y.csv")
