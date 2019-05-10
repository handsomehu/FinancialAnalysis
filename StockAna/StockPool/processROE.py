
import pandas as pd
import numpy as np
import sys


df1 = pd.read_csv("y2016.csv",index_col = "code")
del df1["Unnamed: 0"]
df1["fiscalyear"] = "2016"

df2 = pd.read_csv("y2017.csv",index_col = "code")
del df2["Unnamed: 0"]
df2["fiscalyear"] = "2017"

df3 = pd.read_csv("y2018.csv",index_col = "code")
del df3["Unnamed: 0"]
df3["fiscalyear"] = "2018"

df = pd.concat([df1,df2])
df = pd.concat([df,df3])

df = df.reset_index()
print(df.head(10))

df = df.sort_values(["code","fiscalyear"])
print(df.head(10))

df = df.set_index(["code","fiscalyear"])



df.sort_index()

df1  =  df.groupby("code")
df1["nc"] = df1["roe"].pct_change()
print(df1)

