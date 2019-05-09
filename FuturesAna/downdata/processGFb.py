'''
Created on May 7, 2019

@author: I038825
'''
import pandas as pd
import sys

df = pd.read_csv("./data/002049.XSHE.csv")

print(df.columns)
print(df.index)


df.rename(columns = {'Unnamed: 0':'datetime'}, inplace = True)
df = df.reset_index(drop=True)
print(df.head(100))
df.to_csv("./data/1_002049.XSHE.csv")
sys.exit(0)



del df["Unnamed: 0"]
del df["Unnamed: 0.1"]

df.rename(columns = {'Unnamed: 0.1.1':'datetime'}, inplace = True)
#df = df.set_index('datetime', drop=True, append=False, inplace=False, verify_integrity=False) 
df = df.dropna()
df = df.reset_index(drop=True)
print(df.head(100))
df.to_csv("./data/002049.XSHE.csv")