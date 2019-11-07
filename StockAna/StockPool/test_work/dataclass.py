'''
Created on Nov 6, 2019

@author: I038825
'''
import pandas as pd

df1 = pd.DataFrame.from_csv("../d1.csv")

ddd = df1.groupby("industry").count()
ddd.to_csv("idx.csv")