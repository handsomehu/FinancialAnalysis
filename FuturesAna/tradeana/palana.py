'''
Created on Sep 9, 2019

@author: I038825
'''
import pandas as pd
import matplotlib.pyplot as plt
import sys


pal = pd.read_csv("../tradelog/tr.csv",index_col = 0)
print(pal)
# by order
fig, ax = plt.subplots()
pal["pal"].plot(ax=ax,kind = 'bar')
plt.show()
#sys.exit(0)
#data.groupby(['date','type']).count()['amount'].unstack().plot(ax=ax)
# by strategy
fig, ax = plt.subplots()
pal.groupby("strategyname").sum()['pal'].plot(ax=ax,kind='bar')
ax.legend()
plt.show()

fig, ax = plt.subplots()
pal.groupby("comdt").sum()['pal'].plot(ax=ax,kind='bar')
plt.show()

fig, ax = plt.subplots()
pal.groupby(["comdt","strategyname"]).sum()['pal'].plot(ax=ax,kind='bar')
fig.legend()
plt.show()