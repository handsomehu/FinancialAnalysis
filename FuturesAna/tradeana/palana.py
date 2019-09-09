'''
Created on Sep 9, 2019

@author: I038825
'''
import pandas as pd
import matplotlib.pyplot as plt


pal = pd.read_csv("../tradelog/tr.csv",index_col = 0)
print(pal)
# by order
pal.plot(y = "pal", kind= 'bar')
plt.show()
# by strategy

bystrategy = pal.groupby("strategyname")
bs = bystrategy.sum()
bs.plot(x = "strategyname",y = "pal", kind= 'bar')
plt.show()
