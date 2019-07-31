import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from matplotlib.dates import AutoDateLocator, DateFormatter
'''
df = pd.read_csv("./ma_atr.csv",index_col="major",parse_dates = True)
df = df.dropna()
df = df.pct_change()
df = df.dropna()
df1 = df.resample('M')

print(df1.std())
'''

df = pd.read_csv("./data_ana/MonthlyHML_old.csv",index_col="major",parse_dates = True)

#df[["A8888.XDCE","M8888.XDCE","CF8888.XZCE","J8888.XDCE","RB8888.XSGE"]].plot()




cols = ["AL8888.XSGE","CF8888.XZCE","CU8888.XSGE","IF8888.CCFX","J8888.XDCE","RB8888.XSGE","SR8888.XZCE","TA8888.XZCE"]
df = df[cols]

print(df)
#sys.exit(0)

numPlots = len(cols)

for i in range(numPlots):
    f = plt.figure()
    plt.plot_date(df.index, df[cols[i]])
    plt.ylim(0,1)
    plt.legend(title=cols[i])
    #plt.savefig("./data_ana/"+cols[i]+".png")
    plt.show()
