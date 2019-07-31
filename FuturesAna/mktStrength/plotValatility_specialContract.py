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

df = pd.read_csv("MonthlyVolatility_old.csv",index_col="major",parse_dates = True)

print(df)
#df[["A8888.XDCE","M8888.XDCE","CF8888.XZCE","J8888.XDCE","RB8888.XSGE"]].plot()




cols = ["A8888.XDCE","AL8888.XSGE","AU8888.XSGE","B8888.XDCE","C8888.XDCE","CF8888.XZCE","CU8888.XSGE","ER8888.XZCE","FU8888.XSGE","GN8888.XZCE","IF8888.CCFX","J8888.XDCE","L8888.XDCE","M8888.XDCE","ME8888.XZCE","P8888.XDCE","PB8888.XSGE","RB8888.XSGE","RO8888.XZCE","RU8888.XSGE","SR8888.XZCE","TA8888.XZCE","V8888.XDCE","WR8888.XSGE","WS8888.XZCE","WT8888.XZCE","Y8888.XDCE","ZN8888.XSGE"]

numPlots = len(cols)
f = plt.figure()
ax = []
j=0
for i in range(14):
    print(i)
    ax.append(f.add_subplot(7,2,i+1))
    ax[i].plot_date(df.index, df[cols[i]])
    ax[i].set_title(cols[i])
    #ylim(-0.02,0.02)
    f.subplots_adjust(hspace=0.3)
#plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
f2 = plt.figure()
ax2 = []
for i in range(14):
    print(i)
    ax2.append(f2.add_subplot(7,2,i+1))
    ax2[i].plot_date(df.index, df[cols[i+14]])
    ax2[i].set_title(cols[i+14])
    ax2.ylim(-0.02,0.02)
    f2.subplots_adjust(hspace=0.3)


plt.show()
