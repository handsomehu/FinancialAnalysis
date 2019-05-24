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




i=0
j=0
fig, axs = plt.subplots(7, 4)
for col in cols:
    if j>3:
        j=0
    #plt.subplot(cnt + j) 
    #plt.subplots(nrows=7, ncols=4)
    axs[int(i/4),j]=plt.plot_date(df.index, df[col])
    
    #axs[int(i/4),j] = plt.gca()
    #axs[int(i/4),j].xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))  #设置时间显示格式
    #ax.xaxis.set_major_locator(AutoDateLocator(maxticks=24))       #设置时间间隔  
    plt.ylim(-0.02, 0.02)
    #lt.xticks(rotation=90, ha='center')
    #label = ['speedpoint']
    #plt.legend(label, loc='upper right')
    
    plt.grid()
    
    #ax.set_title(u'传输速度', fontproperties='SimHei',fontsize=14)  
    #axs[int(i/4),j].set_xlabel('Date')
    #axs[int(i/4),j].set_ylabel(col)
    j+=1
    i+=1
 
plt.subplot_tool()  
plt.show()
