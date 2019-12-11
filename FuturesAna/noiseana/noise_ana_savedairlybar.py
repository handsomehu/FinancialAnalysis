'''
Created on Mar 11, 2019

@author: I038825
'''
import jqdatasdk as jqd
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import talib 
import random

'''
n = 10
nrows = 33
index = pd.date_range('2000-1-1', periods=nrows, freq='D')
df = pd.DataFrame(np.ones(nrows), index=index)
print(df)

first = df.index.min()
last = df.index.max() + pd.Timedelta('1D')
secs = int((last-first).total_seconds()//n)
periodsize = '{:d}S'.format(secs)

result = df.resample(periodsize).sum()
print('\n{}'.format(result))
assert len(result) == n

sys.exit(0)
'''

jqd.auth("18621861234", "111111")

#symbolName = "RB8888.XSGE"
s0 = ["A8888.XDCE","AL8888.XSGE","AU8888.XSGE","B8888.XDCE","C8888.XDCE","CF8888.XZCE","CU8888.XSGE","ER8888.XZCE","FU8888.XSGE","GN8888.XZCE","IF8888.CCFX","J8888.XDCE","L8888.XDCE","M8888.XDCE","ME8888.XZCE","P8888.XDCE","PB8888.XSGE","RB8888.XSGE","RO8888.XZCE","RU8888.XSGE","SR8888.XZCE","TA8888.XZCE","V8888.XDCE","WR8888.XSGE","WS8888.XZCE","WT8888.XZCE","Y8888.XDCE","ZN8888.XSGE"]
s1 = ["AG8888.XSGE","BB8888.XDCE","BU8888.XSGE","CS8888.XDCE","FB8888.XDCE","FG8888.XZCE","HC8888.XSGE","I8888.XDCE","JD8888.XDCE","JM8888.XDCE","JR8888.XZCE","LR8888.XZCE","MA8888.XZCE","OI8888.XZCE","PM8888.XZCE","PP8888.XDCE","RI8888.XZCE","RM8888.XZCE","RS8888.XZCE","SF8888.XZCE","SM8888.XZCE","TC8888.XZCE","TF8888.CCFX","WH8888.XZCE"]
s2 = ["AP8888.XZCE","CY8888.XZCE","IC8888.CCFX","IH8888.CCFX","NI8888.XSGE","SN8888.XSGE","T8888.CCFX","ZC8888.XZCE"]
s3 = ["CJ8888.XZCE","EG8888.XDCE","SC8888.XINE","SP8888.XSGE","TS8888.CCFX"]

slist = [
            ["A8888.XDCE","AL8888.XSGE","AU8888.XSGE","B8888.XDCE","C8888.XDCE","CF8888.XZCE","CU8888.XSGE","ER8888.XZCE","FU8888.XSGE","GN8888.XZCE","IF8888.CCFX","J8888.XDCE","L8888.XDCE","M8888.XDCE","ME8888.XZCE","P8888.XDCE","PB8888.XSGE","RB8888.XSGE","RO8888.XZCE","RU8888.XSGE","SR8888.XZCE","TA8888.XZCE","V8888.XDCE","WR8888.XSGE","WS8888.XZCE","WT8888.XZCE","Y8888.XDCE","ZN8888.XSGE"],
            ["AG8888.XSGE","BB8888.XDCE","BU8888.XSGE","CS8888.XDCE","FB8888.XDCE","FG8888.XZCE","HC8888.XSGE","I8888.XDCE","JD8888.XDCE","JM8888.XDCE","JR8888.XZCE","LR8888.XZCE","MA8888.XZCE","OI8888.XZCE","PM8888.XZCE","PP8888.XDCE","RI8888.XZCE","RM8888.XZCE","RS8888.XZCE","SF8888.XZCE","SM8888.XZCE","TC8888.XZCE","TF8888.CCFX","WH8888.XZCE"],
            ["AP8888.XZCE","CY8888.XZCE","IC8888.CCFX","IH8888.CCFX","NI8888.XSGE","SN8888.XSGE","T8888.CCFX","ZC8888.XZCE"],
            ["CJ8888.XZCE","EG8888.XDCE","SC8888.XINE","SP8888.XSGE","TS8888.CCFX"]
    
    ]
#slist = ["M8888.XDCE","RB8888.XSGE"]
barlength = 10
#slist = [["IF8888.CCFX"]]
slpre = "s"
slname = ""

slname = "s0"
print(slist[0])
jjj = 0
#global mdf
mdf = []
for sss in slist:
    for ss in sss:
        DF = jqd.get_price(ss, start_date='2009-01-01', end_date='2019-12-01', frequency='daily',fields=['open','close','high','low']) # 获得IC1506.CCFX的分钟数据, 只获取open+close字段
        DF.to_csv("./data/"+ss+"_1daybar.csv")
