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
from nltk.corpus.reader.childes import NS

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

#jqd.auth("18621861234", "111111")

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
mdf_ns2 = []
ml = 2000

#calc mean use Efficiency ratio
def bootstrapmean1(data,monte_lengh):
    df = data.reset_index()
    noisedata = []
    for i in range(monte_lengh):
        start_idx=int(random.uniform(0,1)*(len(df)-221))
        bs_idx = np.arange(start_idx, start_idx+mul*10)
        if bs_idx[-1] >= len(df):
            continue
        temp =  df.iloc[bs_idx,:] 
        #print(temp)
        #print(df.iloc[-1,0],df.iloc[-1,1])
        rg = temp.iloc[-1,1]-temp.iloc[0,1]
        #print(rg)
        noises = abs( temp["close"]-temp["close"].shift(-1))
        
        noises = noises.fillna(method='ffill')
        #print(noises)
        #print(noises.sum())
        noisedata.append(abs(rg)/noises.sum())
        #break
        
        
    sum = 0
    for ns in noisedata:
        sum += ns
            
    return sum/len(noisedata)

#calc mean use Price Density
def bootstrapmean2(data,monte_lengh):
    df = data.reset_index()
    noisedata = []
    for i in range(monte_lengh):
        start_idx=int(random.uniform(0,1)*(len(df)-221))
        bs_idx = np.arange(start_idx, start_idx+mul*10)
        if bs_idx[-1] >= len(df):
            continue
        temp =  df.iloc[bs_idx,:] 
        #print(temp)
        #print(df.iloc[-1,0],df.iloc[-1,1])
        rg = temp["high"].max()-temp["low"].min()
        rgsum = temp["high"].sum()-temp["low"].sum()
        #print(rg)
        #print(noises)
        #print(noises.sum())
        noisedata.append(abs(rg/rgsum))
        #break
        
        
    sum = 0
    for ns in noisedata:
        sum += ns
            
    return sum/len(noisedata)

#calc mean use standard of Mom
def bootstrapmean3(data,monte_lengh):
    df = data.reset_index()
    noisedata = []
    for i in range(monte_lengh):
        start_idx=int(random.uniform(0,1)*(len(df)-221))
        bs_idx = np.arange(start_idx, start_idx+mul*10)
        if bs_idx[-1] >= len(df):
            continue
        temp =  df.iloc[bs_idx,:] 
        #print(temp)
        #print(df.iloc[-1,0],df.iloc[-1,1])
        rg = temp.iloc[-1,1]-temp.iloc[0,1]
        #print(rg)
        noises = abs( temp["close"]-temp["close"].shift(-1))
        
        noises = noises.dropna()
        #print(noises)
        #print(noises.sum())
        noisedata.append(abs(rg)/noises.std())
        #break
        
        
    sum = 0
    for ns in noisedata:
        sum += ns
            
    return sum/len(noisedata)


#DF = pd.read_csv("./testnoise.csv",index_col = 0,parse_dates = True)
#dfclose = pd.DataFrame(DF["close"])
for sub1 in slist:
    for sub in sub1:
        for mul in range(2,22):
            #global mdf
            #iterlen = ml*mul#mul is larger, time frame is large, so here adjust it
            iterlen = ml #treat every length the same, use large length
            sdays = str(mul*barlength) + "D"
            DF = pd.read_csv("./data/"+sub+"_1daybar.csv",index_col = 0,parse_dates = True)
            DF = DF.dropna()
            dfclose = pd.DataFrame(DF["close"])
            #nm = bootstrapmean1(dfclose,iterlen)
            nm = bootstrapmean3(DF,iterlen)
            mdf.append([sub,sdays,nm])


#sys.exit(0)

print(mdf)
pdf = pd.DataFrame(mdf, columns=[ 'symbol', 'duration','close'])
    
pdf.to_csv("./opt/noisebyrandom_smom.csv")

# RANDOM SELECT DATA AND CALCULATE THIS VALUE


 

    

