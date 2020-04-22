'''
Created on Apr 25, 2019

@author: I038825
'''
'''
Created on Mar 11, 2019

@author: I038825
'''
import jqdatasdk as jqd
import pandas as pd


jqd.auth("18621861857", "")

#symbolName = "RB8888.XSGE"
zce = ["AP8888.XZCE","CF8888.XZCE","CY8888.XZCE","FG8888.XZCE","JR8888.XZCE","LR8888.XZCE","MA8888.XZCE","PM8888.XZCE",
       "RI8888.XZCE","RM8888.XZCE","OI8888.XZCE","RS8888.XZCE","SF8888.XZCE","SM8888.XZCE","SR8888.XZCE","TA8888.XZCE",
       "ZC8888.XZCE","WH8888.XZCE","WT8888.XZCE","CJ8888.XZCE","UR8888.XZCE",]
shfe = ["AG8888.XSGE","AL8888.XSGE","AU8888.XSGE","BU8888.XSGE","CU8888.XSGE","FU8888.XSGE","HC8888.XSGE","NI8888.XSGE",
        "PB8888.XSGE","RB8888.XSGE","RU8888.XSGE","SN8888.XSGE","WR8888.XSGE","ZN8888.XSGE","SP8888.XSGE","SS8888.XSGE"]
dce = ["A8888.XDCE","B8888.XDCE","BB8888.XDCE","C8888.XDCE","CS8888.XDCE","FB8888.XDCE","I8888.XDCE","J8888.XDCE",
       "JD8888.XDCE","JM8888.XDCE","L8888.XDCE","M8888.XDCE","P8888.XDCE","PP8888.XDCE","V8888.XDCE","Y8888.XDCE",
       "EG8888.XDCE","EB8888.XDCE"]
ine = ["SC8888.XINE", "NR8888.XINE"]
ccfx = ["IC8888.CCFX","IF8888.CCFX","IH8888.CCFX","T8888.CCFX","TF8888.CCFX","TS8888.CCFX"]

slist = [ccfx,dce,shfe,zce, ine ]

slist2 = [["NR8888.XINE"]]
for sub in slist:

    for sb in sub:
  
        #DF = jqd.get_price(sb, start_date='2009-01-01', end_date='2020-3-31', frequency='daily',fields=['open', 'close', 'low', 'high', 'volume', 'money', 'factor', 'high_limit','low_limit', 'avg', 'pre_close', 'paused', 'open_interest'])         DF.to_csv("./data/"+sb+".csv")
        DF = jqd.get_price(sb, start_date='2009-01-01', end_date='2020-3-31', frequency='daily',
                           fields=['open', 'high', 'low', 'close',   'volume', 'factor', 'pre_close', 'open_interest'])
        DF.to_csv("./data/" + sb + ".csv")

        symbol = sb.split(".")[0]
        DF.to_csv("./data/fxindex/"+symbol+".csv")
