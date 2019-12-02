import sys
import os
import time
import threading
import ctypes
import pandas as pd
from datetime import datetime,timedelta
from .amsclass import Strategy,Market

# 创建行情源对象
stks = Market.Stk(['0600000','1000001'])
oldpos = []
todaypos = []
openpos = []
calclist = {}
idk = {}
stkpool = pd.DataFrame()
#stks = Market.Stk(['0600000'])
count = 0

def endproc():
    cte = datetime.now()
    cte = cte.replace(second=0, microsecond=0)
    if (cte.hour >14 and cte.miniute > 10):
        for idx, row  in stkpool.iterrows():
            stk = Market.Stk(row["amscode"])
            md = stk.MinuteData1    
            #md = row["amscode"].MinuteData1
            md.OnNewBar -= OnNewBar
    else:
        pass


def calcscore():
    ctc = datetime.now()
    ctc = ctc.replace(second=0, microsecond=0)
    if (ctc.hour > 8 and ctc.miniute > 45):
        # calc score
        pass
    else:
        pass

def OnNewBar(kdata,barNum):
    global count 
    scode = kdata.Stk.ServerCode
    kbarno = barNum
    sclose = kdata[barNum].Close
    temp = 0.0
    row = idk[scode]
    temp += (sclose - row["ma13"])
    temp += (sclose - row["ma55"])
    temp += (sclose - row["mom13"])
    temp += (sclose - row["mom21"])
    temp += (sclose - row["vwap"]) 
    temp = temp / row["changema"]
    calclist[scode]= [temp,barNum]
    count += 1
    calcscore()
    if count > 6:
        dftemp = pd.DataFrame(calclist, ignore_index = True)
        dftemp.to_csv("./out111.csv")
        
    endproc()
def init1():
    #before open, fetch yesterday positon
    #do some check if time is less than 9:30
    #some code
    print("before INIT")
    ypos = Strategy.All_Pos
    for pos in ypos:
        #status code 1: on hold; 2: pending cover; 0: closed
        oldpos.append(pos.ServerCode,pos.CostPrice,pos.CurrentQty,0)
    print("old position",ypos)
    print("load stock pool")
    stkpool = pd.read_csv("test2.csv")   
    stkpool["amscode"]=stkpool["amscode"].str.slice(0,7)
    #stkpool = stkpool.dropna()
    for idx, row  in stkpool.iterrows():        
        ssss = row["amscode"]
        stk = Market.Stk(row["amscode"])
        md = stk.MinuteData1
        md.OnNewBar += OnNewBar
        #col1 is calcu result, cal2 is barnum        
        calclist[row["amscode"]] = [0,0]
        idk[row["amscode"]] = row
        
    #stockpool = pd.read_csv(r"C:\app\pyapps\fetchstocks\test1.csv",index_col = 0)
if __name__ == '__main__':

    ct = datetime.now()
    ct = ct.replace(second=0, microsecond=0)
    while (ct.hour < 8 or (ct.hour == 8 and ct.miniute < 30)):
        time.sleep(360)
    print("before open")
    init1()
    dd = {}
    for s in stks:
        md = s.MinuteData1
        md.OnNewBar += OnNewBar
        dd[s.ServerCode] = s.DailyData
    print(dd["0600000"][10].DateTime)
    print(dd["0600000"].Count)    
