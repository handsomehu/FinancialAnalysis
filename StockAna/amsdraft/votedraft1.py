import sys
import os
import time
import threading
import ctypes
import pandas as pd
from datetime import datetime,timedelta

todaypos = {}
#oldpos = {"1000717":[1,5,100,0,0]}  #buy sell/price/vol/bar countdown/retry no
#openpos = {"1000581":[1,6,100,0,0]} #buy sell/price/vol/bar countdown/retry
oldpos = {}
openpos = {}
calclist = {}
idk = {}
stkpool = pd.DataFrame()
#stks = Market.Stk(['0600000'])
count = 0
totalcnt = 0
lv_size = 10
sellcnt = 0
lv_newsize = 0

def endproc():
    cte = datetime.now()
    cte = cte.replace(second=0, microsecond=0)
    if (cte.hour >14 and cte.minute > 10):
        for idx, row  in stkpool.iterrows():
            stk = Market.Stk(row["amscode"])
            md = stk.MinuteData1    
            #md = row["amscode"].MinuteData1
            md.OnNewBar -= OnNewBar
            print("end")
    else:
        pass
def calcscore(calcdata):
    global count
    global totalcnt
    totalcnt = len(calcdata)
    ctc = datetime.now()
    ctc = ctc.replace(second=0, microsecond=0)
    df = pd.DataFrame()
    if (ctc.hour > 9 or (ctc.hour == 9 and ctc.minute > 45)):
        if count%40 == 0:
            pass
            #print("count,totalcnt"+str(count)+","+str(totalcnt))
        if count == (totalcnt - 10):
            print("1min bar near finish")
        if count > 180:
            print("count:"+str(count))
        if count >= totalcnt:
            print("bigger"+str(count))
            #df = pd.DataFrame(calcdata)
            list1 = []
            for k in calcdata:
                list1.append([k,calcdata[k][0],calcdata[k][1]])
                
            tempdf = pd.DataFrame(list1,columns=["code","score","kbar"])
            
            tempdf.sort_values(by=["score"], axis=0, ascending= False, inplace=True)
            tempdf = tempdf.reset_index(drop=True)
            df = tempdf
            print("ok, reset count!")
            count = 0
        else:
            pass
    else:
        pass
    
    return df
    
def updatepos():
    global calclist
    global openpos
    global oldpos
    global todaypos
    global count
    tempscore = 0
    allstk = set()
    score = calcscore(calclist)
    if len(score) % 50 == 0:
        #print("score len",len(score))
        pass        
    if  len(score) > 0:
        print("in or not?")
        tempscore = score.iloc[4][1]
        print(tempscore)
        for ypos in oldpos:
            allstk.add(ypos)
            if tempscore*0.85 - calclist[oldpos[ypos]][0] > 0 :
                
                Strategy.Order(OrderItem(ypos, 'S', oldpos[ypos][2], calclist[ypos][2]+0.02)) 
                openpos[ypos] = [-1,calclist[ypos][2]+0.02,oldpos[ypos][2],0,0]
                #pass#code to sell this position
                #pass#code to add cover order in onhold list
            #if score less than zero, cover the position
            if calclist[oldpos[ypos]][0] < 0:
                Strategy.Order(OrderItem(ypos, 'S', oldpos[ypos][2], calclist[ypos][2]+0.02)) 
                openpos[ypos] = [-1,calclist[ypos][2]+0.02,oldpos[ypos][2],0,0]            
        global sellcnt
        sellcnt = 0    
        for opos in openpos:
            allstk.add(opos)
            if openpos[opos][0] == -1:
                sellcnt += 1
            if openpos[opos][3] < 7:
                openpos[opos][3] += 1
            elif openpos[opos][4] <2:
                openpos[opos][4]+= 1
                openpos[opos][3] = 0
            elif openpos[opos][0] == 1:
                ords = Strategy.All_Order()
                for ord in ords:
                    if ord["ServerCode"] == opos and ord["BsType"] == 'B':
                        Strategy.WithDraw(ord)
                        openpos = openpos.pop(opos)
                        calclist = calclist.pop(opos)
                        stkpool = stkpool[stkpool["amscode"]!=opos] 
            elif openpos[opos][0] == -1:
                for ord1 in ords1:
                    if ord1["ServerCode"] == opos and ord1["BsType"] == 'S':
                        Strategy.WithDraw(ord1)
                        Strategy.Order(OrderItem(opos, 'S',openpos[ypos][2], calclist[ypos][2]-0.02))
                        openpos[opos][3]=0
                        openpos[opos][4]=0
            else:
                print("maybe something wrong")
        
              
        lv_newsize = lv_size + sellcnt - len(openpos)-len(todaypos)
        for i in range(10):
            if lv_newsize > 0:
                if score.iloc[i][0] in allstk:
                    pass
                else:
                    #trigger order
                    if calclist[score.iloc[i][0]][0] > 0:
                        buyqty = 200
                        if calclist[score.iloc[i][0]][2] > 10:
                            buyqty = buyqty /2
                        Strategy.Order(OrderItem(score.iloc[i][0], 'B',buyqty, calclist[score.iloc[i][0]][2]+0.02))
                        openpos[score.iloc[i][0]] = [1,calclist[score.iloc[i][0]][2]+0.02,buyqty,0,0]
                        print("order",score.iloc[i][0])
                        lv_newsize = lv_newsize - 1
                    else:
                        pass#all stock weak, do not buy                    
            else:
                #clean and exit
                break     
                 
            
        pass
    else:
        pass
    
def OnNewBar(kdata,barNum):
    global count
    global calclist
    w = [0.15,0.3,0.15,0.2,0.2] 
    scode = kdata.Stk.ServerCode
    kbarno = barNum
    sclose = kdata[barNum].Close
    temp = 0.0
    row = idk[scode]
    temp += (sclose - row["ma13"])*w[0]
    temp += (sclose - row["ma55"])*w[1]
    temp += (sclose - row["mom13"])*w[2]
    temp += (sclose - row["mom21"])*w[3]
    temp += (sclose - row["vwap"]) *w[4]
    temp = temp / row["changema"]
    calclist[scode]= [temp,barNum,sclose]   
    updatepos()       
    endproc()
    count += 1

def init1():
    #before open, fetch yesterday positon
    #do some check if time is less than 9:30
    #some code
    print("before INIT")
    ypos = Strategy.All_Pos
    for pos in ypos:
        #buy sell/price/vol/bar countdown/retry no       
        oldpos[pos.ServerCode] = [1,CostPrice,CurrentQty,0,0] #buy sell/
        #.append(pos.ServerCode,CostPrice,CurrentQty,0)
    print("old position",ypos)
    print("load stock pool")
    stkpool = pd.read_csv("test1.csv")   
    stkpool["amscode"]=stkpool["amscode"].str.slice(0,7)
    stkpool=stkpool[stkpool["close"]<20]
    totalcnt = len(stkpool)
    print("total cnt"+str(totalcnt))
    for idx, row  in stkpool.iterrows():        
        stk = Market.Stk(row["amscode"])
        md = stk.MinuteData1
        md.OnNewBar += OnNewBar
        #col1 is calcu result, cal2 is barnum        
        calclist[row["amscode"]] = [0,0]
        idk[row["amscode"]] = row
def clearstock(stk):
    calclist = calclist.pop(stk)
    stkpool = stkpool[stkpool["amscode"]!=stk]
    mstk = Market.Stk(stk)
    md = mstk.MinuteData1
    md.OnNewBar -= OnNewBar
         
def OnRtsChanged(Rts):
    for rt in Rts:
        optype = rt.BSType
        opstk = rt.ServerCode
        curopen = openpos[opstk]
        if optype == "S" and (rt.StatusCode == "Fully_Filled" or rt.StatusCode == "Partially_Filled"):
            #clear this stock, do not operate it anymore
            clearstock(opstk)
            pass
        elif optype == "B" and (rt.StatusCode == "Fully_Filled" or rt.StatusCode == "Partially_Filled"):
            #add to today order and clear this stock
            if rt.KnockPrice > 10:
                qqq = 100
            else:
                qqq = 200
            todaypos[opstk] = [1,rt.KnockPrice,qqq,0,0]
            clearstock(opstk)
            pass
        else:
            pass
            
if __name__ == '__main__':

    ct = datetime.now()
    ct = ct.replace(second=0, microsecond=0)
    while (ct.hour < 8 or (ct.hour == 8 and ct.minute < 30)):
        time.sleep(360)
    print("before open")
    init1()
    Strategy.RtsChanged += OnRtsChanged
    #md.OnNewBar += OnNewBar    
