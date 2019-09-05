'''
Created on Sep 5, 2019

@author: I038825
'''
import sqlite3
import pandas as pd

result = pd.read_csv("./pal_templt.csv")

def match_transaction(data):
    #print("hello")
    data["closestatus"] = 0 # 0 not process yet;1 match closed;2 partial match, partial close
    data["qty"] = data["orderqty"]
    #data["qty"]=data.apply(lambda r:judgeLevel(r),axis=1)
    #data[data["orderdirection"]=="空"]["qty"] = data[data["orderdirection"]=="空"]["qty"]*(-1)
    #data.to_csv("./test.csv")
    opens = data[data["orderoffset"]=="开仓"]
    closes = data[data["orderoffset"]=="平仓"]
    
    
    for index,row in opens.iterrows():
        #print(index, row["qty"])
        #continue

        cursymbol = row["contract"]
        otime = row["ordertime"] 
        odir = row["orderdirection"]
        oprice = row["entryprice"]
        oqty = row["qty"]
        #if odir == "空":
        #    oqty = oqty * -1    
        ctime = otime
        cprice = 0
        cstatus = 0
        cmatch = False  
        tomatch = oqty
        for ix,rw in closes.iterrows():
            partial = []
            pt = ("",0,0)
            if tomatch == oqty:
                if cursymbol == rw["contract"]  and otime < rw["ordertime"]:
                    if odir != rw["orderdirection"] and oqty == rw["qty"]:
                        ctime = rw["ordertime"]
                        cprice = rw["entryprice"]
                        cstatus = 1
                        cmatch = True
                        closes.drop(index = ix,inplace = True)
                        #del(ix)
                        break
                    elif odir != rw["orderdirection"] and oqty < rw["qty"]:
                        ctime = rw["ordertime"]
                        cprice = rw["entryprice"]
                        cstatus = 1
                        cmatch = True
                        closes[ix]["qty"] = rw["qty"] - oqty
                        #closes.del(ix)
                        break                    
                    
                    elif odir != rw["orderdirection"] and oqty > rw["qty"]:
                        
                        ctime = rw["ordertime"]
                        cprice = rw["entryprice"]
                        cstatus = 2
                        #cmatch = True 
                        pt = (ctime,cprice,rw["qty"])
                        tomatch = oqty - rw["qty"]   
                        partial.append(pt)
                        closes.drop(index = ix,inplace = True)
                        continue
                    else:
                        pass
            else:
                if cursymbol == rw["contract"]  and otime < rw["ordertime"]:
                    if odir != rw["orderdirection"] and tomatch == rw["qty"]:
                        ctime = rw["ordertime"]
                        cprice = rw["entryprice"]
                        cstatus = 1
                        cmatch = True
                        pt = (ctime,cprice,rw["qty"])
                        partial.append(pt)
                        closes.drop(index = ix,inplace = True)                 
                        break
                    elif odir != rw["orderdirection"] and tomatch < rw["qty"]:
                        ctime = rw["ordertime"]
                        cprice = rw["entryprice"]
                        cstatus = 1
                        cmatch = True
                        closes[ix]["qty"] = rw["qty"] - oqty
                        pt = (ctime,cprice,rw["qty"])
                        partial.append(pt)
                        #closes.del(ix)
                        break                    
                    
                    elif odir != rw["orderdirection"] and tomatch > rw["qty"]:
                        
                        ctime = rw["ordertime"]
                        cprice = rw["entryprice"]
                        cstatus = 2
                        #cmatch = True 
                        pt = (ctime,cprice,rw["qty"])
                        tomatch = tomatch - rw["qty"]   
                        partial.append[pt]
                        closes.drop(index = ix,inplace = True)
                        continue
                    else:
                        pass                                        
        oneline = dict(result.iloc[0])
        oneline["strategyname"] = row["strategyname"]
        oneline["contract"] = row["contract"]
        oneline["entryprice"] = row["entryprice"]
        oneline["orderoffset"] = row["orderoffset"]
        oneline["orderdirection"] = row["orderdirection"]  
        oneline["orderqty"] = row["orderqty"]  
        oneline["ordertime"] = row["ordertime"]  

                      
        
        if cmatch != True:
            print("there may have something wrong or open")
            oneline["closestatus"] = 0  
            #oneline["orderdirection"] = row["orderdirection"]              
        else:
            if len(pt) > 0:
                print(pt)
                cprice = 0
                tqty = 0
                for sub in pt:
                    print(sub)
                    #ctime = sub[0]
                    #cprice = cprice + sub[1]
                    #tqty = tqty + sub[2]
                if tqty != oqty:
                    print("Amount is not match, check!")
                cprice = cprice /len(pt)
            else:
                pass
            oneline["closetime"] = ctime
            oneline["closeprice"] = cprice
            #oneline["closestatus"] = 1
        result.append(oneline)      
                
                        
                    
                    
        

conn = sqlite3.connect('backupdb.db')



sqlstr = "select strategyname,upper(contract) as contract,entryprice,orderoffset,orderdirection,orderqty,ordertime from tradeorders where strategyname = :st order by  upper(contract) asc,ordertime  asc"

strlist = "select distinct strategyname from tradeorders"

c1 = conn.cursor()
c2 = conn.cursor()
c1 = c1.execute(strlist)
i = 0
for row in c1:
    #print(str(row[0]))
    data = pd.read_sql(sql=sqlstr, con=conn, params={'st': str(row[0])})
    #data.to_csv("./log.csv")
    
    match_transaction(data)
    #break

print(result)  
conn.close()  