'''
Created on Dec 2, 2019

@author: I038825
'''
import pandas as pd
import sys
calclist = {}
calclist["a"]= [1,2,3]
calclist["b"]= [1,2,3]
calclist["c"]= [1,2,3]
print(calclist)
df = pd.DataFrame(calclist)
print(df)
sys.exit(0)

oldpos = {"1000717":[1,5,2,0]}
openpos = {"1000581":[1,1,0,0]}# buy sell/vol/count minutes/count retry
#opencountdown = {"1000581":[0,0]}#count minutes and count retry
sellcnt = 0
lv_newsize = 0
lv_size = 5
df = pd.read_csv("./data/test1.csv",index_col = 0)
df = df.dropna()
df = df[df["close"] < 20]
print(len(df))
print(df)
w = [0.15,0.3,0.15,0.2,0.2]
df["score"] = (df["close"] - df["ma13"])*w[0]
df["score"] = df["score"] +(df["close"] - df["ma55"])*w[1]
df["score"] = df["score"] +(df["close"] - df["mom13"])*w[2]
df["score"] = df["score"] +(df["close"] - df["mom21"])*w[3]
df["score"] = df["score"] +(df["close"] - df["vwap"])*w[4]
df["score"] = df["score"]/df["changema"]
df["amscode"] = df["amscode"].str.slice(0,7)

#df = df.sort_index(ascending=False, by = ["score"],inplace=False)

df = df.sort_values(by = ["score"],  ascending = False)
df = df.reset_index(drop=True)
df.to_csv("./score.csv")

def calcscore(data):
    return data
def updatepos():
    tempscore = 0
    allstk = set()
    score = calcscore(df)   
    if  len(score) > 0:
        tempscore = score.iloc[4][1]
        for ypos in oldpos:
            allstk.add(ypos)
            if tempscore*0.85 - oldpos[ypos][1] > 0 :
                pass#code to sell this position
                pass#code to add cover order in onhold list
        sellcnt =0    
        for opos in openpos:
            allstk.add(opos)
            if openpos[opos][0] == -1:
                sellcnt += 1
            if openpos[opos][2] < 7:
                openpos[opos][2] += 1
            elif openpos[opos][3] <2:
                openpos[opos][3]+= 1
                openpos[opos][2] = 0
            elif openpos[opos][0] == 1:
                pass # cancel order and get rid of this stock  
            elif openpos[opos][0] == -1:
                openpos[opos][2]=0
                openpos[opos][3]=0
                #cancel this order and send new order with new price
                pass # change order price and reset
            else:
                print("maybe something wrong")
        
              
        lv_newsize = lv_size + sellcnt - len(openpos)
        print(allstk)
        for i in range(10):
            if lv_newsize > 0:
                if score.iloc[i]["amscode"] in allstk:
                    pass
                else:
                    #trigger order
                    print("order",score.iloc[i]["amscode"])
                    lv_newsize = lv_newsize - 1;        
            else:
                break     
                 
            
        pass
    else:
        pass

updatepos()