"""
See http://qoppac.blogspot.co.uk/2015/11/using-random-data.html for more examples
by Leon, Steal from Robert github to generate random price
"""


from systematictrading_examples.rob_org.common import arbitrary_timeseries
from systematictrading_examples.rob_org.commonrandom import  generate_trendy_price
from matplotlib.pyplot import show
import random
import pandas as pd

def shiftlist(lst, k):

    for i in range (k):
        pd_temp = lst.pop()
        lst.insert(pd_temp)
    return lst    
    pl = pd.DataFrame(lst[k:],columns = ['close'])
    print(pl)
    pr = pd.DataFrame(lst[:k],columns = ['close'])
    print(pr)
    return pd.concat(pl,pr)
def higher(o,c):
    if o > c:
        return  o+ o*0.01*random.random()
    else:
        return c+ c*0.01*random.random()
def lower(o,c):
    if o < c:
        return  o- o*0.01*random.random()
    else:
        return c- c*0.01*random.random()
def calckpi(df1):

    p1 = 13
    p2 = 21
    p3 = 55
    ma1 = "ma"+str(p1)
    ma2 = "ma"+str(p2)
    ma3 = "ma"+str(p3)
    mom1 = "mom"+str(p1)
    mom2 = "mom"+str(p2)
    mom3 = "mom"+str(p3)
    df1["open"] = df1["close"].shift(-1)
    df1["high"] = df1.apply(lambda x: higher(x.open, x.close), axis = 1)
    #["open"]+ df1["close"]*0.01*random.random()
    df1["low"] = df1.apply(lambda x: lower(x.open, x.close), axis = 1)
    df1[ma1] = df1["close"].shift(p1*-1)
    df1[ma2] = df1["close"].shift(p2*-1)
    df1[ma3] = df1["close"].shift(p3*-1)
    df1[mom1] = df1["close"].shift(p1*-1)
    df1[mom2] = df1["close"].shift(p2*-1)
    df1[mom3] = df1["close"].shift(p3*-1)
    df1[mom3] = df1["close"].shift(p3*-1)
    df1[mom3] = df1["close"].shift(p3*-1)
    df1["vwap"] = df1["close"]
    df1["change"] = df1["close"]-df1["close"].shift(-1)
    df1 = df1.dropna()
    df1= df1[["datetime","open","high","low","close","change",ma1,ma2,ma3,mom1,mom2,mom3,"vwap"]]
    df1["change"] = abs(df1["change"])
    df1 = df1.dropna()
    #print(df1)
    
    #df1 = df1.sort_index( ascending=False)
    #print(df1)
    df1["changema"] = df1["change"].rolling(20 ).mean()
    df1 = df1.dropna()
    df1 = df1.reset_index(drop = True)
    #print(df1)
    #df1 = df1.sort_index( ascending=True)
    #df1 = df1.drop(index)
    #df1 = df1[:1]
    #print(df1)
    #print(df1)
    return df1

lv_cnt = 31
lv_vol = 0
lv_seed = 6
lv_trend = 60

lv_open = 5000

    
for i in range(lv_cnt):
    lv_trend_i = int( lv_trend*(1+random.random()+random.random()))
    
    if lv_trend_i < 10:
        lv_trend_i = 20 + lv_trend_i
    lv_seed_i = 3+int(lv_seed*random.random())
    #print(lv_seed_i)

    lv_vol = 0.1+ random.random()/10
    #lv_trend_i = 60
    ans=arbitrary_timeseries(generate_trendy_price(Nlength=900, Tlength=lv_trend_i, Xamplitude=lv_seed_i, Volscale=lv_vol))    
    #print(ans)
    #ans = list(ans)
    #ans = shiftlist(ans, int(lv_trend_i*random.random()))  
    #print(ans)   
    #ans = pd.DataFrame(ans,columns = ["close"])
    #lv_min = ans.min()    

    #print(ans)
    #ans = ans.values()
    #ans = pd.DataFrame([ans],columns = ["datetime","close"])
    #print(type(ans))
    ans = ans.to_frame().reset_index()
    ans.columns = ["datetime","close"]
    lv_open_i = 2+ lv_open * random.random()
    ans["close"] = 1+ ans["close"].cumsum()/100
    ans["close"] = ans["close"]*lv_open
    #print(ans)
    #ans["close"].plot()
    #show()
    #continue
    ans = calckpi(ans)
    ans = ans.to_csv("../data/random"+str(i)+".csv")

    #ans["ma13"]
    #print(ans)
    #break
    
    #ans.plot()
    #show()
