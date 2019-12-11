'''
Created on Dec 10, 2019

@author: I038825
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
'''
according the random chart, you could see significant different between commodities
For examples, J and RB are normally far more better than A M, which is good I also think so

The by year charts has little outcome, I did not see much value out of them
'''

random = False
byyear = True

if random == True:
    df = pd.read_csv("./opt/noisebyrandom.csv",index_col = 0)    
    df = df.dropna()
    print(df)
    #symbol = "M8888.XDCE"
    #legends = ["20D","30D","40D","50D","60D"]
    #temp = df[df["symbol"]==symbol]
    #temp = temp[temp["duration"].isin(legends)]
    #print(temp)
    #fig = plt.subplots()
    dfm = pd.DataFrame(df[df["symbol"]=="L8888.XDCE"])
    ax = dfm.plot(x = "duration", y = "close", kind='line',legend = "M")
    ax.set_xlabel("duration")
    #ax.legend(['M'])

    #dfa = pd.DataFrame(df[df["symbol"]=="A8888.XDCE"])
    #bx = dfa.plot(x = "duration", y = "close", kind='line',legend = "A",ax = ax)
    #bx.set_xlabel("duration")
    #bx.legend(['A'])

    dfrm = pd.DataFrame(df[df["symbol"]=="PP8888.XDCE"])
    cx = dfrm.plot(x = "duration", y = "close", kind='line',legend = "RM",ax = ax)
    cx.set_xlabel("duration")

    #dfy = pd.DataFrame(df[df["symbol"]=="Y8888.XDCE"])
    #dx = dfy.plot(x = "duration", y = "close", kind='line',legend = "Y",ax = ax)  
    
    dfhc = pd.DataFrame(df[df["symbol"]=="TA8888.XZCE"])
    ex = dfhc.plot(x = "duration", y = "close", kind='line',legend = "Y",ax = ax)  
    ex.set_xlabel("duration")
        
    ax.legend(['RB','AG','AL'])  
    #ax.set_xlabel("duration")
    #ax = df.plot(x="duration", y=close, legend='w0')
    #du_offer.plot(x='max_load', y='w1', legend='w1', title=du, ax=ax)
    plt.show()
    #df.plot(x = "duration", y = "close", kind='line')
if byyear == True:
    df = pd.read_csv("./opt/noisebyyear.csv",index_col = 0)
    print(df) 
    df = df.dropna()
    df1 = pd.DataFrame(df[df["symbol"]=="J8888.XDCE"])
    legends = ["20D","30D","40D","50D","60D"]
    df1 = pd.DataFrame(df1[df1["duration"].isin(legends)])
    
    dfs0 = pd.DataFrame(df1[df1["duration"]==legends[0]])
    ax = dfs0.plot(x = "index", y = "close", kind='line',legend=True)
    plt.legend()
    dfs1 = pd.DataFrame(df1[df1["duration"]==legends[1]])
    ax1 = dfs1.plot(x = "index", y = "close", kind='line',title="RB",legend=True,ax = ax) 
    
    dfs2 = pd.DataFrame(df1[df1["duration"]==legends[2]])
    ax2 = dfs2.plot(x = "index", y = "close", kind='line',legend=True,ax = ax)
    
    dfs3 = pd.DataFrame(df1[df1["duration"]==legends[3]])
    ax3 = dfs3.plot(x = "index", y = "close", kind='line',legend=True,ax = ax)
    
    dfs4 = pd.DataFrame(df1[df1["duration"]==legends[4]])
    ax4 = dfs4.plot(x = "index", y = "close", kind='line',legend=True,ax = ax)
    ax.legend(legends) 
    plt.show()
    