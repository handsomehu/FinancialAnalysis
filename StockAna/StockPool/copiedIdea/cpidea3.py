'''
Created on May 9, 2019

@author: I038825
'''
import scipy.optimize as sco
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import tushare as ts
from sklearn.preprocessing import Imputer
import sys, os



    

    
def get_data_roe(theyear):
    data=ts.get_report_data(year=theyear,quarter=4)
    data = data.drop_duplicates('code')
    data['Year'] = theyear 

def combine_data():
    

    df_init = get_data_roe(2016)
    
    for i in range(2017,2019):

        df_temp = get_data_roe(i)
        df_init = df_init.append(df_temp, ignore_index=True)
        
    print(df_init.head(10))
    df_init.to_csv("./data/TYReport.csv")

combine_data()
sys.exit()        

'''    
    cr = pd.read_csv("./data/CashRatio.csv",index_col= "code")
    qr = pd.read_csv("./data/QReport.csv",index_col= "code")
    
    print(qr.head(10))
    del qr["Unnamed: 0"]
    del cr["Unnamed: 0"]
    qr = qr.join(cr, on='code', how='left',  rsuffix='_right')
    qr = qr.drop_duplicates('name')
    #qr.merge(cr, on='code', how='left')
    print(qr.head(10))
    qr.head(10).to_csv("tmp.csv")
    bft = qr["cashratio"].str.contains('--')

    qr = qr[~bft]
    qr["fcr"] = qr["cashratio"].astype(float)
    #bft = qr["roe"].str.contains('--')
    #qr = qr[~bft]

    #bft =  qr["eps"].str.contains('--')
    #qr = qr[~bft]

        
    bft =  qr["name"].str.contains('st')
    qr = qr[~bft] 
    bft =  qr["name"].str.contains('ST')
    qr = qr[~bft] 
    

       
    #qr = qr.dropna()
    print(qr["cashratio"])
    return qr
    print(qr.head(100))
    
#df1=df1[df1.per >0]



data = read_data()
print(len(data))
data = data[data["fcr"]>20]
print(len(data))
data = data[data["eps"]>0]
print(len(data))
data = data[data["roe"]>1]
print(len(data))

data.sort_values(by="roe",ascending= False)  
data.to_csv("./data/filtered_2019q1.csv")

'''