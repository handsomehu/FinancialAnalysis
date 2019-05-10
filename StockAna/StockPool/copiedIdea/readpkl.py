'''
Created on May 9, 2019

@author: I038825
'''
import pandas as pd
import numpy as np
import pickle
import sys

with open('./data/final_data.pkl','rb') as f:
    data=pickle.load(f)
print(data)

sys.exit(0)
with open('./data/final_data_pandas.pkl','rb') as f:
    data=pickle.load(f)
data.to_csv("./data/final_data_pandas_pkl.csv")