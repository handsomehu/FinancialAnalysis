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
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

jqd.auth("18621861857", "P4ssword")

sl = jqd.get_all_securities(['futures'])


s2 =sl.loc[sl.index.str.contains("8888")]

s2.to_csv("list.csv")
print(s2)
