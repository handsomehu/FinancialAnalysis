'''
Created on Dec 5, 2019

@author: I038825
'''
import pandas as pd
import numpy as np
from  matplotlib import pyplot as plt
#import random

sampleNo = 10000;
# 一维正态分布
# 下面三种方式是等效的
mu = 0
sigma = 1
np.random.seed(0)
s = np.random.randn(sampleNo)#(mu, sigma, sampleNo ) 
s = s
#plt.subplot(141)
plt.hist(s, 20, normed=True)
plt.show()
df = pd.DataFrame(s)
print(df.skew())
print(df.kurt())