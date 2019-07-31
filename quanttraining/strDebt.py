'''
Created on Jun 3, 2019

@author: I038825
do
'''
#Module Imports
import numpy as np
import numpy.random as npr
import math
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats
import xlwings as xw

'''DEFINITION OF VARIABLES
    V0 - Total Firm Value at T=0
    D - Debt Value at T = T
    T - Time in Years
    R - Risk Free Rate
    SIGMA - Volatility of Firm Value
    DT - Time Step = T/N
    DF - Discount Factor = e^-RT
    I - Number of Simulations
    alpha - adjustment factor for curvature of barrier
'''

V0 = 100
alpha = 1.25

D=70
T=4
R=0.05
SIGMA=0.20
I = 1000
N=252*T