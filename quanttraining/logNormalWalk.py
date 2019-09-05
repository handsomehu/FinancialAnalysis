'''
Created on Jun 3, 2019

@author: I038825
reference article:https://medium.com/fintechexplained/monte-carlo-simulation-engine-in-python-a1fa5043c613
'''
# -*- coding: utf-8 -*-  
import numpy as np
from matplotlib import pyplot as plt
import math
import pandas as pd

np.random.seed(18)

#Variables
S = 100  #Beginning Stock Price
Sigma = 0.20  #Volatility
r = 0.03  #risk free rate
T = 1  #TimePeriod in Years
dt = 1/252  #timestep  1/252 is daily using trading days

#Generate Phi Values
x = np.random.standard_normal(int(T/dt))

plt.plot(x)
plt.show()


#Calculation of Phi average - should be approximately zero
mean = np.mean(x)
print(mean)


#Calculation of Phi standard deviation - should be approximately 1
stdev = np.std(x)
print(stdev)

#Generation of single realization of the log normal random walk
walk = np.zeros_like(x)

walk[0]=S

for step in range(1,252):
    walk[step] = walk[step-1] * math.exp( (r-(.5*Sigma**2))*dt + (Sigma*x[step]*math.sqrt(dt)) )
    #                                         ________________why use average rather than sqrt of dt?
    #                                         this is Ito's Lemma which i am not quite understand
    #            S(t) = S(0) exp[(r-0.5σ²)t + σ√tε)]

plt.plot(walk)
plt.show()

paths = 100  #set number of realizations to calculate

total_steps = 252*T
simulation = np.zeros((paths, total_steps))  #Setup empty numpy array to hold data
FV = np.zeros(paths)  #Setup empty numpy array to hold ending values of simulations

for sim in range(0,paths):
    simulation[sim,0] = S  #set V0 to S
    phi = np.random.standard_normal(int(T*252))  #Generate unique random numbers for each simulation
    for step in range(1,int(252*T)):
        simulation[sim,step] = simulation[sim,step-1] * math.exp( (r-(.5*Sigma**2))*dt + (Sigma*phi[step]*math.sqrt(dt)) )
        FV[sim] = simulation[sim,total_steps-1]

plt.plot(np.transpose(simulation))
plt.show()

#Average of simulation ending values
sim_avg = np.mean(FV)
print(sim_avg)
#Standard Deviation of ending values
sim_std = np.std(FV)
print(sim_std)
