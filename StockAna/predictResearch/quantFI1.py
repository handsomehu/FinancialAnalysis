'''
Created on May 17, 2019

@author: I038825
'''
# libraries ----
import pandas as pd 
import numpy as np
import quandl
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import datetime as dt 

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')


sns.set_style('darkgrid')
sns.mpl.rcParams['figure.figsize'] = (10.0, 6.0)
sns.mpl.rcParams['savefig.dpi'] = 90
sns.mpl.rcParams['font.size'] = 14

# authentication ----
quandl_key = 'zs7xyLhXJbVU_Sk2-4aB' # paste your own API key here :)
quandl.ApiConfig.api_key = quandl_key


# downloading the data 
df = quandl.get('WIKI/MSFT', start_date="2000-01-01", end_date="2017-12-31")
df = df.loc[:, ['Adj. Close']]
df.columns = ['adj_close']

# create simple and log returns, multiplied by 100 for convenience
df['simple_rtn'] = 100 * df.adj_close.pct_change()
df['log_rtn'] = 100 * (np.log(df.adj_close) - np.log(df.adj_close.shift(1)))

# dropping NA's in the first row
df.dropna(how = 'any', inplace = True)

print(df.head())


# Plotting the time series ----
fig, ax =plt.subplots(3, 1, figsize=(24, 20))
# price ----
df.adj_close.plot(ax=ax[0])
ax[0].set_ylabel('Stock price ($)')
ax[0].set_xlabel('')
ax[0].set_title('Price vs. returns')
# simple returns ----
df.simple_rtn.plot(ax=ax[1])
ax[1].set_ylabel('Simple returns (%)')
ax[1].set_xlabel('')
# log returns ----
df.log_rtn.plot(ax=ax[2])
ax[2].set_ylabel('Log returns (%)')
fig.show()



# Plotting the distribution of the returns ----
ax = sns.distplot(df.log_rtn, kde = False, norm_hist=True)                                    

xx = np.linspace(min(df.log_rtn), max(df.log_rtn), num=1000)                                                  
yy = scs.norm.pdf(xx, loc=df.log_rtn.mean(), scale=df.log_rtn.std())                                                         
ax.plot(xx, yy, 'r', lw=2)
ax.set_title('Distribution of MSFT returns')

# QQ plot ----
qq = sm.qqplot(df.log_rtn.values, line='s')
qq.show()



# Descriptive statistics ----
print('Range of dates:', min(df.index.date), '-', max(df.index.date))
print('Number of observations:', df.shape[0])
print('Mean: {0:.4f}'.format(df.log_rtn.mean()))
print('Median: {0:.4f}'.format(df.log_rtn.median()))
print('Min: {0:.4f}'.format(df.log_rtn.min()))
print('Max: {0:.4f}'.format(df.log_rtn.max()))
print('Standard Deviation: {0:.4f}'.format(df.log_rtn.std()))
print('Skewness: {0:.4f}'.format(df.log_rtn.skew()))
print('Kurtosis: {0:.4f}'.format(df.log_rtn.kurtosis())) #Kurtosis of std. Normal dist = 0
print('Jarque-Bera statistic: {stat:.2f} with p-value: {p_val:.2f}'.format(stat = scs.jarque_bera(df.log_rtn.values)[0],
                                                                           p_val = scs.jarque_bera(df.log_rtn.values)[1]))
# Autocorrelation plot of log returns ----
acf_r = smt.graphics.plot_acf(df.log_rtn, lags=40 , alpha=0.5)
acf_r.show()

#sns.distplot(df["log_rtn"], hist=True)
plt.show()

# specify the max amount of lags
lags = 40

fig, ax =plt.subplots(3, 1, figsize=(24, 20))
# price ----
smt.graphics.plot_acf(df.log_rtn, lags=lags , alpha=0.5, ax = ax[0])
ax[0].set_ylabel('Returns')
ax[0].set_title('Autocorrelation Plots')
# simple returns ----
smt.graphics.plot_acf(df.log_rtn ** 2, lags=lags, alpha=0.5, ax = ax[1])
ax[1].set_ylabel('Squared Returns')
ax[1].set_xlabel('')
ax[1].set_title('')
# log returns ----
smt.graphics.plot_acf(np.abs(df.log_rtn), lags=lags, alpha=0.5, ax = ax[2])
ax[2].set_ylabel('Absolute Returns')
ax[2].set_title('')
ax[2].set_xlabel('Lag')
fig.show()

plt.show()
