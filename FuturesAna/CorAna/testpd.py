import pandas_datareader.data as web
import pandas as pd
import numpy as np

all_data = {ticker: web.get_data_yahoo(ticker) for ticker in ['AAPL', 'IBM', 'MSFT', 'GOOG']} 
#print(all_data)


price = pd.DataFrame({ ticker: data['Adj Close'] for ticker, data in all_data.items()}) 
print(price.head(10))
volume = pd.DataFrame({ ticker: data['Volume'] for ticker, data in all_data.items()})
print(volume)


