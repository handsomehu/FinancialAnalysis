import jqdatasdk as jqd
from jqdatasdk import *
import pandas as pd
import numpy as np
import sys



jqd.auth("18621861857", "P4ssword")

stocks = get_index_stocks('000906.XSHG')


# 选出所有的总市值大于1000亿元, 市盈率小于10, 营业总收入大于200亿元的股票
df1 = get_fundamentals(query(
        valuation.code, valuation.market_cap, valuation.pb_ratio
    ).filter(
        valuation.code.in_(stocks)
    ).order_by(
        # 按市值降序排列
        valuation.code.desc()
    ).limit(
        # 最多返回100个
        10000
    ), statDate='2016')

df1.to_csv("./data/y2016basic.csv")

