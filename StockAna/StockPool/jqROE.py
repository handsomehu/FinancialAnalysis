import jqdatasdk as jqd
from jqdatasdk import *
import pandas as pd
import numpy as np
import sys



jqd.auth("18621861857", "P4ssword")



# 选出所有的总市值大于1000亿元, 市盈率小于10, 营业总收入大于200亿元的股票
df1 = get_fundamentals(query(
        valuation.code, valuation.market_cap, valuation.pe_ratio, valuation.pb_ratio,indicator.roe
    ).filter(
        valuation.market_cap < 500,
        valuation.pe_ratio > 0,
        indicator.roe > 0
    ).order_by(
        # 按市值降序排列
        valuation.code.desc()
    ).limit(
        # 最多返回100个
        10000
    ), statDate='2016')

df1.to_csv("y2016.csv")


df2 = get_fundamentals(query(
        valuation.code, valuation.market_cap, valuation.pe_ratio, valuation.pb_ratio,indicator.roe
    ).filter(
        valuation.market_cap < 500,
        valuation.pe_ratio > 0,
        indicator.roe > 0
    ).order_by(
        # 按市值降序排列
        valuation.code.desc()
    ).limit(
        # 最多返回100个
        10000
    ), statDate='2017')

df2.to_csv("y2017.csv")

df3 = get_fundamentals(query(
        valuation.code, valuation.market_cap, valuation.pe_ratio, valuation.pb_ratio,indicator.roe
    ).filter(
        valuation.market_cap < 500,
        valuation.pe_ratio > 0,
        indicator.roe > 0
    ).order_by(
        # 按市值降序排列
        valuation.code.desc()
    ).limit(
        # 最多返回100个
        10000
    ), statDate='2018')

df3.to_csv("y2018.csv")