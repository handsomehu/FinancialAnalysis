# -*- coding: utf-8 -*-
'''
今天主要是实现了公司偿债能力指标的量化，构建股票池。
思路:
1.
偿债能力量化指标(短期)
相关指标:
    流动比率=流动资产/流动负债
    速动比率=（流动资产-存货）/流动负债
    现金比率=现金类资产/流动负债

2.
确定偿债能力理论值
流动比率，速动比率，现金比率给定理论值
    2.1:
    求各个行业股票的均值作为各个指标的理论值
--------------------- 
作者：春天的期待 
来源：CSDN 
原文：https://blog.csdn.net/weixin_41866806/article/details/81196812 
版权声明：本文为博主原创文章，转载请附上博文链接！
'''
# copy from csdn, I thought the idea is good and may use it to build the pool
import scipy.optimize as sco
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import tushare as ts
from sklearn.preprocessing import Imputer
import sys, os

def save_data():
    data=ts.get_debtpaying_data(year=2018,quarter=4)
    with open('./data/data_debt.pkl','wb') as f:
        pickle.dump(data,f)
def get_data():
    if not os.path.exists('./data/data_debt.pkl'):
        save_data()
    with open('./data/data_debt.pkl','rb') as f:
        data=pickle.load(f)
    return data

def data_deal():
    data=get_data()
    # print(data.columns)
    data_array=data.values

    # 删除具有缺少值的股票,以及删除*st与st的股票
    x_data=[]
    for stock in data_array:
        if '--' not in stock and stock[1][:3]!='*ST' and stock[1][:2]!='ST':
            x_data.append(stock)
    new_data=np.array(x_data)

    # 将流动比率，速动比率，现金比率拿出来,numpy.adarry格式的
    new_data=new_data[:,:5]
    #行业分类表格，上次出错就在这里
    data_industry=ts.get_industry_classified()
    # 删除重复出现的代码，一个股票只能是一个行业
    data_industry=data_industry.drop_duplicates('code')
    data_industry=pd.DataFrame(data_industry.values[:,-1],index=data_industry['code'],columns=['c_name'])
    #按照负债表得到的code,再依据行业分类表格，对负债表中的股票进行行业分类
    w=[]
    for code in new_data:
        try:
            w.append(data_industry['c_name'][code[0]])
        except:
            #对行业分类缺失的股票认定为：其他行业
            w.append('其他行业')
    final_data=np.hstack((new_data,np.array(w).reshape(-1,1)))
    # 负债表中的股票分好类，转换为DataFrame形式的
    final_data=pd.DataFrame(final_data,columns=['code','name','currentratio','quickratio','cashratio','c_name'])
    # print(final_data)
    final_data=pd.DataFrame(np.hstack((final_data.values[:,:2],final_data.values[:,2:5].astype('f4'),final_data.values[:,-1].reshape(-1,1))),columns=['code','name','currentratio','quickratio','cashratio','c_name'])
    # 求均值，每个行业的三个指标的均值，放在字典里面{'电器行业':[1,23,2],....}
    data_dic={}
    for i in final_data.groupby(final_data['c_name']):
        cur_mean,quic_mean,cash_mean=i[1]['currentratio'].values.mean(),i[1]['quickratio'].values.mean(),i[1]['cashratio'].values.mean()
        data_dic[i[0]] = [cur_mean,quic_mean,cash_mean]
    # 按照每只股票所属行业，根据行业指标字典，将三个指标的行业均值放入进去.
    M=[]
    for c_name in final_data['c_name']:
        M.append(data_dic[c_name])
    final_data=np.hstack((final_data.values,np.array(M)))
    with open('./data/final_data.pkl','wb') as g:
        pickle.dump(final_data,g)
    for stock in final_data:
        name=str(stock[0])+'\t'+str(stock[1])+'\t'+str(stock[2])+'\t'+str(stock[3])+'\t'+str(stock[4])+'\t'+str(stock[5])+'\t'+str(stock[6])+'\t'+str(stock[7])+'\t'+str(stock[8])+'\n'
        with open('./data/data_deal.txt','a') as f:
            if stock[0]==final_data[0][0]:
                f.write('code'+'\t'+'name'+'\t'+'currentratio'+'\t'+'quickratio'+'\t'+'cashratio'+'\t'+'c_name''\t'+'currentratio_mean'+'\t'+'quickratio_mean'+'\t'+'cashratio_mean'+'\n')
            f.write(name)
    return final_data




'''
3.
确定单项量化指标
tar1=(流动比率-理论值)/理论值
tar2=(速动比率-理论值)/理论值
tar3=(现金比率-理论值)/理论值

4.
确定偿债能力量化指标(加权)
final_tar=1/3*tar1+1/3*tar2+1/3*tar3
'''

# 建立股票池
def get_target(final_data):
    # 3.确定单项量化指标
    # tar1=(流动比率-理论值)/理论值
    # tar2=(速动比率-理论值)/理论值
    # tar3=(现金比率-理论值)/理论值
    final_data[:,2]=(final_data[:,2]-final_data[:,6])/final_data[:,6]
    final_data[:,3]=(final_data[:,3]-final_data[:,7])/final_data[:,7]
    final_data[:,4]=(final_data[:,4]-final_data[:,8])/final_data[:,8]
    # 确定偿债能力量化指标(加权)
    # final_tar=1/3*tar1+1/3*tar2+1/3*tar3
    final_data[:,8]=(final_data[:,2]+final_data[:,3]+final_data[:,4])*1/3
    for i in final_data:
        print(i)
    # 提取股票的代码以及名称
    final_data=np.hstack((final_data[:,:2],final_data[:,-1].reshape(-1,1)))
    final_data_pandas=pd.DataFrame(final_data,columns=['code','name','debt_values'])
    with open('./data/final_data_pandas.pkl','wb') as g:
        pickle.dump(final_data_pandas,g)
    for yangben in final_data_pandas.values:
        name=str(yangben[0])+'\t'+str(yangben[1])+'\t'+str('%.3f'%yangben[2])+'\n'
        with open('./data/the_result.txt','a') as f:
            if yangben[0]==final_data_pandas.values[0][0]:
                f.write('code'+'\t'+'name'+'\t'+'debt_values'+'\n')
            f.write(name)
    return final_data_pandas


def save_stocks(final_data_pandas):
    # 按照综合指标从小到大进行排序
    final_data_sort=final_data_pandas.sort_values(by='debt_values')
    with open('./data/final_data_sort.pkl','wb') as f:
        pickle.dump(final_data_sort,f)
    num=0
    for yangben in final_data_sort.values:
        num+=1
        name=str(yangben[0])+'\t'+str(yangben[1])+'\t'+str('%.3f'%yangben[2])+'\n'
        with open('./data/the_result.txt','a') as f:
            if yangben[0]==final_data_sort.values[0][0]:
                f.write('code' + '\t' + 'name' + '\t' + 'debt_values' + '\n')
            f.write(name)
        if num==500:
            break



def get_data_sqr():
    with open('./data/final_data_pandas.pkl', 'rb') as f:
        data_stockcode = pickle.load(f)
    # data_stockcode = pd.read_excel('results.xlsx')
    data = pd.DataFrame()
    for code in data_stockcode['code']:
        try:
            origin_data = ts.get_hist_data(code=str(code), start='2017-01-01', end='2017-7-31', ktype='D')
            data[str(code)] = origin_data['close']
        except:
            print(str(code))
    # print(data.isnull().sum())
    new_data = data.fillna(method='pad')
    with open('./data/new_data.pkl','wb') as f:
        pickle.dump(new_data,f)
    return new_data

def get_comb(new_data):
    # print(new_data.describe())
    # print(new_data.isnull().sum())
    noa=int(new_data.shape[1])
    returns = np.log(new_data / new_data.shift(1))
    variables=returns.cov()*noa
    # print(returns.head())
    # 夏普指数的负值最大化,
    def min_shar(weights):
        x_mean=(returns.mean().dot(weights.T))*noa
        x_variable=np.sqrt(np.dot(weights.T,np.dot(variables,weights)))
        return -1*(x_mean/x_variable)
    weights_begin=np.random.random_sample(noa)
    # 增加限制条件，权重总和是1
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # 权重限制在0,1之间
    bnds = tuple((0, 1) for x in range(noa))
    ops=sco.minimize(min_shar,x0=weights_begin,method='SLSQP',constraints=cons,bounds=bnds)
    # print(ops['x'].round(3))
    print('---------')
    min_shar(ops['x'].round(3))
    #方差最小
    def min_variable(weights):
        x_variable=np.sqrt(np.dot(weights.reshape(1,-1),np.dot(variables,weights.reshape(-1,1))))
        return x_variable
    ops2=sco.minimize(min_variable,x0=weights_begin,method='SLSQP',bounds=bnds,constraints=cons)
    print(ops2['x'])
    min_shar(ops2['x'])
#     组合的有效前沿,给定收益率，使得是最小的,约束条件有2个，一个是收益率是确定的，其次是投资组合之和是1

    target_profit=np.linspace(0.01,0.55,2000)
    all_variables=[]
    for tar in target_profit:
        cons=({'type':'eq','fun':lambda x:(returns.mean().dot(x.T))*noa-tar},{'type':'eq','fun':lambda x:np.sum(x)-1})
        bons=tuple((0,1) for i in range(noa))
        ops3=sco.minimize(min_variable,np.array(15*[1/15,]),method='SLSQP',bounds=bons,constraints=cons)
        print('目标收益%s'%tar)
        print(ops3['x'].round(3))
        # weights=ops3['fun']
        all_variables.append(ops3['fun'])

    # print(all_variables)
    # plt.title('profit-variables')
    # plt.xlabel('profit')
    # plt.ylabel('variables')
    # plt.grid(True)
    # plt.scatter(target_profit,all_variables,c='r',label='profit-variables')
    # plt.show()


if __name__=='__main__':

    final_data=data_deal()
    print(final_data)

  
    with open('./data/final_data.pkl','rb') as f:
        data=pickle.load(f)
    get_target(data)
    with open('./data/final_data_pandas.pkl','rb') as f:
        data=pickle.load(f)
    save_stocks(data)
    print(get_data_sqr())
#根据final_tar进行从大到小进行排序，可选取评分前500/100只股票。



