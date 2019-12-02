'''
Created on Dec 2, 2019

@author: I038825
'''
#Strategy.All_Pos
class Strategy(object):
    # 定义构造方法
    def __init__(self):  #__init__() 是类的初始化方法；它在类的实例化操作后 会自动调用，不需要手动调用；
        # 设置属性
        pass
    # 定义普通方法
    @staticmethod
    def All_Pos(self):
        #print("%s 说：我今年%s岁" % (self.name, self.age))
        poses = [{}]
        poses.append({"ServerCode":"0600000","CostPrice":10,"CurrentQty":100})
        poses.append({"ServerCode":"0600893","CostPrice":20,"CurrentQty":100})
        return poses

class Market(object):
    # 定义构造方法
    def __init__(self):  #__init__() 是类的初始化方法；它在类的实例化操作后 会自动调用，不需要手动调用；
        # 设置属性
        pass
    # 定义普通方法
    @staticmethod
    def Stk(self,stk):
        return Stock(stk)


class Stock(object):
    # 定义构造方法
    sk = ""
    def __init__(self,stk):  #__init__() 是类的初始化方法；它在类的实例化操作后 会自动调用，不需要手动调用；
        # 设置属性
        self.sk = stk
        pass
    # 定义普通方法
    def MinuteData1(self):
        return MinuteData1(self.sk) 
    def ServerCode(self):
        return self.sk
    
class MinuteData1(object):
    # 定义构造方法
    sk = ""
    def __init__(self,stk):  #__init__() 是类的初始化方法；它在类的实例化操作后 会自动调用，不需要手动调用；
        # 设置属性
        self.sk = stk
        pass
 
    def Stk(self):
        return self.sk
    def Count(self):
        return 3
    def OnNewBar(self,kdata, barNum):
        pass