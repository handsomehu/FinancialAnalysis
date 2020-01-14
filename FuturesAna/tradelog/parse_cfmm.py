# -*- coding: utf-8 -*-
import re
import pandas as pd
bslist = [["制表时间","期初结存" ,"期末结存 ","平仓盈亏" ,"持仓盯市盈亏","交易记录"]]

def read_file(bslist):
    lv_found = False
    countdown = 6
    stx = []
    str0 = ""  
    str1 = ""
    str2 = ""
    str3 = ""
    str4 = ""
    #' '-?\d+\.*\d*''
    with open("order.txt", "r", encoding="utf-8") as f:
        for line in f.readlines():
            if re.search("制表时间", line) != None:
                str0 = re.findall(r"-?\d+\.*\d*", line)[0]
                print(str0)             
            if re.match("期初结存 Balance b/f", line) != None:
                str1 = re.findall(r"-?\d+\.*\d*", line)[0]
                print(str1)
            if re.search("期末结存 Balance c/f", line) != None:
                str2 = re.findall(r"-?\d+\.*\d*", line)[1]
                print(str2)             
            if re.search("平仓盈亏 Realized P/L", line) != None:
                str3 = re.findall(r"-?\d+\.*\d*", line)[0]
                print(str3)    
            if re.search("持仓盯市盈亏 MTM P/L", line) != None:
                str4 = re.findall(r"-?\d+\.*\d*", line)[0]
                print(str4)                    
            if re.search("成交记录", line) != None:
                print("found")
                lv_found = True
            if lv_found and countdown > 0:
                #print(line)
                countdown -= 1            
            if lv_found and countdown == 0:         
                if re.match("-", line) != None:
                    lv_found = False   
                else:
                    stx.append(line)         

        bslist.append([str0,str1,str2,str3,str4,stx])
        #print(stx)
read_file(bslist)
print("ddddd")
stx = bslist[1][5]
str = ""
for tx in stx:
    str = str + tx
df = pd.DataFrame([x.split('|') for x in str.split('\n')])
df = df.dropna()
df.to_csv("./test1.csv")