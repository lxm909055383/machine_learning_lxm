
import pandas as pd

"""从网站获取数据"""
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
#观察前几行数据
print(titanic.head())
#查看数据的统计特性
print(titanic.info())



"""数据预处理"""