#!/usr/bin/python
#coding:utf-8

import pandas as pd
import numpy as np

"""利用逻辑回归预测良/恶性肿瘤"""

"""part1——数据预处理"""
#创建特征列表（带目标变量）
column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
#使用pandas从互联网读取数据
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', names=column_names)
#原数据中缺失部分用？代替，要转化为标准的缺失值
data = data.replace(to_replace='?', value=np.nan)
#丢弃有缺失值的行
data = data.dropna(how='any')
#输出数据量和维度
print(data.shape)



"""part2——将数据随机分为训练集和测试集"""
from sklearn.cross_validation import train_test_split      #用于分割数据

#25%作为测试集
X_train, X_test, y_train, y_test = train_test_split(data[column_names[1:10]], data[column_names[10]], test_size=0.25, random_state=33)
#查看训练样本的数量和类别分布
print(y_train.value_counts())
#查看测试样本的数量和类别分布
print(y_test.value_counts())


"""part3——使用逻辑回归模型进行训练、预测"""
from sklearn.preprocessing import StandardScaler   #用于数据标准化
from sklearn.linear_model import LogisticRegression

#标准化，使每个维度数据方差为1，均值为0，不会因为特征值过大过小影响结果
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

lr = LogisticRegression()
#利用fit函数训练模型参数
lr.fit(X_train, y_train)
#利用训练好的模型对测试集进行预测
y_predict = lr.predict(X_test)
#分类目标下属于每个标签的概率
y_predict_pro = lr.predict_proba(X_test)
print(y_predict)
print(y_predict_pro)

"""part4——分析模型的性能"""
from sklearn.metrics import classification_report

#自带评分函数获得模型的准确性
print('准确率为：', lr.score(X_test, y_test))
#获得精确率、召回率、f1值
print(classification_report(y_test, y_predict, target_names=['Benign', 'Malignant']))
