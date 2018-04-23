#!/usr/bin/python
#coding:utf-8

from sklearn import linear_model

clf = linear_model.LinearRegression()

#设置训练集
X = [[0, 0], [1, 1], [2, 2]]   #两个自变量
y = [0, 1, 2]
#训练模型
clf.fit(X, y)

#预测
print(clf.coef_)   #回归系数
print(clf.intercept_)   #截距

# print(y_pred)
# print(y_true)