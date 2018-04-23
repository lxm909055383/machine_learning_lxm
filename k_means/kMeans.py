#!/usr/bin/python
#coding:utf-8

from numpy import *
import matplotlib.pyplot as plt

#输入文件名
#输出数据矩阵
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))   #利用map函数将所有数据转化为float类型，便于计算距离
        dataMat.append(fltLine)
    return dataMat

#输入两个向量
#输出两者的欧氏距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

#输入数据集、k值
#输出k个质心的集合（必须在数据集范围内）
def randCent(dataSet, k):
    n = shape(dataSet)[1]   #数据集列数
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])  #第j列最小值
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))  #确保随机点在数据边界内
    return centroids

#输入数据集、k值、距离函数、质心选择函数
#输出每次质心位置和最终聚类结果
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]   #数据集行数
    clusterAssment = mat(zeros((m, 2)))  #第一列记录索引，第二列记录欧氏距离
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            #计算每个点与k个质心的距离，并标记最小值
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
        # print(centroids)  #输出几个说明质心改变了几次
        #遍历质心更新取值
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment

#绘制类图像
def showCluster(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape
    if dim != 2:
        print("Sorry! I can not draw because the dimension of your data is not 2!")
        return 1
    mark = ['or', 'ob', 'og', 'ok', 'oy', 'oc', 'om']
    if k > len(mark):
        print("Sorry! Your k is too large! please contact Zouxy")
        return 1
    #画所有样本点
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])  #第i行的类别
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])
    # 画质心
    mark = ['Dr', 'Db', 'Dg', 'Dk', 'Dy', 'Dc', 'Dm']
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=10)
    plt.show()

if __name__ == '__main__':
    datMat = mat(loadDataSet('testSet.txt'))
    k = 6
    centroids, clusterAssment = kMeans(datMat, k, distMeas=distEclud, createCent=randCent)
    showCluster(datMat, k, centroids, clusterAssment)

