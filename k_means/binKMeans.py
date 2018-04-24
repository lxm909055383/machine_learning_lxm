#!/usr/bin/python
#coding:utf-8

from k_means.kMeans import *

def biKmeans(dataSet, k, distMeas = distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))  #存放分配结果和平方误差
    centroid0 = mean(dataSet, axis=0).tolist()[0]  #取每列的均值组合成第一个初始质心
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :])**2  #计算每个点的平方误差
    while (len(centList) < k):   # 遍历所有质心
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]   # 搜索到当前质心所聚类的样本
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)  #利用K均值划分每个簇
            sseSplit = sum(splitClustAss[:, 1])  #划分数据集的SSE值
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1]) #未划分数据集的SSE值
            print("sseSplit and notSplit: ", sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return mat(centList), clusterAssment

if __name__ == '__main__':
    datMat = mat(loadDataSet('testSet.txt'))
    k = 4
    centroids, clusterAssment = biKmeans(datMat, k, distMeas=distEclud)
    showCluster(datMat, k, centroids, clusterAssment)