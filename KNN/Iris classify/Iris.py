from numpy import *
import operator
import random

def file2matrix(filename):
    fr = open(filename)
    arrayOlines = fr.readlines()   #按行读取原始数据
    random.shuffle(arrayOlines)    #打乱顺序（所给数据集同一分类聚在一起，不方便后面测试）
    numberOfLines = len(arrayOlines)
    returnMat = zeros((numberOfLines, 4))  #使用了numpy库里的函数
    classLabelVector = []
    index = 0
    #解析文件数据到列表
    for line in arrayOlines:
        line = line.strip()  #移除字符串头尾指定的字符（默认为空格）
        listFromLine = line.split('\t')  #通过指定分隔符对字符串进行切片
        returnMat[index, :] = listFromLine[0:4]  #提取某一行所有特征列数据
        # classLabelVector.append(listFromLine[-1])  # 提取最后一列标签数据
        classLabelVector.append(int(listFromLine[-1]))  #提取最后一列标签数据
        index += 1
    return returnMat, classLabelVector

#归一化特征值
def autoNorm(dataset):
    minVals = dataset.min(0)   #找出每列最小的，是一个数组
    maxVals = dataset.max(0)   #找出每列最大的，是一个数组
    ranges = maxVals - minVals
    m = dataset.shape[0]   #取行数
    normDataSet = dataset - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

#求当前点的分类
def classify(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]   #读取矩阵第一维度的长度（行数）
    #距离计算（欧氏距离）
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    #选择距离最小的k个点
    for i in range(k):
        votellabel = labels[sortedDistIndicies[i]]
        classCount[votellabel] = classCount.get(votellabel,0) + 1
    #sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse= True)  #用到operator库
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#利用测试集测试分类器的正确率
# def ClassTest():
#     hoRatio = 0.20  #所给数据集中抽取作为测试集的比例
#     DataMat, Labels = file2matrix('iris.txt')  #调用第一个函数
#     normMat, ranges, minVals = autoNorm(DataMat)  #调用第二个函数
#     m = normMat.shape[0]
#     numTestVecs = int(m*hoRatio)   #测试集的数量
#     errorCount = 0.0
#     for i in range(numTestVecs):
#         classifierResult = classify(normMat[i, :], normMat[numTestVecs:m, :], Labels[numTestVecs:m], 10)  #调用第三个函数
#         print("the classifier came back with: %s, the real answer is: %s" % (classifierResult, Labels[i]))
#         if classifierResult != Labels[i]:
#             errorCount += 1.0
#     print("the correct rate is: %f" % (1 - errorCount/float(numTestVecs)))
#
# ClassTest()

#利用分类器进行预测
# def classifyPridict():
#     resultList = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
#     sepal_length = float(input("the length of sepal?"))
#     sepal_width = float(input("the width of sepal?"))
#     petal_length = float(input("the length of petal?"))
#     petal_width = float(input("the width of petal?"))
#     DataMat, Labels = file2matrix('iris.txt')
#     normMat, ranges, minVals = autoNorm(DataMat)
#     inArr = array([sepal_length, sepal_width, petal_length, petal_width])
#     classifierResult = classify((inArr - minVals)/ranges, normMat, Labels, 10)
#     print("the kind of the iris is: %s" % classifierResult)
#
# classifyPridict()

#可视化
import matplotlib.pyplot as plt

returnMat, classLabelVector = file2matrix('iris2.txt')
fig = plt.figure()
ax = fig.add_subplot(111)
# ax.scatter(returnMat[:, 0], returnMat[:, 1])
ax.scatter(returnMat[:, 0], returnMat[:, 2], 15.0*array(classLabelVector), 15.0*array(classLabelVector))
plt.xlabel(u'花萼长度')
plt.ylabel(u'花瓣长度')
# 0 the length of sepal 花萼长度
# 1 the width of sepal 花萼宽度
# 2 the length of petal 花瓣长度
# 3 the width of petal 花瓣宽度
plt.show()
