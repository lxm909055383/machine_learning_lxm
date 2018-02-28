from numpy import *
import operator
import random
import matplotlib
import matplotlib.pyplot as plt

#文本处理
def file2matrix(filename):
    fr = open(filename)
    arrayOlines = fr.readlines()
    random.shuffle(arrayOlines)    #打乱顺序
    numberOfLines = len(arrayOlines)
    returnMat = zeros((numberOfLines, 3))  #使用了numpy库里的函数
    classLabelVector = []
    index = 0
    #解析文件数据到列表
    for line in arrayOlines:
        line = line.strip()  #移除字符串头尾指定的字符（默认为空格）
        listFromLine = line.split('\t')  #通过指定分隔符对字符串进行切片
        returnMat[index, :] = listFromLine[0:3]   #提取某一行所有特征列数据
        classLabelVector.append(listFromLine[-1])  #提取最后一列标签数据
        # classLabelVector.append(int(listFromLine[-1]))
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


#k近邻求类别
def classify(inX, dataSet, labels, k):
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
    #sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse= True)
    sortedClassCount = sorted(classCount.items(), key=lambda item: item[1], reverse=True)
    return sortedClassCount[0][0]


#测试分类器
def ClassTest():
    hoRatio = 0.20
    DataMat, Labels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(DataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify(normMat[i, :], normMat[numTestVecs:m, :], Labels[numTestVecs:m], 10)
        print("the classifier came back with: %s, the real answer is: %s" % (classifierResult, Labels[i]))
        if classifierResult != Labels[i]:
            errorCount += 1.0
    print("the correct rate is: %f" % (1 - errorCount/float(numTestVecs)))

ClassTest()


# #可视化
# datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
# fig = plt.figure()
# ax = fig.add_subplot(111)
# #在这里使用datingLabels时必须转为整型
# ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0*array(datingLabels), 15.0*array(datingLabels))
# plt.show()


