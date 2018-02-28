from numpy import *
import operator
from os import listdir    #列出给定目录的文件名

#将一张32*32图片转化为1*1024向量
def img2vector(filename):
    returnVect = zeros((1,1024))  #初始化为1*1024的全0向量
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

#k近邻求类别
def classify(inX, dataSet, labels, k):   #测试数据，训练集特征属性值，训练集类别，给定的k值
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

#手写识别测试代码
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')      #加载训练集内容
    m = len(trainingFileList)  #训练集的文件个数
    trainingMat = zeros((m,1024))   #每个文件是一行向量
    for i in range(m):
        #根据文件名解析分类数字
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  #分割提取0_0.txt中的0_0，[1]代表txt
        classNumStr = int(fileStr.split('_')[0])  #分割提取0_0中前面的数字
        hwLabels.append(classNumStr)
        #将内容写入到向量中
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)

    testFileList = listdir('testDigits')     #加载测试集内容
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, 10)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0
    print("\nthe correct rate is: %f" % (1 - errorCount/float(mTest)))

handwritingClassTest()