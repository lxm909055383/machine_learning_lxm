from numpy import *
import matplotlib.pyplot as plt

#数据集进行分割
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()  #移除空格并分割
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])  #X0设置为1，X1和X2从文件中获得
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat   #dataMat 100行3列，labelMat 1行列表

#sigmoid函数
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

#梯度上升算法得到最优的权系数（矩阵的计算量很大）
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)    #转化为NumPy矩阵类型，100行3列
    labelMat = mat(classLabels).transpose()   #转化为NumPy矩阵类型并转置，100行1列
    m, n = shape(dataMatrix)  #获得行数、列数
    alpha = 0.001  #设置步长
    maxCycles = 500   #迭代步数
    weights = ones((n, 1))  #权系数初始化，3行1列
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)     #矩阵乘法
        error = (labelMat - h)        #向量的减法
        weights = weights + alpha*dataMatrix.transpose()*error  #不太理解？？？？？？？？？
    return weights

# #随机梯度上升算法(优点：全数值计算，没有矩阵的转化过程，全程数组类型)
# def stocGradAscent0(dataMatrix, classLabels):
#     m, n = shape(dataMatrix)
#     alpha = 0.01
#     weights = ones(n)
#     for i in range(m):
#         h = sigmoid(sum(dataMatrix[i]*weights))
#         error = classLabels[i] - h
#         weights = weights + alpha*error*dataMatrix[i]
#     return weights

#画分割线与散点图
def plotBestFit(wei):
    weights = wei.getA()  #将NumPy矩阵转化为NumPy数组，不然下面根据索引提取会越界
    # weights = array(wei)  #该方式也可以转换
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]   #此处x是X1，y是X2， 0=W0*X0+W1*X1+W2*X2  X0=1
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

dataMat, labelMat = loadDataSet()
weights = gradAscent(dataMat, labelMat)
print(weights)
plotBestFit(weights)