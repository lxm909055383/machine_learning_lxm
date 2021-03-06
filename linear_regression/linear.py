#!/usr/bin/python
#coding:utf-8

from numpy import *
import matplotlib.pyplot as plt

#读取文件
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1  # 得到特征的个数
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):    #遍历存取所有特征
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))   #最后一个为目标值
    return dataMat, labelMat

#计算回归系数
def standRegres(xArr, yArr):
    xMat = mat(xArr)  #存放特征的矩阵
    yMat = mat(yArr).T  #存放目标的向量，.T求转置
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:   #求矩阵行列式，为0不可逆
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * yMat)  #.I求逆
    return ws

#画图并求模型的相关系数
def plotData():
    #绘制散点图
    xArr, yArr = loadDataSet('ex0.txt')
    xMat = mat(xArr)
    yMat = mat(yArr)
    figure = plt.figure()
    ax = figure.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])  #flatten返回一个折叠成一维的数组
    #绘制拟合直线图
    xCopy = xMat.copy()
    xCopy.sort(0)  #将输入数据升序排列，不然画直线会混乱
    ws = standRegres(xArr, yArr)
    y_pre = xCopy * ws   #表示预测值的纵坐标，用回归系数求出
    ax.plot(xCopy[:, 1].flatten().A[0], y_pre.flatten().A[0], c='red')
    plt.show()

plotData()

#求相关系数
xArr, yArr = loadDataSet('ex0.txt')
xMat = mat(xArr)
yMat = mat(yArr)
ws = standRegres(xArr, yArr)
y_pre = xMat * ws
print(corrcoef(y_pre.T, yMat))  #两个必须都是行向量







# def lwlr(testPoint, xArr, yArr, k=1.0):
#     xMat = mat(xArr);
#     yMat = mat(yArr).T
#     m = shape(xMat)[0]
#     weights = mat(eye((m)))
#     for j in range(m):  # next 2 lines create weights matrix
#         diffMat = testPoint - xMat[j, :]  #
#         weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
#     xTx = xMat.T * (weights * xMat)
#     if linalg.det(xTx) == 0.0:
#         print
#         "This matrix is singular, cannot do inverse"
#         return
#     ws = xTx.I * (xMat.T * (weights * yMat))
#     return testPoint * ws
#
#
# def lwlrTest(testArr, xArr, yArr, k=1.0):  # loops over all the data points and applies lwlr to each one
#     m = shape(testArr)[0]
#     yHat = zeros(m)
#     for i in range(m):
#         yHat[i] = lwlr(testArr[i], xArr, yArr, k)
#     return yHat
#
#
# def lwlrTestPlot(xArr, yArr, k=1.0):  # same thing as lwlrTest except it sorts X first
#     yHat = zeros(shape(yArr))  # easier for plotting
#     xCopy = mat(xArr)
#     xCopy.sort(0)
#     for i in range(shape(xArr)[0]):
#         yHat[i] = lwlr(xCopy[i], xArr, yArr, k)
#     return yHat, xCopy
#
#
# def rssError(yArr, yHatArr):  # yArr and yHatArr both need to be arrays
#     return ((yArr - yHatArr) ** 2).sum()
#
#
# def ridgeRegres(xMat, yMat, lam=0.2):
#     xTx = xMat.T * xMat
#     denom = xTx + eye(shape(xMat)[1]) * lam
#     if linalg.det(denom) == 0.0:
#         print
#         "This matrix is singular, cannot do inverse"
#         return
#     ws = denom.I * (xMat.T * yMat)
#     return ws
#
#
# def ridgeTest(xArr, yArr):
#     xMat = mat(xArr);
#     yMat = mat(yArr).T
#     yMean = mean(yMat, 0)
#     yMat = yMat - yMean  # to eliminate X0 take mean off of Y
#     # regularize X's
#     xMeans = mean(xMat, 0)  # calc mean then subtract it off
#     xVar = var(xMat, 0)  # calc variance of Xi then divide by it
#     xMat = (xMat - xMeans) / xVar
#     numTestPts = 30
#     wMat = zeros((numTestPts, shape(xMat)[1]))
#     for i in range(numTestPts):
#         ws = ridgeRegres(xMat, yMat, exp(i - 10))
#         wMat[i, :] = ws.T
#     return wMat
#
#
# def regularize(xMat):  # regularize by columns
#     inMat = xMat.copy()
#     inMeans = mean(inMat, 0)  # calc mean then subtract it off
#     inVar = var(inMat, 0)  # calc variance of Xi then divide by it
#     inMat = (inMat - inMeans) / inVar
#     return inMat
#
#
# def stageWise(xArr, yArr, eps=0.01, numIt=100):
#     xMat = mat(xArr);
#     yMat = mat(yArr).T
#     yMean = mean(yMat, 0)
#     yMat = yMat - yMean  # can also regularize ys but will get smaller coef
#     xMat = regularize(xMat)
#     m, n = shape(xMat)
#     returnMat = zeros((numIt, n))  # testing code remove
#     ws = zeros((n, 1));
#     wsTest = ws.copy();
#     wsMax = ws.copy()
#     for i in range(numIt):  # could change this to while loop
#         # print ws.T
#         lowestError = inf;
#         for j in range(n):
#             for sign in [-1, 1]:
#                 wsTest = ws.copy()
#                 wsTest[j] += eps * sign
#                 yTest = xMat * wsTest
#                 rssE = rssError(yMat.A, yTest.A)
#                 if rssE < lowestError:
#                     lowestError = rssE
#                     wsMax = wsTest
#         ws = wsMax.copy()
#         returnMat[i, :] = ws.T
#     return returnMat
#
#
# def scrapePage(inFile, outFile, yr, numPce, origPrc):
#     from BeautifulSoup import BeautifulSoup
#     fr = open(inFile);
#     fw = open(outFile, 'a')  # a is append mode writing
#     soup = BeautifulSoup(fr.read())
#     i = 1
#     currentRow = soup.findAll('table', r="%d" % i)
#     while (len(currentRow) != 0):
#         currentRow = soup.findAll('table', r="%d" % i)
#         title = currentRow[0].findAll('a')[1].text
#         lwrTitle = title.lower()
#         if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
#             newFlag = 1.0
#         else:
#             newFlag = 0.0
#         soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
#         if len(soldUnicde) == 0:
#             print
#             "item #%d did not sell" % i
#         else:
#             soldPrice = currentRow[0].findAll('td')[4]
#             priceStr = soldPrice.text
#             priceStr = priceStr.replace('$', '')  # strips out $
#             priceStr = priceStr.replace(',', '')  # strips out ,
#             if len(soldPrice) > 1:
#                 priceStr = priceStr.replace('Free shipping', '')  # strips out Free Shipping
#             print
#             "%s\t%d\t%s" % (priceStr, newFlag, title)
#             fw.write("%d\t%d\t%d\t%f\t%s\n" % (yr, numPce, newFlag, origPrc, priceStr))
#         i += 1
#         currentRow = soup.findAll('table', r="%d" % i)
#     fw.close()
#
#
# def setDataCollect():
#     scrapePage('setHtml/lego8288.html', 'out.txt', 2006, 800, 49.99)
#     scrapePage('setHtml/lego10030.html', 'out.txt', 2002, 3096, 269.99)
#     scrapePage('setHtml/lego10179.html', 'out.txt', 2007, 5195, 499.99)
#     scrapePage('setHtml/lego10181.html', 'out.txt', 2007, 3428, 199.99)
#     scrapePage('setHtml/lego10189.html', 'out.txt', 2008, 5922, 299.99)
#     scrapePage('setHtml/lego10196.html', 'out.txt', 2009, 3263, 249.99)
#
#
# def crossValidation(xArr, yArr, numVal=10):
#     m = len(yArr)
#     indexList = range(m)
#     errorMat = zeros((numVal, 30))  # create error mat 30columns numVal rows
#     for i in range(numVal):
#         trainX = [];
#         trainY = []
#         testX = [];
#         testY = []
#         random.shuffle(indexList)
#         for j in range(m):  # create training set based on first 90% of values in indexList
#             if j < m * 0.9:
#                 trainX.append(xArr[indexList[j]])
#                 trainY.append(yArr[indexList[j]])
#             else:
#                 testX.append(xArr[indexList[j]])
#                 testY.append(yArr[indexList[j]])
#         wMat = ridgeTest(trainX, trainY)  # get 30 weight vectors from ridge
#         for k in range(30):  # loop over all of the ridge estimates
#             matTestX = mat(testX);
#             matTrainX = mat(trainX)
#             meanTrain = mean(matTrainX, 0)
#             varTrain = var(matTrainX, 0)
#             matTestX = (matTestX - meanTrain) / varTrain  # regularize test with training params
#             yEst = matTestX * mat(wMat[k, :]).T + mean(trainY)  # test ridge results and store
#             errorMat[i, k] = rssError(yEst.T.A, array(testY))
#             # print errorMat[i,k]
#     meanErrors = mean(errorMat, 0)  # calc avg performance of the different ridge weight vectors
#     minMean = float(min(meanErrors))
#     bestWeights = wMat[nonzero(meanErrors == minMean)]
#     # can unregularize to get model
#     # when we regularized we wrote Xreg = (x-meanX)/var(x)
#     # we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
#     xMat = mat(xArr);
#     yMat = mat(yArr).T
#     meanX = mean(xMat, 0);
#     varX = var(xMat, 0)
#     unReg = bestWeights / varX
#     print
#     "the best model from Ridge Regression is:\n", unReg
#     print
#     "with constant term: ", -1 * sum(multiply(meanX, unReg)) + mean(yMat)
