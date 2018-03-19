
from numpy import *

#导入训练集数据与目标分类
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]    #1代表侮辱性文，0代表正常言论
    return postingList, classVec

#输入训练集所有数据
#输出不重复的特征词向量
def createVocabList(dataSet):
    vocabSet = set([])  #创建一个空集合
    for document in dataSet:
        vocabSet = vocabSet | set(document) #两个集合的并集
    return list(vocabSet)

#输入特征词向量、某一条训练数据
#输出该条训练数据在特征词向量下的0-1编码（0代表没有，1代表有，多次出现也为1）
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)  #创建一个全是0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

#输入特征词向量、某一条训练数据
#输出该条训练数据在特征词向量下的编码（0代表没有，对应数字代表出现的次数）
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

#输入训练集数字编码后的矩阵、对应分类
#输出类别为0时各特征的权重、类别为1时各特征的权重、类别为1的概率
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    #类别为1的概率
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords);  p1Num = ones(numWords)      #由0改为1，防止某分类下的特征为0
    p0Denom = 2.0;  p1Denom = 2.0             #由0改为2，防止只有一个类别时分母为0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)          #改为log()，防止下溢出
    p0Vect = log(p0Num/p0Denom)          #改为log()，防止下溢出
    return p0Vect, p1Vect, pAbusive

#输入待分类项编码后数组、类别为0时各特征的权重、类别为1时各特征的权重、类别为1的概率
#输出所属类别
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    #使用log函数将求乘积转化为求和
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #对应元素相乘再求和
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0

#封装了所有的内容
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    print(myVocabList)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))  #setOfWords2Vec
        print(trainMat)
        exit()
        trainMat.append(bagOfWords2VecMN(myVocabList, postinDoc))  #bagOfWords2VecMN
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))

    #验证测试集
    testEntry = ['love', 'my', 'dalmation']
    # thisDoc = array(setOfWords2Vec(myVocabList, testEntry))  #setOfWords2Vec
    thisDoc = array(bagOfWords2VecMN(myVocabList, testEntry))  #bagOfWords2VecMN
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

    testEntry = ['stupid', 'garbage']
    # thisDoc = array(setOfWords2Vec(myVocabList, testEntry))   #setOfWords2Vec
    thisDoc = array(bagOfWords2VecMN(myVocabList, testEntry))   #bagOfWords2VecMN
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

if __name__ == '__main__':
    testingNB()


