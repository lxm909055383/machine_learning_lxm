
from numpy import *

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1代表侮辱性文，0代表正常言论
    return postingList,classVec

#在所有文档中出现的不重复词的列表
def createVocabList(dataSet):
    vocabSet = set([])  #创建一个空集合
    for document in dataSet:
        vocabSet = vocabSet | set(document) #两个集合的并集
    return list(vocabSet)

#每个词只能出现一次，后面需要修改的
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)  #创建一个全是0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

#词袋模型，词向量对应的值会随单词出现的次数增加
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords); p1Num = ones(numWords)      #由0改为1
    p0Denom = 2.0; p1Denom = 2.0                        #由0改为2
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)          #改为log()
    p0Vect = log(p0Num/p0Denom)          #改为log()
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #元素相乘
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        # trainMat.append(setOfWords2Vec(myVocabList, postinDoc))  #setOfWords2Vec
        trainMat.append(bagOfWords2VecMN(myVocabList, postinDoc))  #bagOfWords2VecMN
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    # thisDoc = array(setOfWords2Vec(myVocabList, testEntry))  #setOfWords2Vec
    thisDoc = array(bagOfWords2VecMN(myVocabList, testEntry))  #bagOfWords2VecMN
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    # thisDoc = array(setOfWords2Vec(myVocabList, testEntry))   #setOfWords2Vec
    thisDoc = array(bagOfWords2VecMN(myVocabList, testEntry))   #bagOfWords2VecMN
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))

if __name__ == '__main__':
    testingNB()    #便利函数，封装了所有的内容


