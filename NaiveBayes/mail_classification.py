
import random
from numpy import *
from bayes import loadDataSet, createVocabList, bagOfWords2VecMN, trainNB0, classifyNB

#将字符串文本转化为字符串列表
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)   #\W匹配字母或数字或下划线或汉字
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []   #特征词列表
    classList = []
    fullText = []
    for i in range(1, 26):
        #将文件导入并解析
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # 创建特征词向量
    trainingSet = range(50)
    testSet = []  #随机选择10个作为测试集
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error", docList[docIndex])
    print('the error rate is: ', float(errorCount) / len(testSet))
    # return vocabList,fullText


if __name__ == '__main__':
    spamTest()



