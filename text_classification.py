
# coding:utf-8
import jieba
from sklearn.feature_extraction.text import CountVectorizer    #将词语转换为词频矩阵
from sklearn.feature_extraction.text import TfidfTransformer   #统计词语的tf-idf


# #语料
# corpus = ['This is the first document.',
#     'This is the second second document.',
#     'And the third one.',
#     'Is this the first document?',]
#
# #将文本中的词语转换为词频矩阵
# vectorizer = CountVectorizer()
# #计算每个词语出现的次数
# X = vectorizer.fit_transform(corpus)
# #获取词袋中所有文本关键词
# word = vectorizer.get_feature_names()
# print(word)
# #查看词频次数结果
# print(X.toarray())
#
# #类调用
# transformer = TfidfTransformer()
# #将词频矩阵X统计成TF-IDF值
# tfidf = transformer.fit_transform(X)
# #查看数据结构 tfidf[i][j]表示i类文本中的tf-idf权重
# print(tfidf.toarray())

if __name__ == "__main__":
    # 分词前的语料库
    corpus = ["我来到北京清华大学",  # 第一类文本
              "他来到了网易杭研大厦",  # 第二类文本
              "小明硕士毕业于中国科学院",  # 第三类文本
              "我爱北京天安门"]  # 第四类文本
    # 分词后的语料库
    corpus_cut = []
    for i in range(len(corpus)):
        cut_text = jieba.cut(corpus[i])
        result = ' '.join(cut_text)  # 用空格进行分词
        corpus_cut.append(result)
    print(corpus_cut)

    vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus_cut))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语（不重复），会自动过滤掉停用词
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    print(word)
    print(weight)
    for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
        print("-------这里输出第%d类文本的词语tf-idf权重------" % (i+1))
        for j in range(len(word)):
            print(word[j], weight[i][j])