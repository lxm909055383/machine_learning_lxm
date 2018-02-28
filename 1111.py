# coding:utf-8
import jieba

corpus = ["我来到北京清华大学",  # 第一类文本切词后的结果，词之间以空格隔开
              "他来到了网易杭研大厦",  # 第二类文本的切词结果
              "小明硕士毕业于中国科学院",  # 第三类文本的切词结果
              "我爱北京天安门"]  # 第四类文本的切词结果
list = []
for i in range(len(corpus)):
    cut_text = jieba.cut(corpus[i])
    result = ' '.join(cut_text)  #用空格进行分词
    list.append(result)
print(list)
