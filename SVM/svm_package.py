
"""识别手写体数字图片"""

"""从sklearn数据集读取数据"""
from sklearn.datasets import load_digits

#通过数据加载器将图像数据储存在digits变量中
digits = load_digits()
print(digits.data.shape)


"""将数据随机分为训练集和测试集"""
from sklearn.cross_validation import train_test_split      #用于分割数据
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=1)
print(y_train.shape)
print(y_test.shape)


"""使用支持向量机进行训练、预测"""
from sklearn.preprocessing import StandardScaler   #用于数据标准化
from sklearn.svm import LinearSVC  #导入基于线性假设的支持向量机分类器

#标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

lsvc = LinearSVC()
#利用fit函数训练模型参数
lsvc.fit(X_train, y_train)
#利用训练好的模型对测试集进行预测
y_predict = lsvc.predict(X_test)
print(y_predict)


"""分析模型的性能"""
from sklearn.metrics import classification_report

#自带评分函数获得模型的准确性
print('准确率为：', lsvc.score(X_test, y_test))
#获得精确率、召回率、f1值
print(classification_report(y_test, y_predict, target_names=digits.target_names.astype(str)))