#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 中文文本分类(朴素贝叶斯)
import os
import jieba
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pickle

warnings.filterwarnings('ignore')

def cut_words(file_path):
    """
    对文本进行切词
    :param file_path: txt文本路径
    :return: 用空格分词的字符串
    """
    text_with_spaces = ''
    text=open(file_path, 'r', encoding='gb18030').read()
    textcut = jieba.cut(text)
    for word in textcut:
        text_with_spaces += word + ' '
    return text_with_spaces

def loadfile(file_dir, label):
    """
    将路径下的所有文件加载
    :param file_dir: 保存txt文件目录
    :param label: 文档标签
    :return: 分词后的文档列表和标签
    """
    file_list = os.listdir(file_dir)
    words_list = []
    labels_list = []
    for file in file_list:
        file_path = file_dir + '/' + file
        words_list.append(cut_words(file_path))
        labels_list.append(label)                                                                                                                 
    return words_list, labels_list

# 训练数据
train_files = [
    ('大数据分析工程师', '大数据分析工程师'),
    ('非数据分析岗', '非数据分析岗'),
    ('数据产品', '数据产品'),
    ('数据处理', '数据处理'),
    ('数据治理工程师', '数据治理工程师'),
    ('数据科学家', '数据科学家'),
    ('数据分析师', '数据分析师')
    
]

train_words_list = []
train_labels = []
for filename, label in train_files:
    words_list, labels = loadfile(f'data/train/{filename}', label)
    train_words_list += words_list
    train_labels += labels

# 测试数据
test_files = [
    ('大数据分析工程师', '大数据分析工程师'),
    ('非数据分析岗', '非数据分析岗'),
    ('数据产品', '数据产品'),
    ('数据处理', '数据处理'),
    ('数据治理工程师', '数据治理工程师'),
    ('数据科学家', '数据科学家'),
    ('数据分析师', '数据分析师')
]

test_words_list = []
test_labels = []
for filename, label in test_files:
    words_list, labels = loadfile(f'data/test/{filename}', label)
    test_words_list += words_list
    test_labels += labels

# 加载停用词
stop_words = open('data/stop/stopwords.txt', 'r', encoding='utf-8').read()
stop_words = stop_words.encode('utf-8').decode('utf-8-sig') # 列表头部\ufeff处理
stop_words = stop_words.split('\n') # 根据分隔符分隔

# 计算单词权重
tf = TfidfVectorizer(stop_words=stop_words, max_df=0.5)

train_features = tf.fit_transform(train_words_list)

with open('tfidf_model.pkl', 'wb') as f:
    pickle.dump(tf, f)

# 上面fit过了，这里transform
test_features = tf.transform(test_words_list) 

# 多项式贝叶斯分类器

from sklearn.naive_bayes import MultinomialNB  
clf = MultinomialNB(alpha=0.01).fit(train_features, train_labels)
predicted_labels=clf.predict(test_features)

# 计算准确率
print('准确率为：', metrics.accuracy_score(test_labels, predicted_labels))

# 保存模型
with open('text_classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)