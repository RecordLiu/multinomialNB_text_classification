#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import jieba
import pickle
import os

# 加载分类器模型
with open('text_classifier.pkl', 'rb') as file:
    classifier = pickle.load(file)

# 加载TfidfVectorizer模型
with open('tfidf_model.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# 读取数据文件
df = pd.read_excel('BOSS直聘数据分析职位数据源.xlsx')

# 将职位描述列转换为字符串类型
df['职位描述'] = df['职位描述'].astype(str)

# 对职位描述列进行分词
def tokenize(text):
    words = jieba.cut(text, cut_all=False, HMM=True)
    return ' '.join(words)

# 分词处理
corpus = df['职位描述'].apply(tokenize)

# 转换为TF-IDF向量
tfidf_matrix = tfidf_vectorizer.transform(corpus)

# 使用分类器进行分类
predicted_labels = classifier.predict(tfidf_matrix)

# 添加新列职位分类
df.insert(df.columns.get_loc('职位') + 1, '职位分类', predicted_labels)

# 指定文件路径
file_path = '分类结果.xlsx'

# 如果文件已存在，则删除
if os.path.exists(file_path):
    os.remove(file_path)

# 保存结果到新的Excel文件
df.to_excel(file_path, index=False)
