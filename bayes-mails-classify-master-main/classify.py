import re
import os
from jieba import cut
from itertools import chain
from collections import Counter
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

# 获取脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAILS_DIR = os.path.join(BASE_DIR, r'e:\code\1\nlp\Code-of-hameo\bayes-mails-classify-master-main\邮件_files')

def get_words(filename):
    """读取文本并过滤无效字符和长度为1的词"""
    words = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            # 过滤无效字符
            line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)
            # 使用jieba.cut()方法对文本切词处理
            line = cut(line)
            # 过滤长度为1的词
            line = filter(lambda word: len(word) > 1, line)
            words.extend(line)
    return words

def get_top_words(top_num, all_words):
    """返回出现次数最多的词"""
    freq = Counter(chain(*all_words))
    return [i[0] for i in freq.most_common(top_num)]

def build_features(feature_method, top_num=100):
    """构建特征向量"""
    filename_list = [os.path.join(MAILS_DIR, '{}.txt'.format(i)) for i in range(151)]
    all_words = [get_words(filename) for filename in filename_list]
    
    if feature_method == "high_freq":
        # 高频词特征
        top_words = get_top_words(top_num, all_words)
        vector = []
        for words in all_words:
            word_map = list(map(lambda word: words.count(word), top_words))
            vector.append(word_map)
        vector = np.array(vector)
    elif feature_method == "tfidf":
        # TF-IDF加权特征
        documents = [" ".join(words) for words in all_words]
        vectorizer = TfidfVectorizer(max_features=top_num)
        vector = vectorizer.fit_transform(documents).toarray()
        top_words = vectorizer.get_feature_names_out()
    else:
        raise ValueError("Invalid feature_method. Choose 'high_freq' or 'tfidf'.")
    
    return vector, top_words

# 构建特征向量和标签
feature_method = "tfidf"  # 可切换为 "high_freq" 或 "tfidf"
vector, top_words = build_features(feature_method, top_num=100)
labels = np.array([1]*127 + [0]*24)

# 训练模型
model = MultinomialNB()
model.fit(vector, labels)

def predict(filename):
    """对未知邮件分类"""
    words = get_words(filename)
    if feature_method == "high_freq":
        current_vector = np.array(
            tuple(map(lambda word: words.count(word), top_words)))
    elif feature_method == "tfidf":
        document = " ".join(words)
        vectorizer = TfidfVectorizer(vocabulary=top_words)
        current_vector = vectorizer.fit_transform([document]).toarray()[0]
    else:
        raise ValueError("Invalid feature_method. Choose 'high_freq' or 'tfidf'.")
    
    result = model.predict(current_vector.reshape(1, -1))
    return '垃圾邮件' if result == 1 else '普通邮件'

# 测试分类
print('151.txt分类情况:{}'.format(predict(os.path.join(MAILS_DIR, '151.txt'))))
print('152.txt分类情况:{}'.format(predict(os.path.join(MAILS_DIR, '152.txt'))))
print('153.txt分类情况:{}'.format(predict(os.path.join(MAILS_DIR, '153.txt'))))
print('154.txt分类情况:{}'.format(predict(os.path.join(MAILS_DIR, '154.txt'))))
print('155.txt分类情况:{}'.format(predict(os.path.join(MAILS_DIR, '155.txt'))))
