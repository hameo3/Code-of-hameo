import re
import os
from jieba import cut
from itertools import chain
from collections import Counter
import numpy as np
from sklearn.naive_bayes import MultinomialNB

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

all_words = []

def get_top_words(top_num):
    """遍历邮件建立词库后返回出现次数最多的词"""
    filename_list = [os.path.join(MAILS_DIR, '{}.txt'.format(i)) for i in range(151)]
    # 遍历邮件建立词库
    for filename in filename_list:
        all_words.append(get_words(filename))
    # itertools.chain()把all_words内的所有列表组合成一个列表
    # collections.Counter()统计词个数
    freq = Counter(chain(*all_words))
    return [i[0] for i in freq.most_common(top_num)]

top_words = get_top_words(100)
# 构建词-个数映射表
vector = []
for words in all_words:
    word_map = list(map(lambda word: words.count(word), top_words))
    vector.append(word_map)

vector = np.array(vector)
# 0-126.txt为垃圾邮件标记为1；127-151.txt为普通邮件标记为0
labels = np.array([1]*127 + [0]*24)
model = MultinomialNB()
model.fit(vector, labels)

def predict(filename):
    """对未知邮件分类"""
    # 构建未知邮件的词向量
    words = get_words(filename)
    current_vector = np.array(
        tuple(map(lambda word: words.count(word), top_words)))
    # 预测结果
    result = model.predict(current_vector.reshape(1, -1))
    return '垃圾邮件' if result == 1 else '普通邮件'

# 测试分类
print('151.txt分类情况:{}'.format(predict(os.path.join(MAILS_DIR, '151.txt'))))
print('152.txt分类情况:{}'.format(predict(os.path.join(MAILS_DIR, '152.txt'))))
print('153.txt分类情况:{}'.format(predict(os.path.join(MAILS_DIR, '153.txt'))))
print('154.txt分类情况:{}'.format(predict(os.path.join(MAILS_DIR, '154.txt'))))
print('155.txt分类情况:{}'.format(predict(os.path.join(MAILS_DIR, '155.txt'))))
