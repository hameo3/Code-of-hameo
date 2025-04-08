from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# 构建特征矩阵
feature_type = 'tfidf'  # 可切换为 'frequency' 或 'tfidf'
def build_features(feature_type, top_num):
    """构建特征矩阵和高频词列表"""
    # 示例实现，需根据实际需求调整
    if feature_type == 'frequency':
        vector = np.random.rand(151, top_num)  # 随机生成特征矩阵
        top_words = [f'word{i}' for i in range(top_num)]  # 示例高频词
    elif feature_type == 'tfidf':
        vector = np.random.rand(151, top_num)  # 随机生成特征矩阵
        top_words = [f'word{i}' for i in range(top_num)]  # 示例高频词
    else:
        raise ValueError("Invalid feature_type. Choose 'frequency' or 'tfidf'.")
    return vector, top_words

vector, top_words = build_features(feature_type=feature_type, top_num=100)

# 0-126.txt为垃圾邮件标记为1；127-151.txt为普通邮件标记为0
labels = np.array([1]*127 + [0]*24)

# 使用SMOTE进行过采样
smote = SMOTE(random_state=42)
vector_resampled, labels_resampled = smote.fit_resample(vector, labels)

# 训练模型
model = MultinomialNB()
model.fit(vector_resampled, labels_resampled)

def predict(filename):
    """对未知邮件分类"""
    words = get_words(filename)  # Ensure the function is defined below
    if feature_type == 'frequency':
        # 构建未知邮件的词向量（高频词特征）
        current_vector = np.array(
            tuple(map(lambda word: words.count(word), top_words)))
    elif feature_type == 'tfidf':
        # 构建未知邮件的词向量（TF-IDF加权特征）
        document = ' '.join(words)
        vectorizer = TfidfVectorizer(vocabulary=top_words)
        current_vector = vectorizer.fit_transform([document]).toarray()[0]
    else:
        raise ValueError("Invalid feature_type. Choose 'frequency' or 'tfidf'.")
    
    # 预测结果
    result = model.predict(current_vector.reshape(1, -1))
    return '垃圾邮件' if result == 1 else '普通邮件'

def get_words(filename):
    """Extract words from a file."""
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
    return content.split()  # Split content into words

# 定义邮件文件夹路径
MAILS_DIR =  r'Code-of-hameo\bayes-mails-classify-master-main\邮件_files'  # Replace 'path/to/mails' with the actual directory path

# 测试分类
print('151.txt分类情况:{}'.format(predict(os.path.join(MAILS_DIR, '151.txt'))))
print('152.txt分类情况:{}'.format(predict(os.path.join(MAILS_DIR, '152.txt'))))
print('153.txt分类情况:{}'.format(predict(os.path.join(MAILS_DIR, '153.txt'))))
print('154.txt分类情况:{}'.format(predict(os.path.join(MAILS_DIR, '154.txt'))))
print('155.txt分类情况:{}'.format(predict(os.path.join(MAILS_DIR, '155.txt'))))
