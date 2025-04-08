## 项目说明

本项目实现了基于贝叶斯分类器的邮件分类功能，能够对邮件内容进行分类，判断其是否为垃圾邮件。

### 核心功能
1. **邮件预处理**：对邮件内容进行分词、去停用词等预处理操作。
2. **特征提取**：支持高频词和TF-IDF两种特征提取模式。
3. **贝叶斯分类**：基于朴素贝叶斯算法对邮件进行分类。

### 特征模式切换方法
在代码中，可以通过修改配置文件或代码中的参数来切换特征提取模式：
- **高频词模式**：设置参数 `feature_mode = "high_freq"`。
- **TF-IDF模式**：设置参数 `feature_mode = "tfidf"`。

示例代码：
```python
# 切换特征模式
feature_mode = "tfidf"  # 可选值："high_freq" 或 "tfidf"
```

### 运行结果

以下为运行结果示例：

<img src="https://github.com/hameo3/Code-of-hameo/blob/main/images/bayes.png" alt="贝叶斯" width="200">

### 数学公式示例
贝叶斯公式：

```math
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
```
