# 任务一：基于sklearn的文本分类

也是用机器学习和前面的模型内容基本一致，主要试验了用sklearn进行文本分类的试验。

数据集：[Classify the sentiment of sentences from the Rotten Tomatoes dataset](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews)

### 特征构建

创建特征提取器，在训练集上进行拟合(fit)，运用于训练集和测试机(transform)。

```python
from sklearn.feature_extraction.text import CountVectorizer

#ngram_rangetuple (min_n, max_n), default=(1, 1)
ngram_range=(1,1)
#analyzer{‘word’, ‘char’, ‘char_wb’} or callable, default=’word’
analyzer='word'

vectorizer = CountVectorizer(ngram_range=ngram_range,analyzer=analyzer).fit(df['Phrase'])
X=vectorizer.transform(df['Phrase'])
y=df['Sentiment']
test_X=vectorizer.transform(df_test['Phrase'])
```



### 数据集划分

将训练集划分为训练集和验证集。

```python
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split( X, y, test_size=0.2, random_state=42,shuffle=True)
```



### 模型训练

公式上如下所示，下面的公式函数都已经集成好了。

我们首先需要知道我们采用的模型的公式表达，损失函数表达，参数更新方式。

SoftMax Regression 公式（P75）：
$$
\begin{aligned}
\hat{\boldsymbol{y}} &=\operatorname{softmax}\left(\boldsymbol{W}^{\top} \boldsymbol{x}\right) \\
&=\frac{\exp \left(\boldsymbol{W}^{\top} \boldsymbol{x}\right)}{\mathbf{1}_{C}^{\top} \exp \left(\boldsymbol{W}^{\top} \boldsymbol{x}\right)}
\end{aligned}
$$
其中 $\boldsymbol{W}=\left[\boldsymbol{w}_{1}, \cdots, \boldsymbol{w}_{C}\right]$ 是由 $C$ 个类的权重向量组成的矩阵, $\mathbf{1}_{C}$ 为 $C$ 维的全 1 向 量, $\hat{\boldsymbol{y}} \in \mathbb{R}^{C}$ 为所有类别的预测条件概率组成的向量, 第 $c$ 维的值是第 $c$ 类的预测 条件概率.

损失函数（p76）：
$$
\begin{aligned}
\mathcal{R}(\boldsymbol{W}) &=-\frac{1}{N} \sum_{n=1}^{N} \sum_{c=1}^{C} \boldsymbol{y}_{c}^{(n)} \log \hat{\boldsymbol{y}}_{c}^{(n)} \\
&=-\frac{1}{N} \sum_{n=1}^{N}\left(\boldsymbol{y}^{(n)}\right)^{\mathrm{T}} \log \hat{\boldsymbol{y}}^{(n)},
\end{aligned}
$$
其中 $\hat{\boldsymbol{y}}^{(n)}=\operatorname{softmax}\left(\boldsymbol{W}^{\top} \boldsymbol{x}^{(n)}\right)$ 为样本 $\boldsymbol{x}^{(n)}$ 在每个类别的后验概率.

参数更新采用梯度下降的方式进行更新（P76）：
$$
\boldsymbol{W}_{t+1} \leftarrow \boldsymbol{W}_{t}+\alpha\left(\frac{1}{N} \sum_{n=1}^{N} \boldsymbol{x}^{(n)}\left(\boldsymbol{y}^{(n)}-\hat{\boldsymbol{y}}_{W_{t}}^{(n)}\right)^{\top}\right)
$$
其中 $\alpha$ 是学习率, $\hat{y}_{W_{t}}^{(n)}$ 是当参数为 $\boldsymbol{W}_{t}$ 时, Softmax 回归模型的输出。

如下是训练代码，batch处用numpy的矩阵乘法可以有效加快速度，实测中大约快了5倍。

```python
# multi_class{‘auto’, ‘ovr’, ‘multinomial’}, default=’auto’
multi_class='multinomial'
clf = LogisticRegression(random_state=0,multi_class=multi_class,max_iter=1000)
clf.fit(X_train,y_train)
```



## 实验

没有做详细的试验，就进行了一次测试。```macro```的意思是分别对每一类进行统计，最后平均。

```python
predict=clf.predict(X_val)

acc=accuracy_score(y_val,predict)
precision=precision_score(y_val,predict,average='macro')
recall=recall_score(y_val,predict,average='macro')
f1=f1_score(y_val,predict,average='macro')

print('acc:{0},precision:{1},recall:{2},f1:{3}.'.format(acc,precision,recall,f1))
```

如下是模型在验证集上的各种指标，和前面那篇文章对比，可以看到有非常明显的提升,可能是因为有做了特征选择。

```
acc:0.6510957324106112,precision:0.599869910409861,recall:0.4847234444959456,f1:0.5226272120327005.
```

最后提交到Kaggle上的分数是0.61。

```python
df_test['Sentiment']=clf.predict(test_X)
df_test.to_csv('./submission.csv',index=False,columns=['PhraseId','Sentiment'])
```





### 引用

BOG&N-gram模型特征构造：

https://scikit-learn.org/stable/modules/feature_extraction.html

Softmax Regression分类器

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

sklearn评价指标：

https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

https://blog.csdn.net/Yqq19950707/article/details/90169913