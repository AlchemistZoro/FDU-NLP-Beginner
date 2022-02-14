### 任务四：基于LSTM+CRF的序列标注

用LSTM+CRF来训练序列标注模型：以Named Entity Recognition为例。

1. 参考
   1. 《[神经网络与深度学习](https://nndl.github.io/)》 第6、11章  第6章就是LSTM没有必要看，第11章是讲概率图模型从p268-p300。
   2. https://arxiv.org/pdf/1603.01354.pdf
   3. https://arxiv.org/pdf/1603.01360.pdf
2. 数据集：CONLL 2003，https://www.clips.uantwerpen.be/conll2003/ner/
3. 实现要求：Pytorch
4. 知识点：
   1. 评价指标：precision、recall、F1
   2. 无向图模型、CRF
5. 时间：两周

---

代码参考：

https://github.com/liu-nlper/SLTK 过于详细，看不太懂。

https://github.com/ZhixiuYe/NER-pytorch 看着还行，不知道代码是否是正确的。

https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html pytorch官方参考

文章理解：

序列标注：Bi-LSTM + CRF - 希葛格的韩少君的文章 - 知乎 https://zhuanlan.zhihu.com/p/42096344

如何用简单易懂的例子解释条件随机场（CRF）模型？它和HMM有什么区别？ - 知乎 https://www.zhihu.com/question/35866596



---

关于数据集：

https://blog.csdn.net/Elvira521yan/article/details/118028020

数据集的标注：

```
[word][POS tag][chunk tag][NER tag]
```

如：

```PYTHON
U.N. NNP I-NP I-ORG #组织的实体之中
  official NN I-NP O 
  Ekeus NNP I-NP I-PER # 人的实体之中
  heads VBZ I-VP O 
  for IN I-PP O 
  Baghdad NNP I-NP I-LOC # 人的实体之中
  . . O O  # 表示句子的结束 
```

我们需要做的任务是NER任务，所以可以不用管那个POS。

英语标注数据的数据量：

```
	文章数	句子数	词语数
训练集	946	14987	203621
开发集	216	3466	51362
测试集	231	3684	46435
```



---

论文阅读：

### *End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF*

有点麻烦，原理和下面的那篇差不多。

### *Neural Architectures for Named Entity Recognition*

搭建模型主要用这个模板。

#### LSTM-CRF Model

2.1LSTM

一个双向的LSTM。

2.2 CRF Tagging Models

需要用CRF的原因是因为任务本身决定的，因为命名的原因。像I-PER不能接在B-loc的后面，相当于是有一个先验的规则。需要再接一层CRF从而避免掉这种错误。文中对这类任务的描述是：

> strong dependencies across output labels

输入：
$$
\mathbf{X}=\left(\mathrm{x}_{1}, \mathrm{x}_{2}, \ldots, \mathrm{x}_{n}\right)
$$
对于出现（X,y）这个点对我们给一个分数。分数的公式为：
$$
s(\mathbf{X}, \mathbf{y})=\sum_{i=0}^{n} A_{y_{i}, y_{i+1}}+\sum_{i=1}^{n} P_{i, y_{i}}
$$
A是tag之间的转移概率。A是一个我们需要学习的矩阵，也就是转移矩阵。在推断的时候我们直接用训练好的结果去进行计算。P是lstm输出的概率。of size n xk。这里的k是tag的种类数量，而在BiLstm中的输出应当是 [batch size,prem sent len,embedding dim*2]。

**问题：P的输出为什么是这个形状？**参考下面的论文



对于输入，我有可能有很多的输出也就是标注结果，用softmax的形式对所有的输出进行一个归一化。

条件概率表达式：
$$
p(\mathbf{y} \mid \mathbf{X})=\frac{e^{s(\mathbf{X}, \mathbf{y})}}{\sum_{\widetilde{\mathbf{y}} \in \mathbf{Y}_{\mathbf{X}}} e^{s(\mathbf{X}, \widetilde{\mathbf{y}})}}
$$
我们的目标就是最大化正确标注的条件概率，同时让别的预测结果的值尽量小。如下是预测的公式：
$$
\begin{aligned}
\log (p(\mathbf{y} \mid \mathbf{X})) &=s(\mathbf{X}, \mathbf{y})-\log \left(\sum_{\tilde{\mathbf{y}} \in \mathbf{Y}_{\mathbf{X}}} e^{s(\mathbf{X}, \tilde{\mathbf{y}})}\right) \\
&=s(\mathbf{X}, \mathbf{y})-\underset{\tilde{\mathbf{y}} \in \mathbf{Y}_{\mathbf{X}}}{\operatorname{logadd}} s(\mathbf{X}, \widetilde{\mathbf{y}}),
\end{aligned}
$$
那个logaddexp，pytorch里面有相关的函数实现。在运算的时候这个函数要加负号，因为是最小化。

在decode的时候，也就是预测的时候，我们用可能产生的最大的分数的标签。
$$
\mathbf{y}^{*}=\underset{\widetilde{\mathbf{y}} \in \mathbf{Y}_{\mathbf{X}}}{\operatorname{argmax}} s(\mathbf{X}, \widetilde{\mathbf{y}})
$$
因为用的是bigram的方法，所以前面两个算法都可以用动态规划的方式进行解决。

那么这里就是用的是维特比算法了。

2.4 Tagging Schema：

介绍了两种实体命名的方式，IOB和IOBES。他们用的是IOBES，但是没有什么提升。如果我实验的时候就直接用官方给的就行了。

跳过了3 ，3是讲的是另外的一种方法。

#### 4 InputWord Embeddings

文中乱七八糟的搞得不是很懂。有提到用wordvector的做法，按照我自己的理解我就直接用glove 6来训练了。



#### 5 Train

隐层维度设置为100

dropout rate设置为0.5



### *Finding Function in Form: Compositional Character Models for*

这篇讲的就是如何构建出好的词语表示以适配各种各样的网络。也就是解决的P的问题，我们直接来看其解决方案。

对于一个输入：
$$
\mathbf{e}_{c_{1}}^{C}, \ldots, \mathbf{e}_{c_{m}}^{C}
$$
Lstm的输出是：
$$
\mathbf{s}_{0}^{f}, \ldots, \mathbf{s}_{m}^{f}
$$
反向的lstm输出是：
$$
\mathbf{s}_{m}^{b}, \ldots, \mathbf{s}_{0}^{b}
$$
把两个方向的LSTM的hidden layer加起来就是结果：
$$
\mathbf{e}_{w}^{C}=\mathbf{D}^{f} \mathbf{s}_{m}^{f}+\mathbf{D}^{b} \mathbf{s}_{0}^{b}+\mathbf{b}_{d}
$$
那么可以让输出结果变成m*w的维数，这里的m就是词语的序列数，w是tag的种类数。对应的是前面提到的n和k。

---



代码实现部分参考了

https://github.com/ZhixiuYe/NER-pytorch 

这个是一个韩国人写的代码。也挺多的，也看不太懂的样子。

我这边用的方法就比较简单就是直接用word embedding的方式+bilstm+CRF。论文中提到的基于词的embedding就暂时没有实现。

---

#### 数据读取

训练集：eng.train

验证集：eng.testa

测试集：eng.testb

word_embedding: glove.6B.100d.txt

文本的读取是用普通的方法实现，后续的处理用torchtext实现。读取出来成为batch的形式。

#### 模型构造

LSTM的架构：

如果说用了chart_embedding/cap_embedding之类的就如下构造：

```python
embeds = torch.cat((embeds, chars_embeds, cap_embedding), 1)
```

当然我就直接用word_embedding了。

对于最后的hidden2tag也就是开始说的从nxd到nxk+2。

```
__init__：
	self.hidden2tag = nn.Linear(hidden_dim*2, self.tagset_size)
forward:
	lstm_feats = self.hidden2tag(lstm_out)
```





#### 训练

训练的过程和判别式的模型训练形式一样。都是模型出来一个loss，然后直接调用```.backward()```函数进行运算。

然后再optimizer接一下这样。

训练的时候调用```neg_log_likelihood```，其函数如下所示：

```
def neg_log_likelihood；
	1. _get_lstm_features 获得 feat：size：nx(k+2)
	2. _forward_alg 获得logexpsum的值
	3. _score_sentence 获得正确的socre的值
```

关于CRF和HMM的一些术语形式的表述：

```
观测序列（observations）：实际观测到的现象序列
隐含状态（states）：所有的可能的隐含状态
初始概率（start_probability）：每个隐含状态的初始概率
转移概率（transition_probability）：从一个隐含状态转移到另一个隐含状态的概率
发射概率（emission_probability）：某种隐含状态产生某种观测现象的概率
```

其中主要是```_forward_alg```难以看懂。如下是其代码，我们将结合计算公式进行理解：

```python
def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas
        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                print('emit_score:',emit_score)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha
```

如下是我们需要计算的公式：
$$
{\operatorname{logadd}} s(\mathbf{X}, \widetilde{\mathbf{y}})
$$
核心的思想就是动态规划。对于一个特定(n_i,k_i)的位置的状态，也就是说比如说第三个位置，如果是第2中tag的分数。那么这个分数就可以由前面的所有可能的tag转移过来，同时这些tag还要加上从这个tag转移到已经有的输出上面，也就是经典的推断问题。

即每一步，每一个状态下的分数=log_sum_exp(上一步的各状态分数向量+转移向量（从不同的状态转移到此状态的概率）+发射概率分数（从该状态转移到输出（固定）的概率）经过广播之后向量）

此状态下的分数变可以用于后续的计算。

同时需要注意的是starttag的初始化和endtag的收尾概率。利用动态规划的方式将复杂度降低到O(NK)级别。如下是每一步的公式计算过程。

![image-20220124134624285](https://gitee.com/AICollector/picgo/raw/master/image-20220124134624285.png)

#### 验证与测试

验证与测试的时候，我们需要计算出转移概率，我们是利用已经有的转移概率表进行推断，此时就用的是维特比算法。我对于维特比的印象就是时刻记录这每一步的转移分数，最后倒推回得到最终的计算分数。

---

### 实验

因为我的代码实在pytorch给出的官方代码上进行更改，不知道如何使用batch，因此实验速度上非常慢。

---

### 运用batch进行代码的构建

[batch实现博客参考](https://blog.csdn.net/baqnliaozhihui/article/details/109244094)

[batch实现代码参考](https://github.com/liaozhihui/MY_MODEL/blob/master/batch_bilstm_crf.py)

我这边简单说一下思路，具体的实现就暂时先不管了。如果是用batch的话，核心的步骤应当是和不用batch是一致的。会有不一样的地方是对于句子我们需要做一个padding，也就是说按照最长的句子进行处理，但是在计算的时候进行截断。在lstm的时候是可以加速的，但是在后面的两步的时候可能还是同样的做法不能加速。具体的实现等后续有空的时候在慢慢实践。

----

### 已有的代码

│   BiLSTM_CRF_BATCH.py  	用batch实现，代码参考在：[batch实现代码参考](https://github.com/liaozhihui/MY_MODEL/blob/master/batch_bilstm_crf.py)
│   BiLSTM_CRF_MODEL.py  	pytorch官网的代码实现，仅有模型
│   pytorch_copy.ipynb 	pytorch官网的代码实现运用本次实验的数据集 
│   pytorch_tutorial.ipynb  	 pytorch官网的代码实现

---

2022年1月24日15:25:41

傻逼了，pytorch居然有CRF的代码，早知道直接看就行了，真的是md。不过至少把东西理解了，只能说是不亏而已。

我还是想试着实现一下，如果能够做到趁现在还比较有印象直接做到位就行，应该也不是很麻烦的事情啊！

---

[Step-by-step NER Model for Bahasa Indonesia with PyTorch and Torchtext](https://yoseflaw.medium.com/step-by-step-ner-model-for-bahasa-indonesia-with-pytorch-and-torchtext-6f94fca08406)

该博客的第一个链接讲了一个读取tag的方法

[How to perform sequence labeling with pytorch?](https://colab.research.google.com/drive/1wxhnsG-l5ElsyneO-PV0eCXN8KvnaVyF?usp=sharing#scrollTo=Xtfi8kEesimF)



---

[RuntimeError: CUDA error: device-side assert triggered 问题的解决思路](https://blog.csdn.net/Penta_Kill_5/article/details/118085718)

下标出错了，改成tag_num=10就是对的答案。

按照官方的意思进行复现，但是并不能达到它的精度，F1最后的结果大约是86左右。这个坑还是挺多的。

