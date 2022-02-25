# NLP-Beginner Code&Report

[NLP-Beginner](https://github.com/FudanNLP/nlp-beginner)是复旦大学自然语言处理组发布的自然语言入门练习项目。此项目包含了完成该项目中5道题所需的代码与报告。

### 运行提示

* 一级文件目录的标号即为任务的编号。
* data文件放原始数据与预处理过后的数据。
* model文件夹保存模型的breakpoint。
* .vector_cache文件夹放预训好的词向量模型。

### 任务一：基于机器学习的文本分类

此任务要求仅用numpy实现基于logistic/softmax regression的文本分类。为了熟悉机器学习框架sklearn，在此任务的基础上附加了利用sklearn完成文本分类的实现。

基于numpy的任务实现：[代码](./1/numpy_base.ipynb)、[报告](./1/report/利用逻辑回归进行文本分类.md)

基于sklearn框架的任务实现：[代码](./1/sklearn_base.ipynb)、[报告](./1/report/利用sklearn进行文本分类.md)

### 任务二：基于深度学习的文本分类

熟悉Pytorch，用Pytorch重写《任务一》，实现基于CNN、RNN的文本分类。

基于LSTM的任务实现：[代码](./2/LSTM_classify.ipynb)

基于TextCNN的任务实现代码：[代码](./2/textCNN_classify.ipynb)

整体任务实现：[报告](./2/report/基于深度学习的文本分类.md)

关于任务中所用到的相关模型以及框架的笔记：

* [LSTM:从理论到实践](./2/report/Lstm从理论到实践.md)

* [TextCNN从理论到实践](./2/report/TextCNN从理论到实践.md)

* [TorchText使用指南](./2/report/TorchText使用指南.md)

### 任务三：基于注意力机制的文本匹配

输入两个句子判断，判断它们之间的关系。参考[ESIM]( https://arxiv.org/pdf/1609.06038v3.pdf)（可以只用LSTM，忽略Tree-LSTM），用双向的注意力机制实现。为了实现本任务我参考了开源项目：[pytorch-nli](https://github.com/bentrevett/pytorch-nli)，了解NLI基本的建模技巧。

基于双向注意力机制的文本匹配：[代码](./3/code/ESIM.ipynb)、[报告](./3/基于注意力机制的文本匹配.md)

关于任务中所用到的相关模型以及框架的笔记：

* [理解注意力机制](./3/note/理解注意力机制.md)


### 任务四：基于LSTM+CRF的序列标注

用LSTM+CRF来训练序列标注模型：以Named Entity Recognition为例。

基于LSTM+CRF的序列标注：[代码](./4/code/BiLSTM_CRF.ipynb)、[报告](./4/基于LSTM&CRF的序列标注.md)

关于任务中所用到的相关模型以及框架的笔记：

* [CRF理解与实践](./4/CRF理解与实践.md)

### 任务五：基于神经网络的语言模型

用LSTM、GRU来训练字符级的语言模型，计算困惑度

基于神经网络的语言模型：[代码](./5/code/GenerateModel.ipynb)、[报告](./5/基于神经网络的语言模型.md)

### 附加任务：基于LSTM的对对联机器人

此任务为受任务三与任务五启发，利用较大型的对联数据集，基于BiLSTM训练对对联机器人。

项目链接：[AIVTuber1.0-CoupletBot](https://github.com/seu-wll/AIVTuber1.0-CoupletBot)







