# LSTM：理解与实践



### LSTM单元架构图

![](https://gitee.com/AICollector/picgo/raw/master/20220117214424.png)

### 各单元公式

三门，二元，一状态。

##### 输入门、忘记门和输出门

三个公式一样的，通过激活函数归一到0和1之间了。
$$
\begin{aligned}
\mathbf{I}_{t} &=\sigma\left(\mathbf{X}_{t} \mathbf{W}_{x i}+\mathbf{H}_{t-1} \mathbf{W}_{h i}+\mathbf{b}_{i}\right) \\
\mathbf{F}_{t} &=\sigma\left(\mathbf{X}_{t} \mathbf{W}_{x f}+\mathbf{H}_{t-1} \mathbf{W}_{h f}+\mathbf{b}_{f}\right) \\
\mathbf{O}_{t} &=\sigma\left(\mathbf{X}_{t} \mathbf{W}_{x o}+\mathbf{H}_{t-1} \mathbf{W}_{h o}+\mathbf{b}_{o}\right)
\end{aligned}
$$

##### 候选记忆元、记忆元

候选记忆元表示本阶段的记忆元，后续和输入门进行权重缩放。
$$
\tilde{\mathbf{C}}_{t}=\tanh \left(\mathbf{X}_{t} \mathbf{W}_{x c}+\mathbf{H}_{t-1} \mathbf{W}_{h c}+\mathbf{b}_{c}\right)
$$
记忆元参与下一阶段的训练。
$$
\mathbf{C}_{t}=\mathbf{F}_{t} \odot \mathbf{C}_{t-1}+\mathbf{I}_{t} \odot \tilde{\mathbf{C}}_{t}
$$

##### 隐状态

$$
\mathbf{H}_{t}=\mathbf{O}_{t} \odot \tanh \left(\mathbf{C}_{t}\right)
$$

### 各单元作用

- 输入门it: it控制当前词xt的信息融入记忆单元ct。在理解一句话时，当前词xt可能对整句话的意思很重要，也可能并不重要。输入门的目的就是判断当前词xt对全局的重要性。当it开关打开的时候，网络将不考虑当前输入xt。它控制了新数据参与训练的量。
- 遗忘门ft: ft控制上一时刻记忆单元ct-1的信息融入记忆单元ct。在理解一句话时，当前词xt可能继续延续上文的意思继续描述，也可能从当前词xt开始描述新的内容，与上文无关。和输入门it相反， ft不对当前词xt的重要性作判断， 而判断的是上一时刻的记忆单元ct-1对计算当前记忆单元ct的重要性。当ft开关打开的时候，网络将不考虑上一时刻的记忆单元ct-1。
- 输出门ot: 输出门的目的是从记忆单元ct产生隐层单元ht。并不是ct中的全部信息都和隐层单元ht有关，ct可能包含了很多对ht无用的信息，因此， ot的作用就是判断ct中哪些部分是对ht有用的，哪些部分是无用的。
- 记忆单元ct：ct综合了当前词xt和前一时刻记忆单元ct-1的信息。这和ResNet中的残差逼近思想十分相似，通过从ct-1到ct的”短路连接”， 梯度得已有效地反向传播。 当ft处于闭合状态时， ct的梯度可以直接沿着最下面这条短路线传递到ct-1，不受参数W的影响，这是LSTM能有效地缓解梯度消失现象的关键所在。

### 深度循环神经网络

上面的图片多是一个单元的内容，但是在pytorch的架构中，lstm可以进行深度的叠加。如下是深度循环神经网络的架构图：

![image-20220118120936333](https://gitee.com/AICollector/picgo/raw/master/image-20220118120936333.png)

输入仍然是具有时间序列的XT，三门二元的计算方式仍然是不变的，但是输出门没有计算的必要了，因为隐藏的层的计算方式变化了：设置$H^{(0)}_t=Xt$， 第$l$个隐藏层的隐状态使用激活函数$ϕ_l$，则:
$$
\mathbf{H}_{t}^{(l)}=\phi_{l}\left(\mathbf{H}_{t}^{(l-1)} \mathbf{W}_{x h}^{(l)}+\mathbf{H}_{t-1}^{(l)} \mathbf{W}_{h h}^{(l)}+\mathbf{b}_{h}^{(l)}\right)
$$
简单来说就是隐状态由上一个深度同一个时间的隐状态和上一个时间同一个深度的隐状态决定。隐状态横向（向右）堆叠的输出是最终的隐状态结果，纵向（向上）堆叠的结果是输出。

### 双向循环神经网络

模型的架构图如下所示：

![image-20220118122925213](https://gitee.com/AICollector/picgo/raw/master/image-20220118122925213.png)

原理就是搞两个不同方向但是有同样结构的LSTM。
$$
\begin{aligned}
&\overrightarrow{\mathbf{H}}_{t}=\phi\left(\mathbf{X}_{t} \mathbf{W}_{x h}^{(f)}+\overrightarrow{\mathbf{H}}_{t-1} \mathbf{W}_{h h}^{(f)}+\mathbf{b}_{h}^{(f)}\right) \\
&\overleftarrow{\mathbf{H}}_{t}=\phi\left(\mathbf{X}_{t} \mathbf{W}_{x h}^{(b)}+\overleftarrow{\mathbf{H}}_{t+1} \mathbf{W}_{h h}^{(b)}+\mathbf{b}_{h}^{(b)}\right)
\end{aligned}
$$
接下来, 将前向隐状态 $\overrightarrow{\mathbf{H}}_{t}$ 和反向隐状态 $\overleftarrow{\mathbf{H}}_{t}$ 连接起来, 获得需要送入输出层的隐状态 $\mathbf{H}_{t} \in \mathbb{R}^{n \times 2 h}$ 。在具 有多个隐藏层的深度双向循环神经网络中, 该信息作为输入传递到下一个双向层。最后, 输出层计算得到 的输出为 $\mathbf{O}_{t} \in \mathbb{R}^{n \times q}$ ( $q$ 是输出单元的数目) :
$$
\mathbf{O}_{t}=\mathbf{H}_{t} \mathbf{W}_{h q}+\mathbf{b}_{q}
$$

### LSTM(PYTORCH实现)

#### 官方api：

- 参数
  * **inp ut_size**：The number of expected features in the input x。输入的向量的长度，即word embedding的向量长度，用于表示每个单词的feature的长度。
  *  **hidden_size**：The number of features in the hidden state h。隐藏状态h的变量长度，反应的是模型的一个复杂程度。
  *  **num_layers**： Number of recurrent layers. E.g., setting `num_layers=2` would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1。深度lstm的参数，是否使用多层的lstm。
  * **bias**： If `False`, then the layer does not use bias weights b_ih and b_hh. Default: `True`。是否带上偏差。
  *  **batch_first**：If `True`, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). Note that this does not apply to hidden or cell states. See the Inputs/Outputs sections below for details. Default: `Fals`。一个trick让batch可以被并行运算。
  *  **dropout**： If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to `dropout`. Default: 0。是否使用dropout机制，即随机地丢弃部分参数。
  *  **bidirectional**： If `True`, becomes a bidirectional LSTM. Default: `False`。是否使用双向LSTM。
- 输入
  -  **input** (seq_len, batch, input_size)
  -  **h_0** (num_layers * num_directions, batch, hidden_size)
  -  **c_0** (num_layers * num_directions, batch, hidden_size)
- 输出
  – **output** (seq_len, batch, num_directions * hidden_size)
  – **h_n** (num_layers * num_directions, batch, hidden_size)
  – **c_n** (num_layers * num_directions, batch, hidden_size)



#### pack&pad

我们需要让lstm知道序列的长度，到序列结束的时候的time step就不要继续更新。

如下是完整的例子：

```python
import torch as t
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

a = t.tensor([[1,2,3],[6,0,0],[4,5,0]]) #(batch_size, max_length)
lengths = t.tensor([3,1,2])

# 排序
a_lengths, idx = lengths.sort(0, descending=True)
_, un_idx = t.sort(idx, dim=0)
a = a[un_idx]

# 定义层 
emb = t.nn.Embedding(20,2,padding_idx=0) 
lstm = t.nn.LSTM(input_size=2, hidden_size=4, batch_first=True) 

a_input = emb(a)
a_packed_input = t.nn.utils.rnn.pack_padded_sequence(input=a_input, lengths=a_lengths, batch_first=True)
packed_out, _ = lstm(a_packed_input)
out, _ = pad_packed_sequence(packed_out, batch_first=True)
# 根据un_idx将输出转回原输入顺序
out = t.index_select(out, 0, un_idx)
```

查看输出：

```python
a_input.shape: torch.Size([3, 3, 2])# 经过embedding的输入大小
a_packed_input.data.shape torch.Size([6, 2])# pack过后的输入大小，即已经是正常的序列
packed_out.data.shape: torch.Size([6, 4])# 隐藏层的输出
out.shape: torch.Size([3, 3, 4])#经过拆解之后的输出
```

总之就是用一个index记录每个样本的位置然后平滑放到lstm中训练，最后训练的结果再展开成batch的形式。



### 引用：

1. [长短期记忆网络（LSTM）(动手学习深度学习)](https://zh.d2l.ai/chapter_recurrent-modern/lstm.html)

2. [一招理解LSTM/GRU门控机制](https://cloud.tencent.com/developer/article/1109746)
3. [LSTM细节分析理解（pytorch版）](https://zhuanlan.zhihu.com/p/79064602)









