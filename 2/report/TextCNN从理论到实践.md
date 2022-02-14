# TextCNN：理解与实践

参考：[TextCNN的PyTorch实现](https://cloud.tencent.com/developer/article/1548999)

主要需要更改的地方是

1. 数据读取部分我用的torchtext并且需要设定长度.
2. 在maxpool里面设置windows的大小的长度是句子长度-1而不是2.

详细更改内容参考代码。

流程图：

![image-20220119144307569](https://gitee.com/AICollector/picgo/raw/master/image-20220119144307569.png)

输入：[batch_size, seqence_length]

经过embedding：[batch_size, sequence_length, embedding_size]

使用：unsqueeze(1)，变成CNN可以进行卷积的维度[batch_size, 1, sequence_length, embedding_size]这个1相当于是channel的维度。本来卷积的CNN的输入数据就是[batch_size, in_channel, height, width]，这里的channel就是RGB。





出问题：

```
RuntimeError: Function AddmmBackward returned an invalid gradient at index 1 - got [128, 3] but expected shape compatible with [128, 147]
```



torch.Size([128, 100, 50])  embedding_X.shape  [batch_size, sequence_length, embedding_size]

torch.Size([128, 1, 100, 50]) embedding_X.unsqueeze(1)   add channel(=1) [batch, channel(=1), sequence_length, embedding_size]

 ==torch.Size([128, 3, 49, 1]) conv [batch_size, output_channel, 1, 1]==

 torch.Size([128, 147]) [batch_size, output_channel*1*1]

 torch.Size([128, 5]) 

torch.Size([128])

在卷积的那步出错了，卷积的结果应当是[128,3,1,1]。



