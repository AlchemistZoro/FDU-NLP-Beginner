# TorchText使用指南

torchtext 的安装。安装出问题主要是版本不对。直接使用```pip install torchtext```。为了让版本配对，会同时下载torch库。如果使用的是anaconda的话，注意用```conda list```而非```pip list``` 查看torch版本。或者在python中查看：

```python
import torch
from torch import nn
print(torch.__version__) #1.7.0
```

再在[官方对应表](https://github.com/pytorch/text)中找到torch对应的torchtext版本：

![image-20220116173734226](https://gitee.com/AICollector/picgo/raw/master/image-20220116173734226.png)

出问题torch的版本好像被改变为cpu版本了，直接重新创建环境。重新创建环境然后再pip install 对应的版本即可。



torchtext使用指南

https://www.pythonf.cn/read/128035

中文

https://blog.csdn.net/u012436149/article/details/79310176

英文对照

http://anie.me/On-Torchtext/

