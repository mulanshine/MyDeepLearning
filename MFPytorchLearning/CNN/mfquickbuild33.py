# This Python file uses the following encoding: utf-8
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F    # 激励函数都在这
# x data (tensor), shape=(100, 1)

#建立数据集
# 假数据
n_data = torch.ones(100, 2)         # 数据的基本形态
x0 = torch.normal(2*n_data, 1)      # 类型0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # 类型0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)     # 类型1 x data (tensor), shape=(100, 1)
y1 = torch.ones(100)                # 类型1 y data (tensor), shape=(100, 1)

# 注意 x, y 数据的数据形式是一定要像下面一样 (torch.cat 是在合并数据)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # LongTensor = 64-bit integer
x, y = torch.autograd.Variable(x), Variable(y)
#plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=10, lw=0, cmap='RdYlGn')
#plt.show()

#建立神经网络
#method 1
class Net(torch.nn.Module):     # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # 隐藏层线性输出
        self.out = torch.nn.Linear(n_hidden, n_output)       # 输出层线性输出

    def forward(self, x):
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))      # 激励函数(隐藏层的线性值)
        x = self.out(x)                 # 输出值, 但是这个不是预测值, 预测值还需要再另外计算
        return x
#method 2
net2=torch.nn.Sequential(#Sequential 的意思是直接累神经层就可以了
    torch.nn.Linear(2,10),#加入一个神经层，输入输出和上面一致
    torch.nn.ReLU(),#将激励层当神经层输入进去
    torch.nn.Linear(10,2),#再加入一个神经层
    )
net1 = Net(n_feature=2, n_hidden=10, n_output=2) # 几个类别就几个 output
print(net1)  # net 的结构
print(net2)
"""
method 1和2的功能是一样的，输出形式不同
Net (
  (hidden): Linear (2 -> 10)#定义层的时候定义了一个属性叫hidden，当做索引
  (out): Linear (10 -> 2)
)
#上无relu层，是因为F.relu（）定义的是一个函数、方法，是没有名字的
Sequential (
  (0): Linear (2 -> 10)
  (1): ReLU ()#这里ReLU是当做一个层的类，所以有名字，但是两个的功能是完全一致的
  (2): Linear (10 -> 2)
)
"""