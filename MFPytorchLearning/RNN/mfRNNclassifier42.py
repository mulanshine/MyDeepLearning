# -*-coding:utf-8-*-
import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 64		# 批处理设置大小
TIME_STEP = 28      # rnn 时间步数 / 图片高度（RNN考虑多少个时间点数据）
INPUT_SIZE = 28     # rnn 每步输入值 / 图片每行像素（每个时间点数据上面给RNN多少个数据点）
# 输入图像28*28，即一步（TIME_STEP中一步）读取一行（输入有28个像素点，即INPUT_SIZE = 28），读取28步（TIME_STEP=28）
LR = 0.01           # learning rate
DOWNLOAD_MNIST = True  # 如果你已经下载好了mnist数据就写上 Fasle

# Mnist 手写数字
train_data = dsets.MNIST(
    root='./mnist/',    # 保存或者提取位置
    train=True,  # this is training data
    transform=transforms.ToTensor(),    # 转换 PIL.Image or numpy.ndarray 成
                                        # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_MNIST,          # 没下载就下载, 下载了就不用再下了
)

# plot one example
print(train_data.train_data.size())     # (60000, 28, 28)
print(train_data.train_labels.size())   # (60000)
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()

# Data Loader for easy mini-batch return in training
# 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# convert test data into Variable, pick 2000 samples to speed up testing
test_data = dsets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())
# 为了节约时间, 我们测试时只测试前2000个
test_x = Variable(test_data.test_data, volatile=True).type(torch.FloatTensor)[:2000]/255.
# shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels.numpy().squeeze()[:2000]  # covert to numpy array

# 我们用一个 class 来建立 RNN 模型. 这个 RNN 整体流程是
# 1.(input0, state0) -> LSTM -> (output0, state1);
# 2.(input1, state1) -> LSTM -> (output1, state2);
# 3.…
# 4.(inputN, stateN)-> LSTM -> (outputN, stateN+1);
# 5.outputN -> Linear -> prediction. 通过LSTM分析每一时刻的值, 并且将这一时刻和前面时刻的理解合并在一起,
# 生成当前时刻对前面数据的理解或记忆. 传递这种理解给下一时刻分析.


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()  # 继承Module的功能

        self.rnn = nn.LSTM(     # LSTM效果要比 nn.RNN() 好多了
            input_size=INPUT_SIZE,      # 图片每行的数据像素点
            hidden_size=64,     # rnn hidden unit
            num_layers=1,       # 有几层 RNN layers，节省时间取为1
            batch_first=True,   # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size),batch放在第一个
        )

        self.out = nn.Linear(64, 10)    # 输出层

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)              # None 表示 hidden state 会用全0的 state

        # 选取最后一个时间点的 r_out 输出，r_out 有28个output，读完一行一个
        # 这里 r_out[:, -1, :] 的值也是 h_n 的值
        out = self.out(r_out[:, -1, :])                   # 选取最后一个时间点的output
        return out

rnn = RNN()
print(rnn)
"""
RNN (
  (rnn): LSTM(28, 64, batch_first=True)
  (out): Linear (64 -> 10)
)
"""
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # 选择Adam来优化rnn optimize all parameters
loss_func = nn.CrossEntropyLoss()                       # 使用CrossEntropyLoss计算loss_func，目标标签非二进制表示

# training and testing
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):        # gives batch data
        b_x = Variable(x.view(-1, 28, 28))              # reshape x to (batch, time_step, input_size)
        b_y = Variable(y)                               # batch y

        output = rnn(b_x)                               # rnn output
        loss = loss_func(output, b_y)                   # cross entropy loss
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients

        if step % 50 == 0:
            test_output = rnn(test_x)                   # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
            accuracy = sum(pred_y == test_y) / float(test_y.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)

        """
                        ...
                        Epoch:  0 | train loss: 0.0945 | test accuracy: 0.94
                        Epoch:  0 | train loss: 0.0984 | test accuracy: 0.94
                        Epoch:  0 | train loss: 0.0332 | test accuracy: 0.95
                        Epoch:  0 | train loss: 0.1868 | test accuracy: 0.96
        """
# print 10 predictions from test data
test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
"""
[7 2 1 0 4 1 4 9 5 9] prediction number
[7 2 1 0 4 1 4 9 5 9] real number
"""
