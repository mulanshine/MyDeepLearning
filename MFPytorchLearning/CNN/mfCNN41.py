# -*-coding:utf-8-*- 

import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F 
import torch.utils.data as Data

torch.manual_seed(1)    # reproducible
# Hyper Parameters
EPOCH = 1   # 数据量大，为节省时间只epoch=1?
BATCH_SIZE = 50   # 每批50个数据
LR=0.001      # 定义学习率
DOWNLOAD_MNIST = True #还未曾下载数据的话为TRUE，若下载过了则False

#从网络上下载数据手写数字数据MNIST
train_data = torchvision.datasets.MNIST(
	root='./mnist/',  # 下载下来存数据的目录，会自己新建一个目�?
	train=True,  # 下载traindata而不是测试集，False则下载测试集
	transform=torchvision.transforms.ToTensor(),   # 利用这个函数将下载的数据PIL.Image or numpy.ndarray
					                               # 改成所需要的数据格式，训练的时候从像素（0,255）区间normalize 到[0.0, 1.0] 区间
					                               # 彩色照片（0,255），颜色三个通道RGB，黑白照片，颜色一个通道
    download=DOWNLOAD_MNIST,                       # 没下载就下载, 下载了就不用再下载
	)
print train_data.train_data.size()
print train_data.train_labels.size()
#绘制出一个图看看
#plt.imshow(train_data.train_data[0].numpy(),cmap='gray')
#plt.title('%i' %train_data.train_labels[0])
#plt.show()

# 批训练50samples, 1 channel, 28x28 (50, 1, 28, 28)
# loader用来实现批训练
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
# 获取测试data
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)

# 为了节约时间, 我们测试时只测试�?000�?
# 测试集压缩至（0，1）
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255. 
  # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:2000]# 只取前2000个，节省时间

#建立CNN网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 前三步，必须要存在

        # 建立卷积层
        self.conv1 = nn.Sequential(  # 利用Sequential途径搭建卷积层
        # input shape (1, 28, 28)
            nn.Conv2d(# 卷积层，三维（长、宽、高：多少个filter，用来提取属性特征）
                in_channels=1,      # input height图片高度，rgb：3，gray：1
                out_channels=16,    # n_filters，输出高度，这里是卷积核个数
                kernel_size=5,      # filter size卷积核（5,5）
                stride=1,           # filter movement/step步长
                padding=2,      # 如果想要 con2d 出来的图片长宽没有变 padding=(kernel_size-1)/2 ；stride=1
            ),      # output shape (16, 28, 28)
            
            nn.ReLU(),    # activation，激活函数
            nn.MaxPool2d(kernel_size=2),    # 池化层，�?2x2 空间里向下采样取max, 降维output shape (16, 14, 14)
        )
        #又一层卷积层
        self.conv2 = nn.Sequential(  # input shape (1, 28, 28)
        	#16，接上一层，32输出层，核大小仍5*5，步�?，padding2
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x) #输入的图像
        x = self.conv2(x) #卷积2层（batch,32,7,7)
        x = x.view(x.size(0), -1)   # 展平多维的卷积图像(batch_size, 32 * 7 * 7),右边全变一列
        output = self.out(x)#10的输出
        return output,x

cnn = CNN()
print(cnn)  # net architecture
"""
CNN (
  (conv1): Sequential (
    (0): Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): ReLU ()
    (2): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
  )
  (conv2): Sequential (
    (0): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): ReLU ()
    (2): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
  )
  (out): Linear (1568 -> 10)
)
"""

#训练网络
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted

# following function (plot_with_labels) is for visualization, can be ignored if not interested
from matplotlib import cm
try: from sklearn.manifold import TSNE; HAS_SK = True
except: HAS_SK = False; print('Please install sklearn for layer visualization')
def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

plt.ion()

# training and testing
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):   # 分配 batch data, normalize x when iterate train_loader
		b_x = Variable(x)   # batch x
		b_y = Variable(y)   # batch y

		output = cnn(b_x)[0]              # cnn output
		loss = loss_func(output, b_y)   # cross entropy loss
		optimizer.zero_grad()           # clear gradients for this training step
		loss.backward()                 # backpropagation, compute gradients
		optimizer.step()                # apply gradients

		if step % 50==0:
			test_output,last_layer=cnn(test_x)
        	pred_y = torch.max(test_output, 1)[1].data.squeeze()
        	accuracy=sum(pred_y==test_y)/float(test_y.size(0))
        	print('Epoch:',epoch,'|train loss:%.4f' %loss.data[0],'|test accuracy:%.2f' %accuracy)
        	if HAS_SK:
                # Visualization of trained flatten layer (T-SNE)
				tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                plot_only = 500
                low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                labels = test_y.numpy()[:plot_only]
                plot_with_labels(low_dim_embs, labels)
plt.ioff()
"""
...
Epoch:  0 | train loss: 0.0306 | test accuracy: 0.97
Epoch:  0 | train loss: 0.0147 | test accuracy: 0.98
Epoch:  0 | train loss: 0.0427 | test accuracy: 0.98
Epoch:  0 | train loss: 0.0078 | test accuracy: 0.98
"""


test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')

"""
[7 2 1 0 4 1 4 9 5 9] prediction number
[7 2 1 0 4 1 4 9 5 9] real number
"""

