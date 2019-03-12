# This Python file uses the following encoding: utf-8
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F    
# 激励函数都在这
# x data (tensor), shape=(100, 1)
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  
#unsqueeze将一维数据转化为二维，Pytorch只能处理二维数据
y = x.pow(2) + 0.2*torch.rand(x.size())
# noisy y data (tensor), shape=(100, 1)
# 用 Variable 来修饰这些数据 tensor，神经网络只能输入Variable
x, y = torch.autograd.Variable(x), Variable(y)
# 画图（为何必须加。numpy（））
#plt.scatter(x.data.numpy(), y.data.numpy())
#scatter打印散点图
#plt.show()

class Net(torch.nn.Module):#①
#class定义netural network
#继承torch.nn.Module，从而得到很多功能
	#定义所有的层属性（_init_()）
	#def _init_(self):#重要模块一：我们搭建我们这些层所需要的信息#②
	def __init__(self, n_feature, n_hidden, n_output):#加上输入的参数
	#为啥非要两条短横线啊！！！！！！！！
		super(Net, self).__init__()#继承Net到torch.nn.Module,继承_init_()的功能#③
	#①②③为官方步骤，必须这么做
		 # 定义每层用什么样的形式
		 #定义一层神经网络，输入，输出神经元节点数目
		self.hidden = torch.nn.Linear(n_feature,n_hidden)
        # 隐藏层线性输出
        #torch.nn.Linear()为线性变换层，对输入层做一个线性变换
        #预测神经层，输入为隐藏层神经元数目，输出为预测n_output(为1)
		self.predict = torch.nn.Linear(n_hidden, n_output)   # 输出层线性输出
        #def _init_()只是定义好了这些层，并非搭建好了网络，真正搭建在forward（）中
#
	#一层层搭建（forward（x））层与层的关系链接
	#将_init_定义的这些信息放forward中一个个组合起来，forward在Torch中搭建流程图
	def forward(self,x):#重要模块二：神经网络前向传递的过程
		  # 这同时也是 Module 中的 forward 功能
		  # x为正向传播输入值,输入信息 神经网络分析出输出值
		  #hidden layer加工一下输入信息x,再利用relu激活一下
		x = F.relu(self.hidden(x))      # 激励函数(隐藏层的线性值)
          #输出层接受x处理后输出
          #输出层不用激活函数，因为预测值一般从-无穷到+无穷，激活函数会将其截断
		x = self.predict(x)             # 输出值
		return x

#以上神经网络层便已经搭建好了
#输入层x->隐藏层hidden—>激活relu->输出层predict

net = Net(n_feature=1, n_hidden=10, n_output=1)
#输入值有一个，隐藏层有10个神经元，输出层有一个输出
print net


#可视化
#设置实时打印过程
plt.ion()   # 画图
plt.show()


#训练网络（优化网络）
# optimizer 是训练的工具
#使用torch.optim.SGD()模块优化参数，即传入网络参数net.parameters()--神经网络所有参数
#给定lr学习效率，lr越大，学习得越快，但是会忽略越多的东西，故一般lr均选择小于1的数
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)  # 传入 net 的所有参数, 学习率
#计算误差MSELoss()计算均方差，处理回归问题，分类问题用另外一种loss函数计算
loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)
#开始训练设置训练步数：100
for t in range(100):
	prediction = net(x)     # 喂给 net 训练数据 x, 输出每一步的预测值
	loss = loss_func(prediction, y) # 计算真实值y和预测值prediction两者的误差,注意顺序
    #优化步骤：下三步，清零，计算，赋值
	optimizer.zero_grad()   # 清空上一步的残余更新参数值（每一次计算loss之后都会保存在optimizer中
	loss.backward()         # 误差反向传播, 计算参数更新值
	optimizer.step()        # 将参数更新值施加到 net 的 parameters 上，以lr优化网络
	#可视化训练过程

	if t % 5 == 0:#每学习5步打印一次
        # plot and show learning process
		plt.cla()
		plt.scatter(x.data.numpy(), y.data.numpy())#原始数据
        #预测数据，神经网络学习到什么程度
		plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        #神经网络学习误差是多少
		plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color':  'red'})
		plt.pause(0.1)