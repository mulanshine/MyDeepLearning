# This Python file uses the following encoding: utf-8
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F 
#function:保存和提取神经网络，使得今天建立训练的网络，关机后，明天还能提取接着用

torch.manual_seed(1)#reproducible
#fake data
x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1) #x data(tensor),shape=(100,1)
y=x.pow(2)+0.2*torch.rand(x.size())#noisy y data(tensor),shape=(100,1)
x,y=Variable(x,requires_grad=False),Variable(y,requires_grad=False)

#保存功能
def save():
	#saave net1
	#快速搭建法搭建神经网络
	net1=torch.nn.Sequential(
		torch.nn.Linear(1,10),
		torch.nn.ReLU(),
		torch.nn.Linear(10,1)
		)
	optimizer=torch.optim.SGD(net1.parameters(),lr=0.5)#训练神经网络（回归）
	loss_func=torch.nn.MSELoss()

	for t in range(100):
		prediction=net1(x)
		loss=loss_func(prediction,y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	torch.save(net1,'net.pkl')#save entire net
	torch.save(net1.state_dict(),'net_params.pkl')#save all parameters of netural network
	
	#plot result
	plt.figure(1,figsize=(10,3))
	plt.subplot(131)
	plt.title('Net1')
	plt.scatter(x.data.numpy(),y.data.numpy())
	plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
#提取功能 1
def restore_net():
	net2=torch.load('net.pkl')#提取保存的神经网络
	prediction=net2(x)#出图将prediction加上去
	plt.subplot(132)
	plt.title('Net2')
	plt.scatter(x.data.numpy(),y.data.numpy())
	plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)#呈现出prediction

#提取功能 2
#method:先建立与net1一样的网络结构，再将参数换为保存的参数，过程比方法1要快（推荐）
def restore_params():
	net3=torch.nn.Sequential(
		torch.nn.Linear(1,10),
		torch.nn.ReLU(),
		torch.nn.Linear(10,1)
		)
	net3.load_state_dict(torch.load('net_params.pkl'))#与上save_state_dict()相呼应
	prediction=net3(x)
	
	plt.subplot(133)
	plt.title('Net3')
	plt.scatter(x.data.numpy(),y.data.numpy())
	plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)#呈现出prediction
	plt.show()

save()
restore_net()
restore_params()