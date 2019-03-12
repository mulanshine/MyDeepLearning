# This Python file uses the following encoding: utf-8
import torch
import torch.utils.data as Data#torch 的一个模块
#批训练，训练数据量较大时，使用分批训练，效率高
torch.manual_seed(1)    # reproducible
BATCH_SIZE = 5      #批训练中一小批的数量
x = torch.linspace(1, 10, 10)       # x data (torch tensor)，十个数据
y = torch.linspace(10, 1, 10)       # y data (torch tensor)

# 先转换成 torch 能识别的 Dataset
#将数据x，y放进数据库中
#data_tensor=x，用于训练的数据x
#target_tensor=y，用于计算误差的数据y
torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)

# 把 dataset 放入 DataLoader

#使用loader将我们的训练变成一小批一小批的
#Data.DataLoader使用致使批训练
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format,数据传入其中
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 训练时要不要打乱数据再抽样 (True打乱比较好)
    num_workers=2,              # 多线程来读数据
)

for epoch in range(3):   # 训练所有!整套!数据 3 次
    for step, (batch_x, batch_y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
    	#enumerate()表示提供一个索引step
        #training....
        # 假设这里就是你训练的地方...

        # 打出来一些数据
        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
              batch_x.numpy(), '| batch y: ', batch_y.numpy())

