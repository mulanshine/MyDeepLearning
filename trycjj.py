# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models


# step1:准备数据:下载，加载，转换
# transform = transforms.Compose([ transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
normalize = transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))

Train_transform = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,])
Labels_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.RandomSizedCrop(224),
    transforms.ToTensor(),
    normalize,])


train_data = torchvision.datasets.CIFAR10(root='./CIFAR10data', train=True,
                                        download=False, transform=Train_transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=100,
                                          shuffle=True, num_workers=2)

test_data = torchvision.datasets.CIFAR10(root='./CIFAR10data', train=False,
                                       download=False, transform=Labels_transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100,
                                         shuffle=False, num_workers=2)



classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog','frog','horse','ship', 'truck')
# step2:定义模型
# 层的结构全部定义好了，参数也训练好了，False定义好了层结构
resnet18 = models.resnet18(pretrained = False)
resnet18.fc = nn.Linear(512,10) 

# step3:定义loss和optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet18.parameters(),lr = 0.001,momentum = 0.9)
# resnet18.train()
# resnet18.eval()
# step4:train the resnet18
for epoch in range(1):
  running_loss = 0.0
  for i,(inputs, labels) in enumerate(train_loader):
        # inputs = Variable(inputs.cuda())
    labels = Variable(labels)
    output = resnet18(Variable(inputs))
    loss = criterion(output,labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    running_loss += loss.data[0]
    if i % 10 == 9:
      print('[%d,%5d]loss:%.3f'%(epoch+1,i+1,running_loss/10))
      running_loss = 0.0
print('Finished Training')