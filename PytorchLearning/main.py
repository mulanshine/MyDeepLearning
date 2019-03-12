# -*- coding: utf-8 -*-
import argparse  # 命令行解释器相关程序，命令行解释器
import os        # 操作系统文件相关
import shutil    # 文件高级操作
import time      # 调用时间模块

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn        # gpu 使用
import torch.distributed as dist            # 分布式（pytorch 0.2)
import torch.optim                          # 优化器
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# name中若为小写且不以‘——’开头，则对其进行升序排列
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))                
    # callable功能为判断返回对象是否可调用（即某种功能）。

# 创建argparse.ArgumentParser对象
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# 添加命令行元素
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')

# 定义参数
best_prec1 = 0

# 定义主函数main()
def main():
    global args, best_prec1
    # 使用函数parse_args()进行参数解析，输入默认是sys.argv[1:]，
    # 返回值是一个包含命令参数的Namespace，所有参数以属性的形式存在，比如args.myoption。
    args = parser.parse_args()

########## 使用多播地址进行初始化
    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

##### step1: create model and set GPU 
    # 导入pretrained model 或者创建model
    if args.pretrained:
        # format 格式化表达字符串，上述默认arch为resnet18
        print("=> using pre-trained model '{}'".format(args.arch))      
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
    # 分布式运行，可实现在多块GPU上运行
    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            # 批处理，多GPU默认用dataparallel使用在多块gpu上
            model.features = torch.nn.DataParallel(model.features)           
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        # Wrap model in DistributedDataParallel (CUDA only for the moment)
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)


##### step2: define loss function (criterion) and optimizer
    # 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss().cuda()                            
    # optimizer 使用 SGD + momentum
    # 动量，默认设置为0.9
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                # 权值衰减，默认为1e-4                 
                                weight_decay=args.weight_decay)         


   # 恢复模型（详见模型存取与恢复）
####step3：optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):                                 # 判断返回的是不是文件
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)                        # load 一个save的对象
            args.start_epoch = checkpoint['epoch']                      # default = 90
            best_prec1 = checkpoint['best_prec1']                       # best_prec1 = 0
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])          # load_state_dict:恢复模型
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

##### step4: Data loading code base of dataset(have downloaded) and normalize
    # 从 train、val文件中导入数据
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    # 数据预处理：normalize: - mean / std
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],       
                                     std=[0.229, 0.224, 0.225])
 
    # ImageFolder 一个通用的数据加载器
    train_dataset = datasets.ImageFolder(
        traindir,
        # 对数据进行预处理
        transforms.Compose([                      # 将几个transforms 组合在一起
            transforms.RandomSizedCrop(224),      # 随机切再resize成给定的size大小
            transforms.RandomHorizontalFlip(),    # 概率为0.5，随机水平翻转。
            transforms.ToTensor(),                # 把一个取值范围是[0,255]或者shape为(H,W,C)的numpy.ndarray，
                                                  # 转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
            normalize,
        ]))

#######
    if args.distributed:
        # Use a DistributedSampler to restrict each process to a distinct subset of the dataset.
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
######

    # train 数据下载及预处理
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([ 
            # 重新改变大小为`size`，若：height>width`,则：(size*height/width, size)
            transforms.Scale(256),
            # 将给定的数据进行中心切割，得到给定的size。
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)         # default workers = 4

##### step5: 验证函数
    if args.evaluate:
        validate(val_loader, model, criterion)             # 自定义的validate函数，见下
        return

##### step6:开始训练模型
    for epoch in range(args.start_epoch, args.epochs):
        # Use .set_epoch() method to reshuffle the dataset partition at every iteration
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)      # adjust_learning_rate 自定义的函数，见下

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


# 定义相关函数
# def train 函数
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        # criterion 为定义过的损失函数
        loss = criterion(output, target_var)        

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # 每十步输出一次
        if i % args.print_freq == 0:     # default=10
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        # 这是一种用来包裹张量并记录应用的操作
        """
        Attributes:
        data: 任意类型的封装好的张量。
        grad: 保存与data类型和位置相匹配的梯度，此属性难以分配并且不能重新分配。
        requires_grad: 标记变量是否已经由一个需要调用到此变量的子图创建的bool值。只能在叶子变量上进行修改。
        volatile: 标记变量是否能在推理模式下应用（如不保存历史记录）的bool值。只能在叶变量上更改。
        is_leaf: 标记变量是否是图叶子(如由用户创建的变量)的bool值.
        grad_fn: Gradient function graph trace.

        Parameters:
        data (any tensor class): 要包装的张量.
        requires_grad (bool): bool型的标记值. **Keyword only.**
        volatile (bool): bool型的标记值. **Keyword only.**
        """
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg

# 保存当前节点
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

# 计算并存储参数当前值或平均值
class AverageMeter(object):
    # Computes and stores the average and current value
    """
       batch_time = AverageMeter()
       即 self = batch_time
       则 batch_time 具有__init__，reset，update三个属性，
       直接使用batch_time.update()调用
       功能为：batch_time.update(time.time() - end)
               仅一个参数，则直接保存参数值
        对应定义：def update(self, val, n=1)
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        这些有两个参数则求参数val的均值，保存在avg中##不确定##

    """
    def __init__(self):
        self.reset()       # __init__():reset parameters

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 更新 learning_rate ：每30步，学习率降至前的10分之1
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))            # args.lr = 0.1 , 即每30步，lr = lr /10
    for param_group in optimizer.param_groups:       # 将更新的lr 送入优化器 optimizer 中，进行下一次优化
        param_group['lr'] = lr

# 计算准确度
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    """
    maxk = max(topk)
    # size函数：总元素的个数
    batch_size = target.size(0)

    # topk函数选取output前k大个数
    _, pred = output.topk(maxk, 1, True, True)
    ##########不了解t()
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
