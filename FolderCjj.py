# -*- coding: utf-8 -*-
import torch.utils.data as data
#PIL: Python Image Library缩写，图像处理模块
#     Image,ImageFont,ImageDraw,ImageFilter
from PIL import Image    
import os
import os.path
import torch.utils.data
import torchvision.transforms as transforms

# 图片扩展（图片格式）
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',]
def is_image_file(filename):
    # 注意学习any 的使用
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# 结果:classes:['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# classes_to_idx:{'1': 1, '0': 0, '3': 3, '2': 2, '5': 5, '4': 4, '7': 7, '6': 6, '9': 9, '8': 8}
def find_classes(dir):
    '''
    function:找到dir的文件和索引
    '''
    # os.listdir：以列表的形式显示当前目录下的所有文件名和目录名，但不会区分文件和目录。
    # os.path.isdir：判定对象是否是目录，是则返回True，否则返回False
    # os.path.join：连接目录和文件名
    list = os.listdir(dir)
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    # sort:排序
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

    # if os.path.isdir(runfile):
    #   listfile = os.listdir(runfile)
    #for i,dirname in enumerate(listfile.sort()):
    #   imgdir = os.path.join(runfile,dirname)
            #dir = 
            #classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
            # sort:排序
            #classes.sort()
            #class_to_idx = {classes[i]: i for i in range(len(classes))}
            #return classes, class_to_idx
            #else: 

# 如果文件是图片文件，则保留它的路径，和索引至images(path,class_to_idx)
def make_dataset(dir, class_to_idx):
    images = []
    # os.path.expanduser(path)：把path中包含的"~"和"~user"转换成用户目录
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue        
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images
# os.walk:遍历目录下所有内容，产生三元组
        # (dirpath, dirnames, filenames)【文件夹路径, 文件夹名字, 文件名】


# 打开路径下的图片，并转化为RGB模式
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # with as : 安全方面，可替换：try,finally
    # 'r':以读方式打开文件，可读取文件信息
    # 'b':以二进制模式打开文件，而不是文本
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            # convert：，用于图像不同模式图像之间的转换，这里转换为‘RGB’
            return img.convert('RGB')

def accimage_loader(path):
    # accimge:高性能图像加载和增强程序模拟的程序。
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    # get_image_backend:获取加载图像的包的名称
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class ImageFold(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    # 初始化，继承参数
    def __init__(self,root,train = False,transform = None, loader = default_loader): #targrt_transform = None
        if train:
            root = os.path.join(root,'train')
        else: 
            root = os.path.join(root,'test')
        #step1:find the path and image of images in the root
        # 找到root的文件和索引
        classes, class_to_idx = find_classes(root)
        # 保存路径下图片文件路径和索引至imgs
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        # self.target_transform = target_transform
        self.loader = loader

        #step2:save the path and images to imgs
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    root = '/home/jjchu/CIFAR10/'
# 数据预处理：normalize: - mean / std
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],       
                                       std=[0.229, 0.224, 0.225])
     
        # ImageFolder 一个通用的数据加载器
    train_dataset = ImageFold(
        root,
        train = True,
        # 对数据进行预处理
        transform = transforms.Compose([          # 将几个transforms 组合在一起
            transforms.RandomSizedCrop(224),      # 随机切再resize成给定的size大小
            transforms.RandomHorizontalFlip(),    # 概率为0.5，随机水平翻转。
            transforms.ToTensor(),                # 把一个取值范围是[0,255]或者shape为(H,W,C)的numpy.ndarray，
                                                  # 转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
            normalize,
        ]))

    print train_dataset

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=100, shuffle=True,
            num_workers=4)

    for i, (input, target) in enumerate(train_loader):
        print intput,target
        if i == 2:
            break