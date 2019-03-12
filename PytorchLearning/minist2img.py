# -*- coding: utf-8 -*-
import numpy as np
import struct
import matplotlib.pyplot as plt
# 二进制读取数据
filename = 'train-images-idx3-ubyte'
binfile = open(filename, 'rb')
buf = binfile.read()
# '>IIII' 是说使用大端法读取4个unsinged int32
index = 0
magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index)
# '>784B'的意思就是用大端法读取784个unsigned byte
im = struct.unpack_from('>784B', buf, index)
index += struct.calcsize('>784B')
im = np.array(im)
im = im.reshape(28, 28)
fig = plt.figure()
plotwindow = fig.add_subplot(111)
plt.imshow(im, cmap='gray')
plt.show()