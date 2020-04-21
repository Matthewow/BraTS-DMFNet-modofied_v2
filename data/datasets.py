import os
import torch
from torch.utils.data import Dataset
import pickle
from .rand import Uniform
from .transforms import Rot90, Flip, Identity, Compose
from .transforms import GaussianBlur, Noise, Normalize, RandSelect
from .transforms import RandCrop, CenterCrop, Pad,RandCrop3D,RandomRotion,RandomFlip,RandomIntensityChange
from .transforms import NumpyType
from .data_utils import pkload

import numpy as np


class BraTSDataset(Dataset):
    def __init__(self, list_file, root='', for_train=True,transforms=''):
        paths, names = [], []
        self.root="D:\CityU\FYP NN\doc\BraTS-DMFNet-master\pkl"
        #with open(list_file) as f:
        #    for line in f:
        #        line = line.strip()
        #        name = line.split('/')[-1]
        #        names.append(name)
        #        path = os.path.join(root, line , name + '_')
        #        paths.append(path)

        #self.names = names
        self.paths =os.listdir(self.root)

        self.transforms = eval(transforms or 'Identity()')

    def __getitem__(self, index):
        path = self.paths[index]
        path="D:\CityU\FYP NN\doc\BraTS-DMFNet-master\pkl\\"+path
        f=open(path,'rb+')
        x,y=pickle.load(f)


        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]

        # 官方：rint(x.shape, y.shape)#(1, 240, 240,155, 4) (1, 240, 240, 155)
        # 本数据集:(1,_,_,_,1) (1,_,_,_)

        #padx=np.zeros((1,512,512,155,4))
        #pady=np.zeros((1, 512, 512, 155))

        #padx[:,:,:,0:47,0:1]=x
        #pady[:, :, :, 0:47] = y

        #官方：(1,240,240,155,4),(1,240,240,155)

        x,y = self.transforms([x, y])
        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]

        y = np.ascontiguousarray(y)

        x, y = torch.from_numpy(x), torch.from_numpy(y)

        # print(x.shape, y.shape)  # (240, 240, 155, 4) (240, 240, 155)
        # 官网数据集: (1,4,128,128,128) (1,128,128,128)
        # 本数据集：(1,1,256,256,16) (1,256,256,16)

        return x, y

    def __len__(self):
        return len(self.paths)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]

