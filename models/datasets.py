import os, glob
import torch, sys
from torch.utils.data import Dataset
from .data_utils import pkload
import pickle
import SimpleITK as sitk

import matplotlib.pyplot as plt
import numpy as np


class JHUBrainDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, y = pkload(path)
        #print(x.shape)
        #print(x.shape)
        #print(np.unique(y))
        # print(x.shape, y.shape)#(240, 240, 155) (240, 240, 155)
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        x,y = self.transforms([x, y])
        #y = self.one_hot(y, 2)
        #print(y.shape)
        #sys.exit(0)
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(x[0, :, :, 8], cmap='gray')
        # plt.subplot(1, 2, 2)
        # plt.imshow(y[0, :, :, 8], cmap='gray')
        # plt.show()
        # sys.exit(0)
        # y = np.squeeze(y, axis=0)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)


class JHUBrainInferDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, y, x_seg, y_seg = pkload(path)
        #print(x.shape)
        #print(x.shape)
        #print(np.unique(y))
        # print(x.shape, y.shape)#(240, 240, 155) (240, 240, 155)
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        #y = self.one_hot(y, 2)
        #print(y.shape)
        #sys.exit(0)
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        #plt.figure()
        #plt.subplot(1, 2, 1)
        #plt.imshow(x[0, :, :, 8], cmap='gray')
        #plt.subplot(1, 2, 2)
        #plt.imshow(y[0, :, :, 8], cmap='gray')
        #plt.show()
        #sys.exit(0)
        #y = np.squeeze(y, axis=0)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)


class PairedImageDataset(Dataset):
    def __init__(self, pkl_dir):
        self.pkl_files = [os.path.join(pkl_dir, f) for f in os.listdir(pkl_dir) if f.endswith('.pkl')]

    def __len__(self):
        return len(self.pkl_files)

    def __getitem__(self, idx):
        with open(self.pkl_files[idx], 'rb') as f:
            data = pickle.load(f)

        moving_img= torch.tensor(data[0], dtype=torch.float32)
        fixed_img = torch.tensor(data[1], dtype=torch.float32)
        moving_label = torch.tensor(data[2], dtype=torch.float32)
        fixed_label= torch.tensor(data[3], dtype=torch.float32)

        fixed_img = torch.unsqueeze(fixed_img, 0)
        moving_img = torch.unsqueeze(moving_img, 0)
        fixed_label = torch.unsqueeze(fixed_label, 0)
        moving_label = torch.unsqueeze(moving_label, 0)



        return moving_img,fixed_img,moving_label,fixed_label


class PairedImageDatasetTest(Dataset):
    def __init__(self, pkl_dir):
        self.pkl_files = [os.path.join(pkl_dir, f) for f in os.listdir(pkl_dir) if f.endswith('.pkl')]

    def __len__(self):
        return len(self.pkl_files)

    def __getitem__(self, idx):
        with open(self.pkl_files[idx], 'rb') as f:
            data = pickle.load(f)
        fixed_img = torch.tensor(data['fixed'], dtype=torch.float32)
        moving_img = torch.tensor(data['moving'], dtype=torch.float32)
        fixed_label = torch.tensor(data['fixed_label'], dtype=torch.float32)
        moving_label = torch.tensor(data['moving_label'], dtype=torch.float32)

        return fixed_img, moving_img, fixed_label, moving_label


class Dataset2(Dataset):
    def __init__(self, file1, file2):
        # 初始化
        self.file1 = file1
        self.file2 = file2

    def __len__(self):
        # 返回数据集的大小
        return len(self.file1)

    def __getitem__(self, index):
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意
        ##加
        img_arr1 = sitk.GetArrayFromImage(sitk.ReadImage(self.file1[index]))[np.newaxis, ...]
        img_arr2 = sitk.GetArrayFromImage(sitk.ReadImage(self.file2[index]))[np.newaxis, ...]

        # 返回值自动转换为torch的tensor类型
        return img_arr1, img_arr2, self.file1[index], self.file2[index]



class Dataset3(Dataset):
    def __init__(self, file1, file2,xseg,yseg):
        # 初始化
        self.file1 = file1
        self.file2 = file2
        self.xseg = xseg
        self.yseg = yseg

    def __len__(self):
        # 返回数据集的大小
        return len(self.file1)

    def __getitem__(self, index):
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意
        ##加
        img_arr1 = sitk.GetArrayFromImage(sitk.ReadImage(self.file1[index]))[np.newaxis, ...]
        img_arr2 = sitk.GetArrayFromImage(sitk.ReadImage(self.file2[index]))[np.newaxis, ...]
        x_seg = sitk.GetArrayFromImage(sitk.ReadImage(self.xseg[index]))[np.newaxis, ...]
        y_seg = sitk.GetArrayFromImage(sitk.ReadImage(self.yseg[index]))[np.newaxis, ...]
        # 返回值自动转换为torch的tensor类型
        return img_arr1, img_arr2,x_seg, y_seg
