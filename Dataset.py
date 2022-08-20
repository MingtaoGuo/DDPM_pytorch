import torch 
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from PIL import Image 
import numpy as np 
import os 
import scipy.io as sio 

class Dataset_celeba(TensorDataset):
    def __init__(self, path):
        self.path = path
        self.datasets = os.listdir(path)
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        img = Image.open(self.path + self.datasets[item]).resize([160, 192])
        return self.transforms(img)

    def __len__(self):
        return len(self.datasets)

class Dataset_cifar(TensorDataset):
    def __init__(self, path):
         self.data = np.concatenate([sio.loadmat(path + "/data_batch_1.mat")['data'], 
                                     sio.loadmat(path + "/data_batch_2.mat")['data'],
                                     sio.loadmat(path + "/data_batch_3.mat")['data'],
                                     sio.loadmat(path + "/data_batch_4.mat")['data'],
                                     sio.loadmat(path + "/data_batch_5.mat")['data']], axis=0)
         self.data = np.reshape(self.data, [-1, 3, 32, 32])

    def __getitem__(self, item):
        
        return np.float32(self.data[item] / 127.5 - 1.0)

    def __len__(self):
        return self.data.shape[0]