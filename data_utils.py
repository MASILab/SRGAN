from os import listdir
from os.path import join

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
from medicaltorch import transforms as mt_transforms

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.nii.gz', '.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        mt_transforms.CenterCrop2D(crop_size),
        mt_transforms.ToTensor()
    ])#Compose([
        #RandomCrop(crop_size),
        #ToTensor(),
    #])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        #ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])

def pad(img):
    tmp = np.zeros((96, 96, 96, img.shape[3]))

    diff = int((96-img.shape[0])/2)
    lx = max(diff,0)
    lX = min(img.shape[0]+diff,96)

    diff = (img.shape[0]-96) / 2
    rx = max(int(np.floor(diff)),0)
    rX = min(img.shape[0]-int(np.ceil(diff)),img.shape[0])

    diff = int((96 - img.shape[1]) / 2)
    ly = max(diff, 0)
    lY = min(img.shape[1] + diff, 96)

    diff = (img.shape[1] - 96) / 2
    ry = max(int(np.floor(diff)), 0)
    rY = min(img.shape[1] - int(np.ceil(diff)), img.shape[1])

    diff = int((96 - img.shape[2]) / 2)
    lz = max(diff, 0)
    lZ = min(img.shape[2] + diff, 96)

    diff = (img.shape[2] - 96) / 2
    rz = max(int(np.floor(diff)), 0)
    rZ = min(img.shape[2] - int(np.ceil(diff)), img.shape[2])

    tmp[lx:lX,ly:lY,lz:lZ] = img[rx:rX,ry:rY,rz:rZ]

    return tmp, [lx, lX, ly, lY, lz, lZ, rx, rX, ry, rY, rz, rZ]

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.data = pd.read_csv(dataset_dir)
        self.lr_column = self.data['ISOTROPIC']
        print(self.lr_column)
        self.hr_column = self.data['TARGET']
        print(self.hr_column)
        self.is_transform = True
        self.i = 0

    def transform(self, image):
        '''
        This function transforms the 3D image of np.ndarray (z,x,y) to a torch.ShortTensor (B,z,x,y).
        
        '''

        image_torch = torch.ShortTensor(image)
        return image_torch

    def __getitem__(self, i):
        print(i)
        if self.i % 1344 == 0:
            index = np.random.randint(0, len(self.data))
            # low resolution data
            nii_lr = nib.load(self.lr_column[index])
            self.lr_image = nii_lr.get_fdata()
            # pad the surrounding 
            self.lr_image_pad, self.pad_idx = pad(self.lr_image)
            
            # hr resolution data
            nii_hr = nib.load(self.hr_column[index])
            self.hr_image = nii_hr.get_fdata()
            self.hr_image_pad, self.pad_idx = pad(self.hr_image)
        self.i += 1

        vol_idx = np.random.randint(0, self.lr_image.shape[3])
        lr_vol = torch.from_numpy(self.lr_image_pad[:,:,:,vol_idx]).unsqueeze(0).float()
        # normalize the data
        lr_max = np.percentile(lr_vol, 99.99)
        lr_vol = lr_vol / lr_max

        hr_vol = torch.from_numpy(self.hr_image_pad[:,:,:,vol_idx]).unsqueeze(0).float()
        hr_max = np.percentile(hr_vol, 99.99)
        hr_vol = hr_vol / hr_max
        print(lr_vol.size(), hr_vol.size())
        return lr_vol, hr_vol

    def __len__(self):
        return len(self.data) *1344


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        # self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        self.data = pd.read_csv(dataset_dir)
        self.lr_column = self.data['ISOTROPIC']
        print(self.lr_column)
        self.hr_column = self.data['TARGET']
        print(self.hr_column)
        self.is_transform = True
        self.i = 0

    def __getitem__(self, i):
        if self.i % 1344 == 0:
            print('LENGTH', len(self.data))
            index = np.random.randint(0, len(self.data))
            # low resolution data
            nii_lr = nib.load(self.lr_column[index])
            self.image_path = self.lr_column[index]
            self.lr_image = nii_lr.get_fdata()
            # pad the surrounding
            self.lr_image_pad, self.pad_idx = pad(self.lr_image)

            # hr resolution data
            nii_hr = nib.load(self.hr_column[index])
            self.hr_image = nii_hr.get_fdata()
            self.orig_shape = nii_hr.shape
            self.hr_image_pad, self.pad_idx = pad(self.hr_image)
        self.i += 1
        
        
        vol_idx = np.random.randint(0, self.lr_image.shape[3])
        lr_vol = torch.from_numpy(self.lr_image_pad[:, :, :, vol_idx]).unsqueeze(0).float()
        # normalize the data
        lr_max = np.percentile(lr_vol, 99.99)
        lr_vol = lr_vol / lr_max

        hr_vol = torch.from_numpy(self.hr_image_pad[:, :, :, vol_idx]).unsqueeze(0).float()
        hr_max = np.percentile(hr_vol, 99.99)
        hr_vol = hr_vol / hr_max
        print(lr_vol.size(), hr_vol.size())
        return self.image_path, lr_vol, hr_vol, self.pad_idx, self.orig_shape

    def __len__(self):
        return len(self.data) * 1344


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.data = pd.read_csv(dataset_dir)
        self.lr_column = self.data['ISOTROPIC']
        print(self.lr_column)
        # lr reso
        nii_lr = nib.load(self.lr_column[0])
        self.lr_image = nii_lr.get_fdata()
        # pad the surrounding
        self.lr_image_pad, self.pad_idx = pad(self.lr_image)
        
        self.hr_column = self.data['TARGET']
        print(self.hr_column)
        self.is_transform = True
        self.i = 0

    def __getitem__(self, i):
        vol_idx = i
        # hr resolution data
        nii_hr = nib.load(self.hr_column[0])
        self.orig_shape = nii_hr.shape
        self.hr_image = nii_hr.get_fdata()
        self.hr_image_pad, self.pad_idx = pad(self.hr_image)

        lr_vol = torch.from_numpy(self.lr_image_pad[:, :, :, vol_idx]).unsqueeze(0).float()
        # normalize the data
        lr_max = np.percentile(lr_vol, 99.99)
        lr_vol = lr_vol / lr_max

        hr_vol = torch.from_numpy(self.hr_image_pad[:, :, :, vol_idx]).unsqueeze(0).float()
        hr_max = np.percentile(hr_vol, 99.99)
        hr_vol = hr_vol / hr_max
        print(lr_vol.size(), hr_vol.size())

        return self.lr_column[0], lr_vol, hr_vol, hr_vol, self.pad_idx, self.orig_shape

    def __len__(self):
        return self.lr_image_pad.shape[3]
