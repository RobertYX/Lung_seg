import os
import torch
import numpy as np
import augment3D
import nibabel as nib
import random
import torchvision.transforms.functional as F

from torchvision.transforms import *
from torch.utils.data import DataLoader, Dataset



class Luna16_DataSet(torch.utils.data.Dataset):
    def __init__(self, data_path, augmentation=True):

        self.img = [os.path.join(data_path, x)
                for x in os.listdir(data_path)
                if x.startswith("nodule")]

        self.lae = [os.path.join(data_path, x)
                for x in os.listdir(data_path)
                if x.startswith("mask")]

        self.augmentation = augmentation
        #         
        assert len(self.img) == len(self.lae)
        # linux 下必须有这一步, 否则顺序是乱的
        self.img.sort()
        self.lae.sort()

        if self.augmentation:
            self.augmentation_transform = augment3D.RandomChoice(
                transforms=[augment3D.RandomRotation(min_angle=-180, max_angle=180),
                            augment3D.StaticRotation(),
                            augment3D.GaussianNoise(mean=0, std=0.01),
                            augment3D.RandomShift(),
                            augment3D.RandomZoom(),
                            augment3D.RandomFlip()], p=0.5)
    def __getitem__(self, idx):
        # 在被迭代时, 取第idx个
        img_path = self.img[idx]
        lae_path = self.lae[idx]

        # image = nib.load(img_path)
        # image = image.get_fdata()
        # label = nib.load(lae_path)
        # label = label.get_fdata()
        image = np.load(img_path)
        label = np.load(lae_path)

        if self.augmentation:
            # 做增强变换的代码要求image是CDHW格式, label是DHW格式
            # 输出格式保持不变         
            image, label = self.augmentation_transform(image[np.newaxis], label)  # CDHW, DHW
            label = label[np.newaxis]  # CDHW
        else:
            image = image[np.newaxis]  # CDHW
            label = label[np.newaxis]  # CDHW

        # 使内存空间连续, 尽量减少溢出可能
        image = np.ascontiguousarray(image)
        label = np.ascontiguousarray(label)
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
        # 确保label的值是固定的
        # label[label < 1.0] = 0
        # label[label >= 1.0] = 1

        return image, label

    def __len__(self):
        return len(self.img)



if __name__ == '__main__':
    # 设置一个自动批发数据, 使其输出数据, 借此查看数据传输是否正常
    
    Ds = Luna16_DataSet(data_path='/home/chenhao/Documents/sdb1_8T/DataSet/luna16/fix_data/valid', augmentation=True)
    loader = DataLoader(Ds, batch_size=1,
                      shuffle=True, num_workers=4, pin_memory=False)
    t = [] # 有结节的 1081
    f = [] # 无结节的 258               
    for i, data in enumerate(loader):
        img, lae = data

        print(img.shape, lae.shape)
        if torch.max(lae) == torch.tensor([1.]):
            t.append(1)
        else:
            f.append(1)

    print(np.sum(t))
    print(np.sum(f))