import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import cv2 as cv
import random
import pylidc as pl
import matplotlib.pyplot as plt

from utils import *
from scipy.ndimage import zoom

import config as cfg


# 数据类-luna16
# 要求：
# 1.可以通过实例读取对应图像数据【做好优化，避免创建时就读取】
# 2.可以获取图像相关信息，如对比度，尺寸，spacing等等。
# 3.可以根据唯一ID获取对应结节坐标，良恶性信息【可扩展】。

# 当你用别的数据集时，需要修改
class Data_luna16():
    def __init__(self, image_addr:str, nodule_data:pd.DataFrame):
        # 路径
        self.image_addr = image_addr
        # 唯一ID
        self.id = image_addr.split('/')[-1].replace(".mhd","")
        # 图像数据
        self.itk_data = None
        self.image_data = None
        self.image_info = None
        # 结节数据
        self.nodule_data = nodule_data
        self.nodule_info = None
        self.nodule_flag = True
        # 原点坐标
        self.origin_coord = None
        self.spacing = None
    
    def load(self):
        # 加载三维数据，会占用内存
        itk_data = sitk.ReadImage(self.image_addr) # 读取图像数据 要求后缀是.mhd，读取时自动从同名的.raw文件读取
        self.image_data = sitk.GetArrayFromImage(itk_data) # itk2np 
  
        self.origin_coord = itk_data.GetOrigin()
        self.spacing = itk_data.GetSpacing()
        self.shape = self.image_data.shape
        self.itk_data = itk_data
        try:
            self.nodule_info = self.nodule_data.loc[self.id] # 索引结节坐标数据
        except:
            self.nodule_info = None
            self.nodule_flag = False # 实际上没有结节

    def get_nodule(self):
        # 获取结节信息（只有坐标+半径）
        if self.nodule_info is not None:
            nodule_coord = self.nodule_info.to_numpy()
            if nodule_coord.ndim == 1:
                # 如果只有一维,人为加一个维度方便for循环
                nodule_coord = np.expand_dims(nodule_coord, axis=0)
        else:
            # 不存在结节的图像人为捏一个结节
            factor = random.choice([0,1])
            if factor:
                a = VoxelToWorldCoord(voxelCoord=(128., 256., self.shape[0]/2), origin=self.get_origin(), spacing=self.get_spacing())
            else:
                a = VoxelToWorldCoord(voxelCoord=(384., 256., self.shape[0]/2), origin=self.get_origin(), spacing=self.get_spacing())
            a = np.append(a, 5.) # 添加diameter
            # 如果只有一维,人为加一个维度方便for循环
            nodule_coord = np.expand_dims(a, axis=0)
        return nodule_coord

    def get_image(self):
        # 获取图像信息
        return self.image_data
    def get_itk(self):
        # 获取itk信息
        return self.itk_data

    def get_origin(self):
        # 获取mhd的原点坐标
        return np.array(list(self.origin_coord))

    def get_spacing(self):
        # 获取mhd的像素间距spacing <==> shape
        return np.array(list(self.spacing))

    def get_flag(self):
        # 获取是否存在结节的boolean值
        return self.nodule_flag
# 可以归一化，resize，裁剪，并返回处理后数据

class Preprocessor():
    def __init__(self, image:np.ndarray):
        self.image = image
        self.crop_shape = None

    def get_image(self):
        return self.image

    def crop(self, secZ:list=[0.0, 1.0], secY:list=[0.0, 1.0], secX:list=[0.0, 1.0]):
        secZ = list(map(int, secZ))
        secY = list(map(int, secY))
        secX = list(map(int, secX))

        secZ = checkBoundry(self.image, secZ, 0)
        secY = checkBoundry(self.image, secY, 1)
        secX = checkBoundry(self.image, secX, 2)

        self.image = self.image[secZ[0]:secZ[1], secY[0]:secY[1], secX[0]:secX[1]]

        self.crop_shape = self.image.shape

    def resize(self, shape:tuple=(int, int, int), order=3):
        # resize 应该是基于现在图像的形状做resize
        new_shape = (shape[0]/self.image.shape[0], shape[1]/self.image.shape[1], shape[2]/self.image.shape[2])
        self.image = zoom(self.image, new_shape, order=order) # 三线性采样

    def addCircle(self, secZ:list=[0.0, 1.0], secY:list=[0.0, 1.0], secX:list=[0.0, 1.0]):
        secZ = list(map(int, secZ))
        secY = list(map(int, secY))
        secX = list(map(int, secX))
        self.image[secZ[0]:secZ[1], secY[0]:secY[1], secX[0]:secX[1]] = 1





if __name__ == '__main__':
    tiaddr = '/home/chenhao/Documents/sdb1_8T/DataSet/luna16/subset8/1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860.mhd'
    tnaddr = '/home/chenhao/Documents/sdb1_8T/DataSet/luna16/annotations.csv'
    nodule_data = pd.read_csv(tnaddr, index_col='seriesuid') # 读取结节csv文件， 以index_col作为索引
    test01 = Data_luna16(image_addr=tiaddr, nodule_data=nodule_data)
    print("id: "+test01.id)
    test01.load()

    image_data = test01.get_image()
    nodule_info = test01.get_nodule()

    print("image_data: ", image_data.shape)
    print("nodule_info: d&xyz", nodule_info)
    # 加载预处理
    image_data = normlize(image_data, MIN_BOUND=-1000.,MAX_BOUND=400.)

    # 多结节
    for ndx in range(len(nodule_info[:,0])):
        xyzd = nodule_info[ndx,:] # 取出当前结节的xyz和d
        xyz = xyzd[:3]; d = xyzd[3]*1.5 # 半径*1.5
        zyx = worldToVoxelCoord(xyz, origin=test01.get_origin(), spacing=test01.get_spacing())[::-1] # 将xyz坐标从相对转为绝对
        zyxd = np.append(zyx, d)
        nodule_info[ndx,:]=zyxd
        P01 = Preprocessor(image=image_data)
        P01.crop(secZ=[zyxd[0]-d, zyxd[0]+d], secY=[zyxd[1]-d, zyxd[1]+d], secX=[zyxd[2]-d, zyxd[2]+d])
        P01.resize(shape=(96, 96, 96))
        image_A = P01.get_image()

        _, rst=cv.threshold(image_A,0.25,0.85,cv.THRESH_BINARY)

        fig1 = np.expand_dims(rst[48,...],axis=-1)
        fig2 = np.expand_dims(image_A[48,...],axis=-1)
        # print(fig.shape)
        cv.imshow("test1",fig1)
        cv.imshow("test2",fig2)
        cv.waitKey(0)
        # cv.destroyAllWindows()




