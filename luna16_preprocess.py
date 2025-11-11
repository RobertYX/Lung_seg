import os
import numpy as np
import random
import SimpleITK as sitk
import pandas as pd
import cv2 as cv
import nibabel as nib
from tqdm import tqdm
from utils import *
from skimage import measure
from luna16 import Data_luna16, Preprocessor

import config as cfg

image_path = []
label_path = []

if __name__ == '__main__':

    # # 收集九个数据集上的以mhd后缀结尾的文件路径
    # for i in range(0,9): #[0,9)
    #     input_addr = cfg.RawDataPath+'/subset{}'.format(i)
    #     img_pa = [os.path.join(input_addr, x)
    #             for x in os.listdir(input_addr)
    #             if x.endswith("img.nii.gz")]
    #     lae_pa = [os.path.join(input_addr, x)
    #             for x in os.listdir(input_addr)
    #             if x.endswith("seg.nii.gz")]
    #     image_path.extend(img_pa)
    #     label_path.extend(lae_pa)

    input_addr = r'F:\PY\lung_seg\DataSet\luna16\fix_data\train'
    img_pa = [os.path.join(input_addr, x)
              for x in os.listdir(input_addr)
              if x.startswith("nodule")]
    lae_pa = [os.path.join(input_addr, x)
              for x in os.listdir(input_addr)
              if x.startswith("mask")]
    image_path.extend(img_pa)
    label_path.extend(lae_pa)
    # linux系统要整理一下顺序
    image_path.sort()
    label_path.sort()
    print('path')
    print(image_path)
    print(label_path)

    assert len(image_path) == len(label_path)

    # 开始遍历预处理
    for idx in tqdm(range(len(image_path))):
        # 读取itk图像
        image_itk = sitk.ReadImage(image_path[idx])
        label_itk = sitk.ReadImage(label_path[idx])
        print("I am trying to process: ", image_path[idx].split('/')[-1].split('_')[0].split('.')[-1])  #
        # itk -> np.array
        image_data = sitk.GetArrayFromImage(image_itk)
        label_data = sitk.GetArrayFromImage(label_itk).astype('int16')
        label_data[label_data >= 1] = 1  # 避免超值
        # 预处理
        image_data = normlize(image_data, MIN_BOUND=-1000., MAX_BOUND=400.)

        # 计算结节数量
        label, region_nums = measure.label(label_data, background=0, return_num=True, connectivity=2)
        region = measure.regionprops(label)

        if region_nums == 0:
            # 没有结节就不割
            continue
        else:
            # 多结节切割处理
            for ndx in range(region_nums):

                bbox = region[ndx]['bbox']  # 返回(z1,y1,x1,z2,y2,x2)
                print(bbox)
                # d = region[ndx]['feret_diameter_max'] # 返回最大直径d
                a = bbox[3] - bbox[0];
                b = bbox[4] - bbox[1];
                c = bbox[5] - bbox[2]
                d = np.max([a, b, c]) / 2

                label_data[label_data >= 1] = 1  # 避免超值
                # 处理器
                P_image = Preprocessor(image=image_data)
                P_label = Preprocessor(image=label_data)
                # 裁切
                P_image.crop(secZ=[bbox[0] - d, bbox[3] + d], secY=[bbox[1] - d, bbox[4] + d],
                             secX=[bbox[2] - d, bbox[5] + d])
                P_label.crop(secZ=[bbox[0] - d, bbox[3] + d], secY=[bbox[1] - d, bbox[4] + d],
                             secX=[bbox[2] - d, bbox[5] + d])
                # 重塑
                P_image.resize(shape=cfg.xyz_down_scale)
                P_label.resize(shape=cfg.xyz_down_scale)
                # 输出
                image_save = P_image.get_image()
                label_save = P_label.get_image()
                label_save[label_save >= 1] = 1
                # # check
                print(image_save.shape)  # np.unique(image_save)
                print(label_save.shape, np.unique(label_save))

                # 不存在路径? 创建
                if not os.path.exists(cfg.FixDataPath + 'train3'):
                    os.makedirs(cfg.FixDataPath + 'train3')
                if not os.path.exists(cfg.FixDataPath + 'valid3'):
                    os.makedirs(cfg.FixDataPath + 'valid3')
                # 按一定概率分到训练集 和 验证集
                factor = random.choice([0, 1, 1, 1, 1, 1])
                if True:
                    seperate_set = 'train3'
                else:
                    seperate_set = 'valid1'
                # exit(0)
                # 保存为npy格式
                affine = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])
                np.save(cfg.FixDataPath + seperate_set + '/nodule_{}_{}'.format(str(idx).rjust(3, '0'), ndx),
                        image_save)
                image_save1 = nib.Nifti1Image(image_save, affine)
                label_save1 = nib.Nifti1Image(label_save, affine)
                nib.save(image_save1, cfg.FixDataPath + seperate_set + '/tenodule_{}_{}'.format(str(idx).rjust(3, '0'), ndx))
                nib.save(label_save1, cfg.FixDataPath + seperate_set + '/temask_{}_{}'.format(str(idx).rjust(3, '0'), ndx))
                np.save(cfg.FixDataPath + seperate_set + '/mask_{}_{}'.format(str(idx).rjust(3, '0'), ndx), label_save)
