import os
import numpy as np
import random
import SimpleITK as sitk
import pandas as pd
import cv2 as cv

from tqdm import tqdm
from utils import *
from luna16 import Data_luna16, Preprocessor

import config as cfg
    
image_path = []
# 读取结节csv文件， 以index_col作为索引
nodule_data = pd.read_csv(cfg.LaeDataPath, index_col='seriesuid') 

if __name__ == '__main__':

    # 收集九个数据集上的以mhd后缀结尾的文件路径
    for i in range(0,9):
        input_addr = cfg.RawDataPath+'/subset{}'.format(i)
        img_pa = [os.path.join(input_addr, x)
                for x in os.listdir(input_addr)
                if x.endswith(".mhd")]
        image_path.extend(img_pa)
    # linux系统要整理一下顺序
    image_path.sort()

    # 开始遍历预处理
    for idx in tqdm(range(len(image_path))):
        exam_worker = Data_luna16(image_addr=image_path[idx], nodule_data=nodule_data)
        print("I am trying to process: ", exam_worker.id.split('.')[-1])
        # 完成加载过程
        exam_worker.load()
        # 获取到图像数据和结节信息
        image_data = exam_worker.get_image()
        nodule_info = exam_worker.get_nodule()

        # 预处理
        image_data = normlize(image_data, MIN_BOUND=-1000.,MAX_BOUND=400.)

        # 多结节切割处理
        for ndx in range(len(nodule_info[:,0])):
            xyzd = nodule_info[ndx,:] # 取出当前结节的xyz和d
            xyz = xyzd[:3]; d = xyzd[3]*1.5 # 半径*1.5
            # 坐标转换
            zyx = worldToVoxelCoord(xyz, origin=exam_worker.get_origin(), spacing=exam_worker.get_spacing())[::-1] # 将xyz坐标从相对转为绝对
            zyxd = np.append(zyx, d)
            nodule_info[ndx,:]=zyxd
            # 处理器
            P01 = Preprocessor(image=image_data)
            # 裁切
            P01.crop(secZ=[zyxd[0]-d, zyxd[0]+d], secY=[zyxd[1]-d, zyxd[1]+d], secX=[zyxd[2]-d, zyxd[2]+d])
            # 重塑
            P01.resize(shape=cfg.xyz_down_scale)
            # 生成对应的mask
            mask = generate_mask(list(cfg.xyz_down_scale), cfg.xyz_down_scale[0]//4, list(map(lambda x: x // 2 -1, cfg.xyz_down_scale))).astype(int)
            if not exam_worker.get_flag():
                # 如果结节实际上不存在
                mask[mask!=0]=0 # 全部置0

            # 输出
            image_save = P01.get_image()

            # 不存在路径? 创建
            if not os.path.exists(cfg.FixDataPath + 'train'):
                os.makedirs(cfg.FixDataPath+'train')
            if not os.path.exists(cfg.FixDataPath + 'valid'):
                os.makedirs(cfg.FixDataPath+'valid')            
            # 按一定概率分到训练集 和 验证集
            factor = random.choice([0,1,1,1,1,1])
            if factor:
                seperate_set = 'train'
            else:
                seperate_set = 'valid'
            # 保存为npy格式
            np.save(cfg.FixDataPath + seperate_set + '/nodule_{}_{}'.format(str(idx).rjust(3, '0'), ndx), image_save)
            np.save(cfg.FixDataPath + seperate_set + '/mask_{}_{}'.format(str(idx).rjust(3, '0'), ndx), mask)

