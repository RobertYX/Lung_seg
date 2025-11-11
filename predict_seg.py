import os
import numpy as np
import random
import SimpleITK as sitk
import pandas as pd
import cv2 as cv
import time
import math

from tqdm import tqdm
from utils import *
from luna16 import Data_luna16, Preprocessor
from create_model import model

import config as cfg
join = os.path.join
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_path = []
label_path = [] 

# 收集测试数据集上的以mhd后缀结尾的文件路径
for i in range(9, 10):
    # input_addr = cfg.RawDataPath+'/subset{}'.format(i)
    input_addr = cfg.RawDataPath + '/subset001'
    # input_addr1 = cfg.RawDataPath + '/subset92'
    img_pa = [os.path.join(input_addr, x)
            for x in os.listdir(input_addr)
            if x.endswith(".nii")]
    # lae_pa = [os.path.join(input_addr1, x)
    #     for x in os.listdir(input_addr1)
    #     if x.endswith("seg.nii.gz")]
    image_path.extend(img_pa)
    # label_path.extend(lae_pa)
# linux系统要整理一下顺序
image_path.sort()
# label_path.sort()
print(image_path)
# print(label_path)

# assert len(image_path)==len(label_path)
#-------------------------必填注意--------------------------#
model_idx = 'eval' # 模型序列号
post = False # 是否后处理

model.load_state_dict(
    torch.load(join(cfg.ModelSavepath, 'model_best_{}.pth'.format(model_idx)), map_location=device))
model = model.to(device)
model.eval()

# 监测
log_pred = logger(join(cfg.FileSavepath, 'LOG_model_{}'.format(model_idx)+'.log')) # 创建监察记录

if __name__ == '__main__':   
    start_time = time.time()
    flag = 0
    # 开始遍历预处理
    for idx in tqdm(range(len(image_path))):
        flag = flag + 1
        image_itk = sitk.ReadImage(image_path[idx])
        # label_itk = sitk.ReadImage(label_path[idx])
        log_pred.info("I am trying to process: {}".format(image_path[idx].split('/')[-1].split('_')[0].split('.')[-1]))

        # 获取到图像数据和结节信息
        image_data = sitk.GetArrayFromImage(image_itk)
        # label_data = sitk.GetArrayFromImage(label_itk).astype('int16')
        # label_data[label_data>=1]=1 # 避免超值
        # 预处理
        image_data = normlize(image_data, MIN_BOUND=-1000.,MAX_BOUND=400.)

        # 计算结节数量
        # label, region_nums = measure.label(label_data, background=0, return_num=True, connectivity=2)
        # region = measure.regionprops(label)
        df = pd.read_excel('nodule.xls', sheet_name=str(idx))
        # 提取x、y、z和直径列
        x_values = df['x']
        y_values = df['y']
        z_values = df['z']
        diameter_values = df['大小(cm)']
        object_count = len(df.index)
        # if region_nums==0:
        #     # 获取和图像数据相同格式的标签
        label_data = np.zeros_like(image_data)
        # 多结节切割处理
        # for ndx in range(region_nums):
        # bbox = (z_values[i]-5, y_values[i]-5, x_values[i]-5, z_values[i]+5, y_values[i]+5, x_values[i]+5)
        for ndx in range(object_count):
            # bbox = region[ndx]['bbox'] # 返回(z1,y1,x1,z2,y2,x2)
            az = math.ceil(diameter_values[ndx]*10/1.5 + 1)
            ay = math.ceil(diameter_values[ndx]*10/0.7 + 1)
            ax = math.ceil(diameter_values[ndx]*10/0.7 + 1)
            bbox = (z_values[ndx]-az, y_values[ndx]-ay, x_values[ndx]-ax, z_values[ndx]+az, y_values[ndx]+ay, x_values[ndx]+ax)
            print(bbox)
            # d = region[ndx]['feret_diameter_max'] # 返回最大直径d
            a = bbox[3]-bbox[0]; b = bbox[4]-bbox[1]; c = bbox[5]-bbox[2]
            d = np.max([a,b,c]) / 2
            # 处理器
            P_image = Preprocessor(image=image_data)
            # 裁切
            P_image.crop(secZ=[bbox[0]-d, bbox[3]+d], secY=[bbox[1]-d, bbox[4]+d], secX=[bbox[2]-d, bbox[5]+d])
            # 重塑
            P_image.resize(shape=cfg.xyz_down_scale)
            # 输出
            inp = P_image.get_image()
            inp = torch.from_numpy(inp) # tensor
            inp = inp.unsqueeze(0).unsqueeze(0).type(torch.FloatTensor).to(device) # BCDHW

            # 预测
            out = model(inp)
            out = torch_numpy(out) # 转成numpy BDHW
            out = out.squeeze(0) # DHW

            # 处理器2
            P_out = Preprocessor(image=out)
            # 重塑
            P_out.resize(shape=P_image.crop_shape, order=0) 
            # 返还
            label_data = replace_block(ori_block=label_data, input_block=P_out.get_image(), secZ=[bbox[0]-d, bbox[3]+d], secY=[bbox[1]-d, bbox[4]+d], secX=[bbox[2]-d, bbox[5]+d])
        
        # 保存为nii.gz格式
        # image_itk = sitk.GetImageFromArray(image_data)
        label_itk = sitk.GetImageFromArray(label_data)
        if post:
            label_itk = sitk.BinaryMorphologicalClosing(label_itk, kernelRadius=(3, 3, 3))
        # image_itk.CopyInformation(exam_worker.get_itk())
        label_itk.CopyInformation(image_itk)
        # sitk.WriteImage(image_itk, join(cfg.FileSavepath, exam_worker.id + "_image.nii.gz"))
        sitk.WriteImage(label_itk, join(cfg.FileSavepath, str(flag)) + "_label.nii.gz")
        # sitk.WriteImage(label_itk, join(cfg.FileSavepath, image_path[idx].split('/')[-1].split('_')[0]) + "_label.nii.gz")

    end_time = time.time()
    log_pred.info('Waste Time in Prediction: {}s/case'.format((end_time-start_time)/(flag+1)))