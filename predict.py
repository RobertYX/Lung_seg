import os
import numpy as np
import random
import SimpleITK as sitk
import pandas as pd
import cv2 as cv
import time

from tqdm import tqdm
from utils import *
from luna16 import Data_luna16, Preprocessor
from create_model import model

import config as cfg
join = os.path.join
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_path = []
# 读取结节csv文件， 以index_col作为索引
nodule_data = pd.read_csv(cfg.LaeDataPath, index_col='seriesuid') 

# 收集测试数据集上的以mhd后缀结尾的文件路径
for i in range(9, 10):
    input_addr = cfg.RawDataPath+'/subset{}'.format(i)
    img_pa = [os.path.join(input_addr, x)
            for x in os.listdir(input_addr)
            if x.endswith(".mhd")]
    image_path.extend(img_pa)
# linux系统要整理一下顺序
image_path.sort()
print(image_path)

#-------------------------必填注意--------------------------#
model_idx = 'eval' # 模型序列号
post = False # 是否后处理

model.load_state_dict(
    torch.load(join(cfg.ModelSavepath, 'model_best_{}.pth'.format(model_idx)), map_location=device))
    # torch.load(join(cfg.ModelSavepath, 'model_best_{}.pth'.format(model_idx)), map_location=device))
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
        exam_worker = Data_luna16(image_addr=image_path[idx], nodule_data=nodule_data)
        log_pred.info("I am trying to process: {}".format( exam_worker.id.split('.')[-1]))
        # 完成加载过程
        exam_worker.load()
        # 获取到图像数据和结节信息
        image_data = exam_worker.get_image()
        nodule_info = exam_worker.get_nodule()
        log_pred.info("nodule exist: {}".format(exam_worker.get_flag()))
        # 获取和图像数据相同格式的标签
        label_data = np.zeros_like(image_data)

        # 预处理
        image_data = normlize(image_data, MIN_BOUND=-1000.,MAX_BOUND=400.)

        # 多结节切割处理
        for ndx in range(len(nodule_info[:,0])):
            xyzd = nodule_info[ndx,:] # 取出当前结节的xyz和d
            xyz = xyzd[:3]; d = xyzd[3]*1.5 # 半径*1.5
            # 坐标转换
            zyx = worldToVoxelCoord(xyz, origin=exam_worker.get_origin(), spacing=exam_worker.get_spacing())[::-1] # 将xyz坐标从相对转为绝对
            zyxd = np.append(zyx, d)
            log_pred.info("nodule_{} info: {}".format(ndx, zyxd))
            nodule_info[ndx,:]=zyxd

            # 处理器1
            P01 = Preprocessor(image=image_data)
            # 裁切
            P01.crop(secZ=[zyxd[0]-d, zyxd[0]+d], secY=[zyxd[1]-d, zyxd[1]+d], secX=[zyxd[2]-d, zyxd[2]+d])
            # 重塑
            P01.resize(shape=cfg.xyz_down_scale)
            # 输出
            inp = P01.get_image() # DHW numpy
            inp = torch.from_numpy(inp) # tensor
            inp = inp.unsqueeze(0).unsqueeze(0).type(torch.FloatTensor).to(device) # BCDHW

            # 预测
            out = model(inp)
            out = torch_numpy(out) # 转成numpy BDHW

            out = out.squeeze(0) # DHW
            try:
                out = measureimg(out, t_num=1) # 保留最大的块
            except:
                out = out

            # 处理器2
            P02 = Preprocessor(image=out)
            # 重塑
            P02.resize(shape=P01.crop_shape, order=0) 
            # 返还
            label_data = replace_block(ori_block=label_data, input_block=P02.get_image(), secZ=[zyxd[0]-d, zyxd[0]+d], secY=[zyxd[1]-d, zyxd[1]+d], secX=[zyxd[2]-d, zyxd[2]+d])
        
        # 保存为nii.gz格式
        # image_itk = sitk.GetImageFromArray(image_data)
        label_itk = sitk.GetImageFromArray(label_data)
        label_itk = sitk.BinaryMorphologicalClosing(label_itk, kernelRadius=(3, 3, 3))
        # image_itk.CopyInformation(exam_worker.get_itk())
        label_itk.CopyInformation(exam_worker.get_itk())
        # sitk.WriteImage(image_itk, join(cfg.FileSavepath, exam_worker.id + "_image.nii.gz"))
        # sitk.WriteImage(label_itk, join(cfg.FileSavepath, exam_worker.id + "_label.nii.gz"))
        sitk.WriteImage(label_itk, join(cfg.FileSavepath, str(flag) + "_label.nii.gz"))

    print(flag)
    end_time = time.time()
    log_pred.info('Waste Time in Prediction: {}s/case'.format((end_time-start_time)/(flag+1)))