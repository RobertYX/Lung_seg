#################################################
#      CreateTime:20221026                   
#      UpdateTIme:20230315(The Last)         
#      Creator: FuZhou University iipa.fzu.edu.cn
#      Main Editor: Gunhild
#      "To be or not to be." --ShakeSpear
#################################################

import os
import numpy as np
import random
import SimpleITK as sitk
import pandas as pd
import cv2 as cv

from skimage import measure
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
            if x.endswith("seg.nii.gz")]
    image_path.extend(img_pa)

# 收集模型输出的结果的文件路径
label_path = [os.path.join(cfg.FileSavepath, x)
            for x in os.listdir(cfg.FileSavepath)
            if x.endswith("label.nii.gz")]

# linux系统要整理一下顺序
image_path.sort()
label_path.sort()

assert len(image_path)==len(label_path)

# 把每例的结节数量和模型输出数量做一个对比
case_name = []
boolean_flag = [] # 如果out和gt数量不一致，设为False
out_nodule_num = []
gt_nodule_num = []


if __name__ == '__main__':   
    FP = 0
    # 开始遍历
    for idx in tqdm(range(len(image_path))):
        num_nodule = 0 # 当前病例结节数量

        # 真实数据
        exam_worker = Data_luna16(image_addr=image_path[idx], nodule_data=nodule_data)
        exam_worker.load()

        # 模型输出数据
        label_itk = sitk.ReadImage(label_path[idx])
        label = sitk.GetArrayFromImage(label_itk)
        # 计算输出数据的连通域数量
        _, num = measure.label(label, background=0, return_num=True, connectivity=3)
        if len(label[label==1])<8:
            num = 0
        
        # 获取到gt的结节信息
        nodule_flag = exam_worker.get_flag()
        # 
        if not nodule_flag: # 如果本身不存在结节，那就直接置零
            num_nodule = 0
            if num_nodule != num:
                FP += 1 # 假阳
        else:
            nodule_info = exam_worker.get_nodule()
            num_nodule = len(nodule_info[:,0])
        # case name
        case_name.append(exam_worker.id)
        # 正确数量
        out_nodule_num.append(num)
        # 总数量
        gt_nodule_num.append(num_nodule)
        # bool
        if num != num_nodule:
            boolean_flag.append('False')
        else:
            boolean_flag.append('True')
    print("num_pred: ", np.sum(out_nodule_num))
    print("sum_nodules: ", np.sum(gt_nodule_num))
    print("False Positive: ", FP)

    case_name.append('Sum Value')
    out_nodule_num.append(np.sum(out_nodule_num))
    gt_nodule_num.append(np.sum(gt_nodule_num))
    boolean_flag.append('end')

    data = {'OUT': out_nodule_num, 'GT': gt_nodule_num, 'BOOL':boolean_flag}
    df = pd.DataFrame(data=data, columns=['OUT', 'GT', 'BOOL'], index=case_name)
    df.to_csv(join(cfg.Out_Arte, '{}_metrics.csv'.format(cfg.Arte_name)))
