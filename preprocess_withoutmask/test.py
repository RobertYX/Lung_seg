import torch
from create_model import model

# checkpoint = torch.load('F:\PY\lung_seg\output\ResUNet3D\model_saved\model_best_eval.pth')
# print(checkpoint.keys())
# model.load_state_dict(checkpoint)
# print(model)

import os
import shutil

data_dir = r'F:\PY\lung_seg\DataSet\luna16\fix_data\valid'  # 指定数据集所在的文件夹
mask_dir = r'F:\PY\lung_seg\DataSet\luna16\fix_data\valid'  # 指定存储 mask 数据的文件夹
nodule_dir = r'F:\PY\lung_seg\DataSet\luna16\fix_data\valid'  # 指定存储 nodule 数据的文件夹

# 遍历 data_dir 下的所有子文件夹
for root, dirs, files in os.walk(data_dir):
    # 遍历当前子文件夹中的所有文件
    for file in files:
        # 如果文件名以 ".nii" 结尾，则进行处理
        if file.endswith('.nii'):
            # 获取子文件夹的数字名称
            num = root.split("\\")[-1].split('A')[-1]  # 假设文件夹名字为类似 "/path/to/data/123" 的格式
            print(num)
            # 处理 ROI.nii 文件
            if file == 'ROI.nii':
                # 构建 mask 文件的目标路径，并复制 ROI.nii 文件
                mask_file = f"mask_{num}.nii"
                mask_path = os.path.join(mask_dir, mask_file)
                src_path = os.path.join(root, file)
                shutil.copy(src_path, mask_path)

            # 处理另一个文件
            else:
                # 构建 nodule 文件的目标路径，并复制文件
                nodule_file = f"nodule_{num}.nii"
                nodule_path = os.path.join(nodule_dir, nodule_file)
                src_path = os.path.join(root, file)
                shutil.copy(src_path, nodule_path)