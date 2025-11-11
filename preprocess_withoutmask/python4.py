import os
import numpy as np
import nibabel as nib

# 读取numpy数组形式的数据
image_data = np.load(cfg.FixDataPath + seperate_set + '/nodule_{}_{}.npy'.format(str(idx).rjust(3, '0'), ndx))


# 将数据保存为nii格式文件
nifti_image = nib.Nifti1Image(image_data)
filename = cfg.FixDataPath + seperate_set + '/nodule_{}_{}.nii.gz'.format(str(idx).rjust(3, '0'), ndx)
nib.save(nifti_image, filename)