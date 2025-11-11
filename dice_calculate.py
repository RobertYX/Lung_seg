# import nibabel as nib
# import numpy as np
#
# def dice_coef(im1, im2, smooth=1):
#     """
#     im1: NIFTI格式文件1的路径
#     im2: NIFTI格式文件2的路径
#     """
#     img1 = nib.load(im1).get_fdata()
#     img2 = nib.load(im2).get_fdata()
#
#     # 将二者进行降维，使其只保留前三个维度的信息
#     img1 = np.squeeze(img1)
#     img2 = np.squeeze(img2)
#
#     # 将图像中的值二值化
#     img1 = np.round(img1).astype(int)
#     img2 = np.round(img2).astype(int)
#
#     intersection = np.sum(img1 & img2)
#     total_voxels = np.sum(img1) + np.sum(img2)
#
#     # 计算 Dice 系数
#     dice = (2.0 * intersection + smooth) / (total_voxels + smooth)
#
#     return dice
#
# dice = dice_coef(r'F:\PY\lung_seg\output\result\c\a\1_label.nii.gz' ,r'F:\PY\lung_seg\output\result\data\amask_15.nii')
# print(dice)

def calculate_average(arr):
    total_sum = sum(arr)
    array_length = len(arr)
    average = total_sum / array_length
    return average

# 示例
my_array = [0.8872, 0.7618, 0.8370, 0.7353, 0.8369, 0.7857]
result = calculate_average(my_array)
print("数组的平均值为:", result)