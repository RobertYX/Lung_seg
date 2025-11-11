from skimage import measure, util
import numpy as np
import nibabel as nib

nodule = nib.load(r'F:\PY\lung_seg\DataSet\luna16\fix_data\roi\nodule_1.nii')
mask = nib.load(r'F:\PY\lung_seg\DataSet\luna16\fix_data\roi\mask_1.nii')

nodule_data = nodule.get_fdata()
mask_data = mask.get_fdata()

# 获取标签中的区域属性
labels = measure.label(mask_data)
regions = measure.regionprops(labels)

min_bbox = regions[0].bbox
for region in regions[1:]:
    bbox = region.bbox
    min_bbox = (min(min_bbox[0], bbox[0]),
                min(min_bbox[1], bbox[1]),
                min(min_bbox[2], bbox[2]),
                max(min_bbox[3], bbox[3]),
                max(min_bbox[4], bbox[4]),
                max(min_bbox[5], bbox[5]))

min_bbox = np.asarray(min_bbox)

# 截取原图中的区域
nodule_roi = util.crop(nodule_data, min_bbox)

# 创建新的原图文件
new_nodule = nib.Nifti1Image(nodule_roi, nodule.affine, nodule.header)
nib.save(new_nodule, r'F:\PY\lung_seg\DataSet\luna16\fix_data\roi\new_nodule_1.nii')

# 截取标签中的区域
label_roi = util.crop(mask_data, min_bbox)

# 创建新的标签文件
new_mask = nib.Nifti1Image(label_roi, mask.affine, mask.header)
nib.save(new_mask, r'F:\PY\lung_seg\DataSet\luna16\fix_data\roi\new_mask_1.nii')