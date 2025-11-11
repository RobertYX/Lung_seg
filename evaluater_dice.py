#################################################
#      CreateTime:20221026                   
#      UpdateTIme:20230315(The Last)         
#      Creator: FuZhou University iipa.fzu.edu.cn
#      Main Editor: Gunhild
#      "To be or not to be." --ShakeSpear
#################################################

import os
import SimpleITK as sitk
import pandas as pd
import numpy as np
import surface_distance as surfdist
# 请手动安装库 https://github.com/deepmind/surface-distance 

import config as cfg
from skimage import measure
from tqdm import tqdm

join = os.path.join

Out_Arte = cfg.Out_Arte
Arte_name = cfg.Arte_name

FileSavepath =  cfg.FileSavepath# 预测文件目录
PreDataPath = cfg.PreDataPath # 源文件目录

pred_paths = [os.path.join(FileSavepath, x) # prediction path
                for x in os.listdir(FileSavepath)
                    if x.endswith("_label.nii.gz")]

gt_paths = [os.path.join(PreDataPath, x) # groundtruth path
                for x in os.listdir(PreDataPath)
                    if x.endswith("_seg.nii.gz")]

pred_paths.sort()
gt_paths.sort()
print(len(pred_paths) )
print(len(gt_paths))
assert len(pred_paths) == len(gt_paths) # check number

case_name = []
dice = []
assd = []
so = []

for idx in tqdm(range(len(pred_paths))):
    name = pred_paths[idx].split('/')[-1].replace("_label.nii.gz", "")
    print("Try to Process: ", name)
    case_name.append(name) # _'_
    # Get ITK of pred and label
    pred_image, label_image = sitk.ReadImage(pred_paths[idx]), sitk.ReadImage(gt_paths[idx])

    # Spacing equal
    spacing = label_image.GetSpacing()
    pred_image.SetSpacing(spacing)

    # ITK 2 Numpy
    pred, label = sitk.GetArrayFromImage(pred_image).astype('int16'), sitk.GetArrayFromImage(label_image).astype('int16')

    # ReSet pixel value
    pred[pred!=0]=1
    label[label!=0]=1
    # label[label==2]=1
    label, region_nums = measure.label(label, background=0, return_num=True, connectivity=2)
    if region_nums!=0:
        pred = pred.astype(bool)
        label = label.astype(bool)

        # Calculate Dice(3D), ASSD, HD, SO(Surface Overlap)
        # 3DDice
        volume_dice = surfdist.compute_dice_coefficient(label, pred)
        # basic: surface_distances
        surface_distances = surfdist.compute_surface_distances(label, pred, spacing_mm=spacing)
        # ASSD
        avg_surf_dist = surfdist.compute_average_surface_distance(surface_distances)
        avg_surf_dist = (avg_surf_dist[0] + avg_surf_dist[1])/2
        # SO
        surface_overlap = surfdist.compute_surface_overlap_at_tolerance(surface_distances, 1)
        surface_overlap = (surface_overlap[0] + surface_overlap[1]) / 2
    else:
        volume_dice = 1.00
        avg_surf_dist = 0.00
        surface_overlap = 1.00
    dice.append(volume_dice)  # _'_
    assd.append(avg_surf_dist)
    so.append(surface_overlap)


    print("3DDice: ", volume_dice)
    print("ASSD: ", avg_surf_dist)
    print("SO: ", surface_overlap)

    print("=========================================")

print("3DDice: ", np.mean(dice), "ASSD: ", np.mean(assd), "SO: ", np.mean(so))
# _'_
case_name.append('Mean Value')
dice.append(np.mean(dice))
assd.append(np.mean(assd))
so.append(np.mean(so))

data = {'3DDice': dice, 'ASSD': assd, 'SO': so}
df = pd.DataFrame(data=data, columns=['3DDice', 'ASSD', 'SO'], index=case_name)
df.to_csv(join(Out_Arte, '{}_metrics_dice.csv'.format(Arte_name)))
