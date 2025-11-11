import os
import logging
import numpy as np
import torch
from torch.nn.functional import one_hot
from skimage import measure

def replace_block(ori_block:np.ndarray, input_block:np.ndarray, secZ:list=[0.0, 1.0], secY:list=[0.0, 1.0], secX:list=[0.0, 1.0]):
    # 更换部分区域为指定值
    # 可以用在预测ROI后, 反向安装
    # ori_block: 原始图像
    # input_block: 输入图像
    secZ = list(map(int, secZ))
    secY = list(map(int, secY))
    secX = list(map(int, secX))

    errorZ = input_block.shape[0] - (secZ[1] - secZ[0])
    errorY = input_block.shape[1] - (secY[1] - secY[0])
    errorX = input_block.shape[2] - (secX[1] - secX[0])
    print('secz',secZ)
    print(secZ[0])
    print(secZ[1]+errorZ)
    print(secX)
    print(ori_block.shape)
    print(input_block.shape)
    ori_block[secZ[0]:secZ[1]+errorZ, secY[0]:secY[1]+errorY, secX[0]:secX[1]+errorX] = input_block

    return ori_block

def torch_numpy(input_tensor):
    # BCDHW
    input_tensor = torch.softmax(input_tensor, dim=1)
    input_tensor = torch.argmax(input_tensor, dim=1)
    # BDHW
    input_tensor = input_tensor.data.cpu().numpy()
    output_tensor = input_tensor.astype(np.uint8)

    return output_tensor

def generate_mask(img_hwd:list=[int, int, int], radius:int=1, center_xyz:list=[int, int, int]):
    # 生成球状mask
    # img_hwd 是数组形状
    # radius 是半径
    # center_xyz 是球心坐标
    x = np.array(list(range(img_hwd[0]))).reshape([img_hwd[0],1,1])
    y = np.array(list(range(img_hwd[1]))).reshape([1,img_hwd[1],1])
    z = np.array(list(range(img_hwd[2]))).reshape([1,1,img_hwd[2]])
    # circle mask
    mask = (x-center_xyz[0])**2+(y-center_xyz[1])**2+(z-center_xyz[2])**2<=radius**2  
 
    return mask

def normlize(image, MIN_BOUND=-1000., MAX_BOUND=400.):
    # 阈值外截断——归一化+标准化
    ## 阈值外截断——归一化
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    ## 标准化
    i_mean = np.mean(image) # 计算均值
    i_std = np.std(image) #计算标准差
    image = (image-i_mean)/i_std  # 标准化

    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image[image > 1] = 1.
    image[image < 0] = 0.

    return image


def worldToVoxelCoord(worldCoord, origin=[-195, -195, -378], spacing=[0.7617189884185791,0.7617189884185791,2.5]):
    # 从绝对坐标中转到体系坐标
    
    # worldCoord 绝对坐标
    # mhd 原点
    # spacing 像素间距
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

def VoxelToWorldCoord(voxelCoord, origin, spacing):
    # 从体系坐标中转到绝对坐标
    
    # worldCoord 绝对坐标
    # mhd 原点
    # spacing 像素间距

    strechedVocelCoord = voxelCoord * spacing
    worldCoord = strechedVocelCoord + origin
    return worldCoord

def checkBoundry(self_array:np.ndarray, SE:list=[int, int], axis:int=0):
    """
        检验是否越界,返回不越界的首尾参数
        label_array: 本体尺寸, 不可逾越的
        SE: 已裁切的尺寸头尾
        axis: 指定维度
    """
    if SE[0] < 0:
        SE[0] = 0
    if SE[1] >= self_array.shape[axis]:
        SE[1] = self_array.shape[axis] - 1

    return SE

def logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def dice_coeff(output, target):
    smooth = 1e-5
    prediction = torch.softmax(output, dim=1)
    prediction = torch.argmax(prediction, dim=1)
    if len(prediction.shape)==3:
        prediction = one_hot(prediction.long(), num_classes=2).permute(0, 3, 1, 2).contiguous()
        target = one_hot(target, num_classes=2).permute(0, 3, 1, 2).contiguous()
    else:
        prediction = one_hot(prediction.long(), num_classes=2).permute(0, 4, 1, 2, 3).contiguous()
        target = one_hot(target, num_classes=2).permute(0, 4, 1, 2, 3).contiguous()

    batchsize = target.size(0)
    num_classes = target.size(1)
    prediction = prediction.view(batchsize, num_classes, -1)
    target = target.view(batchsize, num_classes, -1)

    intersection = (prediction * target)

    dice = (2. * intersection.sum(2) + smooth) / (prediction.sum(2) + target.sum(2) + smooth)
    dice = dice.detach().tolist()
    dsc = dice[0][1]

    return dsc

def measureimg(o_img,t_num=1):
    "保留o_img中按大小排序后的前t_num个连通域"
    p_img=np.zeros_like(o_img)
    # temp_img=morphology.binary_dilation(o_img.astype("bool"),iterations=2)
    testa1 = measure.label(o_img.astype("bool"))
    props = measure.regionprops(testa1)
    numPix = []
    for ia in range(len(props)):
        numPix += [props[ia].area]
    # print(numPix)
    # 像素最多的连通区域及其指引
    for i in range(0, t_num):
        index = numPix.index(max(numPix)) + 1
        p_img[testa1 == index]=o_img[testa1 == index]
        numPix[index-1]=0
    return p_img

if __name__ == '__main__':
    mask = generate_mask([4,4,4],1,[2,2,2]).astype(int)
    print(mask.shape)
    # print(mask)
    # mask[mask==True]=255clear
    print(np.unique(mask))