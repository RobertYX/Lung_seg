#################################################
#      CreateTime:20230511                   
#      UpdateTIme:2023(The Last)         
#      Creator: FuZhou University iipa.fzu.edu.cn
#      Main Editor: Gunhild
#      "To be or not to be." --ShakeSpear
#################################################


# For Luna16 Dataset

import os
join = os.path.join
# 预处理参数集合
# 预处理后最终的xyz方向尺寸
xyz_down_scale = (96, 96, 96)

# 模型参数集合
in_channel = 1 # 模型输入通道
num_class = 2  # 分割几类(包含背景)
if_freeze = False # 是否冻结参数
Architecture = {'UNet3D':'UNet3D', 'ResUNet3D':'ResUNet3D'} # 选择一个模型架构
Arte_name = Architecture['ResUNet3D'] # 选择UNet_CBAM

# 路径集合
#---可能要修改的路径---#
OrginalPath = './DataSet/' # 数据集根目录
RawDataPath = OrginalPath + 'luna16/LUNA16_Seg' # 原始数据路径
LaeDataPath = OrginalPath + 'luna16/annotations.csv'# 标注数据路径(可选)
FixDataPath = OrginalPath + 'luna16/fix_data/' # 预处理后数据路径
ValDataPath = OrginalPath + 'luna16/subset/' # 用于验证的原始数据路径
PreDataPath = OrginalPath + 'luna16/LUNA16_Seg/subset9/' # 用于预测的原始数据路径

#-------------------#
TraintxtDatapath = FixDataPath + 'train.txt' # 训练数据的目录txt文件路径
ValidtxtDatapath = FixDataPath + 'valid.txt'  # 验证数据的目录txt文件路径


OutDatapath = './output/' # 所有方法结果的主目录

Out_Arte = join(OutDatapath, Arte_name) # 每个方法的目录
ModelSavepath = join(Out_Arte, 'model_saved') # 每个方法的模型保存路径
FileSavepath = join(Out_Arte, 'prefile_saved') # 每个方法的输出文件路径

# 设备参数
multi_gpu = False

# 训练参数设置
epoch = 20 # 训练轮数 100
learning_rate = 1e-4 # 学习率

# 测试参数设置
postprocess = False

if __name__ == '__main__':
    # 如果不存在该方法的保存路径, 则创造一个
    if not os.path.exists(Out_Arte):
        os.makedirs(Out_Arte)
    # 如果不存在该方法的模型权重保存路径, 则创造一个    
    if not os.path.exists(ModelSavepath):
        os.makedirs(ModelSavepath)
    # 如果不存在该方法的输出文件保存路径, 则创造一个 
    if not os.path.exists(FileSavepath):
        os.makedirs(FileSavepath)    
