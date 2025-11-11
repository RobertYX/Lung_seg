#################################################
#      CreateTime:20221026                   
#      UpdateTIme:20230315(The Last)         
#      Creator: FuZhou University iipa.fzu.edu.cn
#      Main Editor: Gunhild
#      "To be or not to be." --ShakeSpear
#################################################

import torch
import config as cfg

from net.UNet3D import UNet
from net.ResUNet3D import ResUNet, ResUNet3D
from networks.d_lka_former.d_lka_net_synapse import D_LKA_Net
from networks.d_lka_former.transformerblock import TransformerBlock_3D_single_deform_LKA, TransformerBlock


Arte_name = cfg.Arte_name
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 输入输出
in_channel = cfg.in_channel
num_classes = cfg.num_class

# 选择模型

if Arte_name == 'UNet3D':
    model = UNet(in_channels=in_channel, out_channels=num_classes)
elif Arte_name == 'ResUNet3D':
    # model = ResUNet(in_channels=in_channel, out_channels=num_classes)
    # model = ResUNet3D(in_channels=1, n_classes=2, base_channels=64)
    model = D_LKA_Net(in_channels=1,
                           out_channels=num_classes,
                           img_size=[96, 96, 96],
                           patch_size=(2,2,2),
                           input_size=[48*48*48, 24*24*24,12*12*12,6*6*6],
                           trans_block=TransformerBlock_3D_single_deform_LKA,
                           do_ds=False)
else:
    model = None
    print("No Model Found.")
    exit(0)

if __name__ == '__main__':
    # 测试模型是否能正常运行
    # 自定义一个期望输入的tensor变量
    input = torch.randn(1, in_channel, 96, 96, 96).to(device) # B C D H W 
    model = model.to(device)
    output = model(input)
    print(output.shape)
    # 计算模型计算量和参数量
    from thop import profile
    flops, params = profile(model, inputs=(input,))
    print("Params(M): ", params / (1024 ** 2))
    print("Flops(G): ", flops / (1024 ** 3))

