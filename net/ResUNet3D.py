"""
This code is referenced from https://github.com/assassint2017/MICCAI-LITS2017
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2 ,training=True):
        super().__init__()

        self.training = training
        self.dorp_rate = 0.2

        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, 1, padding=1),
            nn.PReLU(16),

            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=2, dilation=2),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=4, dilation=4),
            nn.PReLU(64),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, padding=3, dilation=3),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=4, dilation=4),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=5, dilation=5),
            nn.PReLU(128),
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(128 + 64, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(32 + 16, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.PReLU(128)
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.PReLU(128)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.PReLU(64)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.PReLU(32)
        )

        # 最后大尺度下的映射（256*256），下面的尺度依次递减
        self.map4 = nn.Sequential(
            nn.Conv3d(32, out_channels, 1, 1),
            nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(64, out_channels, 1, 1),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(128, out_channels, 1, 1),
            nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear', align_corners=False),

            nn.Softmax(dim=1)
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(256, out_channels, 1, 1),
            nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )
        
        self.conv1x1 = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, inputs):

        long_range1 = self.encoder_stage1(inputs) + inputs

        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.encoder_stage2(short_range1) + short_range1
        long_range2 = F.dropout(long_range2, self.dorp_rate, self.training)

        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.encoder_stage3(short_range2) + short_range2
        long_range3 = F.dropout(long_range3, self.dorp_rate, self.training)

        short_range3 = self.down_conv3(long_range3)

        long_range4 = self.encoder_stage4(short_range3) + short_range3
        long_range4 = F.dropout(long_range4, self.dorp_rate, self.training)

        short_range4 = self.down_conv4(long_range4)

        outputs = self.decoder_stage1(long_range4) + short_range4
        outputs = F.dropout(outputs, self.dorp_rate, self.training)

        #output1 = self.map1(outputs)

        short_range6 = self.up_conv2(outputs)

        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6
        outputs = F.dropout(outputs, self.dorp_rate, self.training)

        #output2 = self.map2(outputs)

        short_range7 = self.up_conv3(outputs)

        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
        outputs = F.dropout(outputs, self.dorp_rate, self.training)

        #output3 = self.map3(outputs)

        short_range8 = self.up_conv4(outputs)
        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8
        
        outputs = self.conv1x1(outputs)
        # outputs = F.sigmoid(outputs)
        
        return outputs
        
        #output4 = self.map4(outputs)

        #if self.training is True:
            # return output1, output2, output3, output4
        # else:
        #     return output4

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ResUNet3D(nn.Module):
    def __init__(self, in_channels, n_classes, base_channels=64):
        super().__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.base_channels = base_channels

        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.conv_down1 = DoubleConv(in_channels, base_channels)
        self.conv_down2 = DoubleConv(base_channels, base_channels*2)
        self.conv_down3 = DoubleConv(base_channels*2, base_channels*4)
        self.conv_down4 = DoubleConv(base_channels*4, base_channels*8)

        self.conv_middle = DoubleConv(base_channels*8, base_channels*16)

        self.conv_up4 = DoubleConv(base_channels*16 + base_channels*8, base_channels*8)
        self.conv_up3 = DoubleConv(base_channels*8 + base_channels*4, base_channels*4)
        self.conv_up2 = DoubleConv(base_channels*4 + base_channels*2, base_channels*2)
        self.conv_up1 = DoubleConv(base_channels*2 + base_channels, base_channels)

        self.conv_last = nn.Conv3d(base_channels, n_classes, 1)

        self.residual1 = nn.Sequential(
            nn.Conv3d(base_channels*2, base_channels*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels*2, base_channels*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels*2)
        )

        self.residual2 = nn.Sequential(
            nn.Conv3d(base_channels*4, base_channels*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels*4),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels*4, base_channels*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels*4)
        )

        self.residual3 = nn.Sequential(
            nn.Conv3d(base_channels*8, base_channels*8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels*8),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels*8, base_channels*8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels*8)
        )

    def forward(self, x):
        conv1 = self.conv_down1(x)
        x = self.max_pool(conv1)

        conv2 = self.conv_down2(x)
        x = self.max_pool(conv2)

        conv3 = self.conv_down3(x)
        x = self.max_pool(conv3)

        conv4 = self.conv_down4(x)
        x = self.max_pool(conv4)

        x = self.conv_middle(x)

        x = self.upsample(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.conv_up4(x)
        x = self.residual3(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.conv_up3(x)
        x = self.residual2(x)

        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv_up2(x)
        x = self.residual1(x)

        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.conv_up1(x)

        output = self.conv_last(x)

        return output

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.randn([1, 1, 96, 96, 96]).to(device)
    # unet = ResUNet(in_channels=1, out_channels=2).to(device)
    unet = ResUNet3D(in_channels=1, n_classes=2, base_channels=32).to(device)
    print('#parameters:', sum(param.numel() for param in unet.parameters()))
    out = unet(tensor)
    print(out.shape)