import torch
from torch import nn


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class transconv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(transconv_block, self).__init__()

        self.trans_conv = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.trans_conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        n = 32
        filters = [n, n*2, n*4, n*8, n*16]
        self.conv0 = conv_block(in_channels, filters[0])
        self.conv1 = conv_block(filters[0], filters[1])
        self.conv2 = conv_block(filters[1], filters[2])
        self.conv3 = conv_block(filters[2], filters[3])

        self.conv6 = conv_block(filters[3], filters[2])
        self.conv7 = conv_block(filters[2], filters[1])
        self.conv8 = conv_block(filters[1], filters[0])

        self.maxpool1 = nn.MaxPool3d(2)
        self.maxpool2 = nn.MaxPool3d(2)
        self.maxpool3 = nn.MaxPool3d(2)

        # self.transconv3 = nn.ConvTranspose3d(filters[3], filters[3] // 2, kernel_size=2, stride=2)
        # self.transconv2 = nn.ConvTranspose3d(filters[2], filters[2] // 2, kernel_size=2, stride=2)
        # self.transconv1 = nn.ConvTranspose3d(filters[1], filters[1] // 2, kernel_size=2, stride=2)
        self.transconv3 = transconv_block(filters[3], filters[3] // 2)
        self.transconv2 = transconv_block(filters[2], filters[2] // 2)
        self.transconv1 = transconv_block(filters[1], filters[1] // 2)

        self.conv1x1 = nn.Conv3d(filters[0], out_channels, kernel_size=1)

        self.init_conv3d()

    def init_conv3d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv3d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, x):
        # encoder #
        x = self.conv0(x)
        x_cat1 = x
        x = self.maxpool1(x)

        x = self.conv1(x)
        x_cat2 = x
        x = self.maxpool2(x)

        x = self.conv2(x)
        x_cat3 = x
        x = self.maxpool3(x)

        mid_feats = self.conv3(x)

        x_trans3 = self.transconv3(mid_feats)
        x = torch.cat([x_cat3, x_trans3], dim=1)
        x = self.conv6(x)

        x_trans2 = self.transconv2(x)
        x = torch.cat([x_cat2, x_trans2], dim=1)
        x = self.conv7(x)

        x_trans1 = self.transconv1(x)
        x = torch.cat([x_cat1, x_trans1], dim=1)
        x = self.conv8(x)

        output = self.conv1x1(x)

        return output


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.randn([1, 1, 96, 96, 96]).to(device)
    unet = UNet(in_channels=1, out_channels=2).to(device)
    print('#parameters:', sum(param.numel() for param in unet.parameters()))
    out = unet(tensor)
    print(out.shape)