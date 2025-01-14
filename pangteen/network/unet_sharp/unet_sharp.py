import torch
from torch import nn

class Decoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv_relu(x1)
        return x1


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNet_sharp(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        #         self.bn = nn.BatchNorm2d(nb_filter[0])
        #         self.out_bn = nn.BatchNorm2d(nb_filter[0]*5)
        #         self.re = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(4, 4)
        self.pool8 = nn.MaxPool2d(8, 8)
        self.pool16 = nn.MaxPool2d(16, 16)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1] + nb_filter[2], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2] + nb_filter[3], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3] + nb_filter[4], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1] + nb_filter[2] + nb_filter[3], nb_filter[0],
                                nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2] + nb_filter[3] + nb_filter[4], nb_filter[1],
                                nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1] + nb_filter[2] + nb_filter[3] + nb_filter[4],
                                nb_filter[0], nb_filter[0])

        #         self.x20_conv = nn.Conv2d(nb_filter[2], nb_filter[0], 3, padding=1)
        #         self.x30_conv = nn.Conv2d(nb_filter[3], nb_filter[0], 3, padding=1)
        #         self.x21_conv = nn.Conv2d(nb_filter[2], nb_filter[0], 3, padding=1)
        #         self.x40_conv = nn.Conv2d(nb_filter[4], nb_filter[0], 3, padding=1)
        #         self.x31_conv = nn.Conv2d(nb_filter[3], nb_filter[0], 3, padding=1)
        #         self.x22_conv = nn.Conv2d(nb_filter[2], nb_filter[0], 3, padding=1)

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final5 = nn.Conv2d(nb_filter[4], num_classes, kernel_size=3, padding=1)
            self.final6 = nn.Conv2d(nb_filter[3], num_classes, kernel_size=3, padding=1)
            self.final7 = nn.Conv2d(nb_filter[2], num_classes, kernel_size=3, padding=1)
            self.final8 = nn.Conv2d(nb_filter[1], num_classes, kernel_size=3, padding=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input, l=4):
        # ---------Downsample-------------
        # ===>down layer 1
        x0_0 = self.conv0_0(input)
        # ===>down layer 2
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
        else:
            l = 4
        if l == 1:
            return [output1]

        # ===>down layer 3
        x2_0 = self.conv2_0(self.pool(x1_0))
        #         x2_0_up4 = self.re(self.bn(self.x20_conv(self.up4(x2_0))))#c=32
        x2_0_up4 = self.up4(x2_0)  # c=128
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1), x2_0_up4], 1))

        if self.deep_supervision:
            output2 = self.final2(x0_2)
        if l == 2:
            return [output2]

        # ===>down layer 4
        x3_0 = self.conv3_0(self.pool(x2_0))
        #         x3_0_up4 = self.re(self.bn(self.x30_conv(self.up4(x3_0))))#c=32
        #         x3_0_up8 = self.re(self.bn(self.x30_conv(self.up8(x3_0))))#c=32
        x3_0_up4 = self.up4(x3_0)  # c=256
        x3_0_up8 = self.up8(x3_0)  # c=256
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        #         x2_1_up4 = self.re(self.bn(self.x21_conv(self.up4(x2_1))))#c=32
        x2_1_up4 = self.up4(x2_1)  # c=128
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1), x3_0_up4], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2), x2_1_up4, x3_0_up8], 1))

        if self.deep_supervision:
            output3 = self.final3(x0_3)
        if l == 3:
            return [output3]

        #         #-----------Upsample----------------
        #         #===>down layer 5
        x4_0 = self.conv4_0(self.pool(x3_0))  # c=512
        #         x4_0_up4 = self.re(self.bn(self.x40_conv(self.up4(x4_0))))#c=32
        #         x4_0_up8 = self.re(self.bn(self.x40_conv(self.up8(x4_0))))#c=32
        #         x4_0_up16 = self.re(self.bn(self.x40_conv(self.up16(x4_0))))#c=32
        x4_0_up4 = self.up4(x4_0)  # c=512
        x4_0_up8 = self.up8(x4_0)  # c=512
        x4_0_up16 = self.up16(x4_0)  # c=512

        #         #===>up layer 4
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))  # c=256
        #         x3_1_up4 = self.re(self.bn(self.x31_conv(self.up4(x3_1))))#c=32
        #         x3_1_up8 = self.re(self.bn(self.x31_conv(self.up8(x3_1))))#c=32
        x3_1_up4 = self.up4(x3_1)  # c=256
        x3_1_up8 = self.up8(x3_1)  # c=256

        # ===>up layer 3
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1), x4_0_up4], 1))  # c=128
        #         x2_2_up4 = self.re(self.bn(self.x22_conv(self.up4(x2_2))))#c=32
        x2_2_up4 = self.up4(x2_2)  # c=128

        # ===>up layer 2
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2), x3_1_up4, x4_0_up8], 1))  # c=64
        # ===>up layer 1
        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3), x2_2_up4, x3_1_up8, x4_0_up16], 1))  # c=32

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output5 = self.final5(x4_0)
            output5 = self.up16(output5)
            output6 = self.final6(x3_1)
            output6 = self.up8(output6)
            output7 = self.final7(x2_2)
            output7 = self.up4(output7)
            output8 = self.final8(x1_3)
            output8 = self.up(output8)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output5, output6, output7, output8, output4]
        else:
            output = self.final(x0_4)
            return [output]



