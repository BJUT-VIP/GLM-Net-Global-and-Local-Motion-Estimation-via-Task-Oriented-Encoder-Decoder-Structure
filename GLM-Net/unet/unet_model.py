""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.up_without_concat_1 = Up_without_concat(1024, 512)
        self.up_without_concat_2 = Up_without_concat(512, 256)
        self.up_without_concat_3 = Up_without_concat(256, 128)
        self.up_without_concat_4 = Up_without_concat(128, 64)

    def forward(self, x):
        # with concat
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        # without concat
        # x = self.inc(x)
        # x = self.down1(x)
        # x = self.down2(x)
        # x = self.down3(x)
        # x = self.down4(x)
        # x = self.up_without_concat_1(x)
        # x = self.up_without_concat_2(x)
        # x = self.up_without_concat_3(x)
        # x = self.up_without_concat_4(x)
        # logits = self.outc(x)

        return logits

class UNet_noC(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_noC, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.up_without_concat_1 = Up_without_concat(1024, 512)
        self.up_without_concat_2 = Up_without_concat(512, 256)
        self.up_without_concat_3 = Up_without_concat(256, 128)
        self.up_without_concat_4 = Up_without_concat(128, 64)

    def forward(self, x):
        # with concat
        # x1 = self.inc(x)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        # logits = self.outc(x)

        # without concat
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.up_without_concat_1(x)
        x = self.up_without_concat_2(x)
        x = self.up_without_concat_3(x)
        x = self.up_without_concat_4(x)
        logits = self.outc(x)

        return logits


class Attention_U_Net(nn.Module):
    def __init__(self, n_channels, n_classes, upsample=True):
        super(Attention_U_Net, self).__init__()

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.upsample = upsample

        self.Up5 = up_conv(1024, 512, upsample)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = DoubleConv(1024, 512)

        self.Up4 = up_conv(512, 256, upsample)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = DoubleConv(512, 256)

        self.Up3 = up_conv(256, 128, upsample)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = DoubleConv(256, 128)

        self.Up2 = up_conv(128, 64, upsample)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = DoubleConv(128, 64)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # encoding path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # x2 = self.Maxpool(x1)
        # x2 = self.Conv2(x2)
        #
        # x3 = self.Maxpool(x2)
        # x3 = self.Conv3(x3)
        #
        # x4 = self.Maxpool(x3)
        # x4 = self.Conv4(x4)
        #
        # x5 = self.Maxpool(x4)
        # x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4, psi4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3, psi3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2, psi2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1, psi1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.outc(d2)

        # return d1, psi1, psi2, psi3, psi4
        return d1

