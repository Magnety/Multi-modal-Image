import torch
import torch.nn as nn


class Fire(nn.Module):

    def __init__(self, in_channel, out_channel, squzee_channel):

        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channel, squzee_channel, 1),
            nn.BatchNorm2d(squzee_channel),
            nn.ReLU(inplace=True)
        )

        self.expand_1x1 = nn.Sequential(
            nn.Conv2d(squzee_channel, int(out_channel / 2), 1),
            nn.BatchNorm2d(int(out_channel / 2)),
            nn.ReLU(inplace=True)
        )

        self.expand_3x3 = nn.Sequential(
            nn.Conv2d(squzee_channel, int(out_channel / 2), 3, padding=1),
            nn.BatchNorm2d(int(out_channel / 2)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x = self.squeeze(x)
        x = torch.cat([
            self.expand_1x1(x),
            self.expand_3x3(x)
        ], 1)

        return x

class OursNet(nn.Module):

    """mobile net with simple bypass"""
    def __init__(self, class_num=100):

        super().__init__()
        self.stem_1 = nn.Sequential(
            nn.Conv2d(3, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.stem_2 = nn.Sequential(
            nn.Conv2d(3, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.fire2_1 = Fire(96, 128, 16)
        self.fire3_1 = Fire(128, 128, 16)
        self.fire4_1 = Fire(128, 256, 32)
        self.fire5_1 = Fire(256, 256, 32)
        self.fire6_1 = Fire(256, 384, 48)
        self.fire7_1 = Fire(384, 384, 48)
        self.fire8_1 = Fire(384, 512, 64)
        self.fire9_1 = Fire(512, 512, 64)
        self.fire2_2 = Fire(96, 128, 16)
        self.fire3_2 = Fire(128, 128, 16)
        self.fire4_2 = Fire(128, 256, 32)
        self.fire5_2 = Fire(256, 256, 32)
        self.fire6_2 = Fire(256, 384, 48)
        self.fire7_2 = Fire(384, 384, 48)
        self.fire8_2 = Fire(384, 512, 64)
        self.fire9_2 = Fire(512, 512, 64)

        self.conv10 = nn.Conv2d(1024, class_num, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.fc = nn.Softmax(dim=1)

    def forward(self, x1,x2):
        x1 = self.stem_1(x1)

        f2_1 = self.fire2_1(x1)
        f3_1 = self.fire3_1(f2_1) + f2_1
        f4_1 = self.fire4_1(f3_1)
        f4_1 = self.maxpool(f4_1)
        f5_1 = self.fire5_1(f4_1) + f4_1
        f6_1 = self.fire6_1(f5_1)
        f7_1 = self.fire7_1(f6_1) + f6_1
        f8_1 = self.fire8_1(f7_1)
        f8_1 = self.maxpool(f8_1)
        f9_1 = self.fire9_1(f8_1)

        x2 = self.stem_2(x2)

        f2_2 = self.fire2_2(x2)
        f3_2 = self.fire3_2(f2_2) + f2_2
        f4_2 = self.fire4_2(f3_2)
        f4_2 = self.maxpool(f4_2)
        f5_2 = self.fire5_2(f4_2) + f4_2
        f6_2 = self.fire6_2(f5_2)
        f7_2 = self.fire7_2(f6_2) + f6_2
        f8_2 = self.fire8_2(f7_2)
        f8_2 = self.maxpool(f8_2)
        f9_2 = self.fire9_2(f8_2)
        f9 = torch.cat((f9_1, f9_2), dim=1)
        feature = f9.view(f9.size()[0], -1)
        c10 = self.conv10(f9)
        x = self.avg(c10)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x,x1,x2,f2_1,f2_2,f4_1,f4_2,f6_1,f6_2,f9_1,f9_2
def ours(class_num=2):
    return OursNet(class_num=class_num)
