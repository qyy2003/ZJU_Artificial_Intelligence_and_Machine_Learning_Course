import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F




class MobileNet(nn.Module):
    def __init__(self, classes=2):
        super(MobileNet, self).__init__()
        lastconv_output_channels=960
        last_channel=50
        self.mobilebone = torchvision.models.mobilenet_v3_large(pretrained=True, width_mult=1.0,  reduced_tail=False, dilated=False).features
        for param in self.mobilebone.parameters():
            # print(param)
            param.requires_grad = False
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels, last_channel),
            # nn.ReLU(),
            # nn.Hardswish(inplace=True),
            # nn.Dropout(p=0.2, inplace=True),
            nn.Linear(last_channel, classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.mobilebone(x)
        # print(x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)
        return x

    def _top_conv(self, in_channel, out_channel, blocks):
        layers = []
        for i in range(blocks):
            layers.append(self._conv_dw(in_channel, out_channel, 1))
        return nn.Sequential(*layers)

    def _conv_bn(self, in_channel, out_channel, stride):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def _conv_dw(self, in_channel, out_channel, stride):
        return nn.Sequential(
            #nn.Conv2d(in_channel, in_channel, 3, stride, 1, groups=in_channel, bias=False),
            #nn.BatchNorm2d(in_channel),
            #nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=False),
        )


if __name__ == "__main__":
    import numpy as np

    x = np.zeros((3, 160, 160))
    x = torch.from_numpy(x).float().unsqueeze(0)
    print(x.shape)
    con_block = MobileNet(2)
    prob = con_block(x)
    print(prob.shape)


# class MobileNet(nn.Module):
#     def __init__(self, classes=2):
#         super(MobileNet, self).__init__()
#         self.mobilebone = torchvision.models.mobilenet_v2(pretrained=True).features
#         # for param in self.mobilebone.parameters():
#         #     param.requires_grad = False
#         # self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
#         self.classifier = nn.Sequential(
#             nn.Dropout(p=0.2),
#             nn.Linear(1280, classes),
#         )
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, (2. / n) ** .5)
#             if isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def forward(self, x):
#         x = self.mobilebone(x)
#         x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x
#
#     def _top_conv(self, in_channel, out_channel, blocks):
#         layers = []
#         for i in range(blocks):
#             layers.append(self._conv_dw(in_channel, out_channel, 1))
#         return nn.Sequential(*layers)
#
#     def _conv_bn(self, in_channel, out_channel, stride):
#         return nn.Sequential(
#             nn.Conv2d(in_channel, out_channel, 3, stride, padding=1, bias=False),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU(inplace=True),
#         )
#
#     def _conv_dw(self, in_channel, out_channel, stride):
#         return nn.Sequential(
#             #nn.Conv2d(in_channel, in_channel, 3, stride, 1, groups=in_channel, bias=False),
#             #nn.BatchNorm2d(in_channel),
#             #nn.ReLU(inplace=True),
#             nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU(inplace=False),
#         )
