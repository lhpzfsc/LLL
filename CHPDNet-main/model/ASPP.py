import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

class simam_module(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

class ASPP(nn.Module):
    def __init__(self, out_channels=256):
        super(ASPP, self).__init__()
        self.layer6_0 = nn.Sequential(
            nn.Conv2d(out_channels , out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
        )
        self.layer6_1 = nn.Sequential(
            nn.Conv2d(out_channels , out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            )
        self.layer6_2 = nn.Sequential(
            nn.Conv2d(out_channels , out_channels , kernel_size=3, stride=1, padding=6,dilation=6, bias=True),
            nn.ReLU(),
            )
        self.layer6_3 = nn.Sequential(
            nn.Conv2d(out_channels , out_channels, kernel_size=3, stride=1, padding=12, dilation=12, bias=True),
            nn.ReLU(),
            )
        self.layer6_4 = nn.Sequential(
            nn.Conv2d(out_channels , out_channels , kernel_size=3, stride=1, padding=18, dilation=18, bias=True),
            nn.ReLU(),
            )
        self.pre_siam = simam_module()
        self.lat_siam = simam_module()


        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, x):
        # feature_size = x.shape[-2:]
        # global_feature = F.avg_pool2d(x, kernel_size=feature_size)
        # global_feature = self.layer6_0(global_feature)
        # global_feature = global_feature.expand(-1, -1, feature_size[0], feature_size[1])

        global_feature = self.pre_siam(self.layer6_0(x))

        out = torch.cat(
            [global_feature, self.layer6_1(x), self.layer6_2(x), self.layer6_3(x), self.layer6_4(x)], dim=1)
        out =self.lat_siam(out)
        return out


if __name__ == '__main__':
    input_shape = (1, 256, 64, 64)
    input = torch.randn(input_shape)

    # 实例化FMB类
    block = ASPP(256)

    # 将输入张量传入FMB实例
    output = block(input)

    # 打印输入和输出的形状
    print(input.size())
    print(output.size())