import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1*h*w
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        # 2*h*w
        x = self.conv(x)
        # 1*h*w
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, c1, c2, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(c1, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        # c*h*w
        # c*h*w * 1*h*w
        out = self.spatial_attention(out) * out
        return out

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3)
        return scale



class FEM(nn.Module):
    def __init__(self, inplanes, reduction_ratio=16):
        super(FEM, self).__init__()

        self.in_channels = inplanes
        self.inter_channels = 1
        if self.inter_channels == 0:
            self.inter_channels = 1

        self.common_v = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                         padding=0)
        self.Conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3,stride=1, padding=1)
        self.ChannelGate1 = ChannelGate(self.in_channels, pool_types=['max'], reduction_ratio=reduction_ratio)
        self.ChannelGate2 = ChannelGate(self.in_channels, pool_types=['max'], reduction_ratio=reduction_ratio)


    def forward(self, q, s_mask, mask_feat):
        bsize, ch_sz, ha, wa = q.size()[:]

        tmp_query = q.contiguous().view(bsize, ch_sz, -1)  # 4*256*3600
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True)  # 4*1*3600

        tmp_supp = s_mask.contiguous().view(bsize, ch_sz, -1)  # 4*256*3600
        tmp_supp = tmp_supp.contiguous().permute(0, 2, 1)  # 4*3600*256
        tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)  # 4*3600*1

        tmp_test_mask = mask_feat.contiguous().view(bsize, ch_sz, -1)  # 4*256*3600
        tmp_test_mask = tmp_test_mask.contiguous().permute(0, 2, 1)  # 4*3600*256
        tmp_tets_mask_norm = torch.norm(tmp_test_mask, 2, 2, True)  # 4*3600*1

        similarity_1 = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + 1e-7)             #  4*3600*3600

        similarity_1 = similarity_1.max(1)[0].view(bsize, ha * ha)
        similarity_1 = (similarity_1 - similarity_1.min(1)[0].unsqueeze(1)) / (similarity_1.max(1)[0].unsqueeze(1) - similarity_1.min(1)[0].unsqueeze(1) + 1e-7)
        similarity_1 = similarity_1.view(bsize, 1, ha, wa)
        similarity_1 = F.interpolate(similarity_1, size=(q.size()[2], q.size()[3]),mode='bilinear', align_corners=True)

        similarity_2 = torch.bmm(tmp_test_mask, tmp_query) / (torch.bmm(tmp_tets_mask_norm, tmp_query_norm) + 1e-7)   #   4*3600*3600
        similarity_2 = similarity_2.max(1)[0].view(bsize, ha * ha)
        similarity_2 = (similarity_2 - similarity_2.min(1)[0].unsqueeze(1)) / (similarity_2.max(1)[0].unsqueeze(1) - similarity_2.min(1)[0].unsqueeze(1) + 1e-7)
        similarity_2 = similarity_2.view(bsize, 1, ha, wa)
        similarity_2 = F.interpolate(similarity_2, size=(q.size()[2], q.size()[3]),mode='bilinear', align_corners=True)

        p_v_s_mask = self.ChannelGate1(s_mask) * similarity_1
        p_v_mask_feat = self.ChannelGate2(mask_feat) * similarity_2

        # cosin = torch.cat([p_v_s_mask, p_v_mask_feat], 1)
        for i in range(len(p_v_mask_feat)):
            s_r = torch.where(p_v_s_mask[i] > 0, p_v_s_mask[i], p_v_mask_feat[i])
            p_v_s_mask[i] = s_r
        out=self.Conv(p_v_s_mask)
        return out

