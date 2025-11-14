import torch
from torch import nn
import torch.nn.functional as F
from model.Transformer import Transformer
import model.resnet as models
from einops import rearrange
import clip
import math
from model.get_cam import get_img_cam
from pytorch_grad_cam import GradCAM
from clip.clip_text import new_class_names, new_class_names_coco

import model.resnet as models
import model.vgg as vgg_models
from model.ASPP import ASPP
from model.cosine import FEM

def zeroshot_classifier(classnames, templates, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).to(device) #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights.t()

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


def get_similarity(q, s, mask):
    if len(mask.shape) == 3:
        mask = mask.unsqueeze(1)
    mask = F.interpolate((mask == 1).float(), q.shape[-2:])
    cosine_eps = 1e-7
    s = s * mask
    bsize, ch_sz, sp_sz, _ = q.size()[:]
    tmp_query = q
    tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
    tmp_query_norm = torch.norm(tmp_query, 2, 1, True)
    tmp_supp = s
    tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1).contiguous()
    tmp_supp = tmp_supp.contiguous().permute(0, 2, 1).contiguous()
    tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)
    similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
    similarity = similarity.max(1)[0].view(bsize, sp_sz * sp_sz)
    similarity = similarity.view(bsize, 1, sp_sz, sp_sz)
    return similarity


def get_gram_matrix(fea):
    b, c, h, w = fea.shape
    fea = fea.reshape(b, c, h * w)  # C*N
    fea_T = fea.permute(0, 2, 1)  # N*C
    fea_norm = fea.norm(2, 2, True)
    fea_T_norm = fea_T.norm(2, 1, True)
    gram = torch.bmm(fea, fea_T) / (torch.bmm(fea_norm, fea_T_norm) + 1e-7)  # C*C
    return gram


def get_vgg16_layer(model):
    layer0_idx = range(0, 7)
    layer1_idx = range(7, 14)
    layer2_idx = range(14, 24)
    layer3_idx = range(24, 34)
    layer4_idx = range(34, 43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]
    layer0 = nn.Sequential(*layers_0)
    layer1 = nn.Sequential(*layers_1)
    layer2 = nn.Sequential(*layers_2)
    layer3 = nn.Sequential(*layers_3)
    layer4 = nn.Sequential(*layers_4)
    return layer0, layer1, layer2, layer3, layer4


def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False):
        super(BaseConv, self).__init__()
        pad         = (ksize - 1) // 2
        self.conv   = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn     = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act    = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13)):
        super(SPPBottleneck, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1      = BaseConv(in_channels, hidden_channels, 1, stride=1)
        self.m          = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        conv2_channels  = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2      = BaseConv(conv2_channels, out_channels, 1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x
# ######################## Focus Module ###########################
def SA_Weighted_GAP(supp_feat, mask, supp_pred_mask):
    supp_pred = supp_pred_mask+mask
    new_mask1 = torch.zeros_like(mask)
    new_mask2 = torch.zeros_like(mask)

    new_mask1[supp_pred==2] = 1
    new_mask2[supp_pred==1] = 1

    new_mask1[mask==0] = 0
    new_mask2[mask==0] = 0

    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    new_area1 = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    new_area2 = F.avg_pool2d(new_mask2, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat1 = supp_feat * mask
    supp_feat1 = F.avg_pool2d(input=supp_feat1, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / new_area1
    supp_feat2 = supp_feat * new_mask2
    supp_feat2 = F.avg_pool2d(input=supp_feat2, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / new_area2
    return supp_feat1, supp_feat2


class Focus(nn.Module):
    def __init__(self, channel1, channel2):
        super(Focus, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        self.output_map = nn.Conv2d(self.channel1, 2, 7, 1, 3)
        self.aux_1 = SPPBottleneck(self.channel1,self.channel1)
        self.aux_2 = SPPBottleneck(self.channel1,self.channel1)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gama = nn.Parameter(torch.ones(1))
        self.bn1 = nn.BatchNorm2d(self.channel1)
        self.relu1 = nn.ReLU()

        self.bn2 = nn.BatchNorm2d(self.channel1)
        self.relu2 = nn.ReLU()
        self.input_map=nn.Sigmoid()

    def forward(self, x, y,in_map):
        in_put = torch.argmax(in_map, dim=1, keepdim=True)
        f_feature = x * in_put
        b_feature = x * (1 - in_put)
        fp = self.aux_1(f_feature)
        fn = self.aux_2(b_feature)
        refine1 = y - (self.alpha * fp)
        refine1 = self.bn1(refine1)
        refine1 = self.relu1(refine1)
        refine2 = refine1 + (self.beta * fn)
        refine2 = self.bn2(refine2)
        refine2 = self.relu2(refine2)
        output_map = self.output_map(refine2)
        return refine2, output_map


class OneModel(nn.Module):
    def __init__(self, args, cls_type=None,pretrained=True,layers=50):
        super(OneModel, self).__init__()

        self.cls_type = cls_type  # 'Base' or 'Novel'
        self.dataset = args.data_set
        if self.dataset == 'pascal':
            self.base_classes = 20
        elif self.dataset == 'coco':
            self.base_classes = 60
        self.low_fea_id = args.low_fea[-1]

        assert args.layers in [18,34,50, 101, 152]
        from torch.nn import BatchNorm2d as BatchNorm
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.shot = args.shot
        self.vgg = args.vgg
        models.BatchNorm = BatchNorm

        if self.vgg:
            print('INFO: Using VGG_16 bn')
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=pretrained)
            print(vgg16)
            self.layer0, self.layer1, self.layer2, \
                self.layer3, self.layer4 = get_vgg16_layer(vgg16)

        else:
            print('INFO: Using ResNet {}'.format(layers))
            if layers == 50:
                resnet = models.resnet50(pretrained=pretrained)
            elif layers == 101:
                resnet = models.resnet101(pretrained=pretrained)
            else:
                resnet = models.resnet152(pretrained=pretrained)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2, resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4   #self.layer0 ，self.layer1, self.layer2, self.layer3, self.layer4

            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512
        #
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, 256, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, 256, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.down_mask = nn.Sequential(
            nn.Conv2d(fea_dim, 256, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        if self.shot==1:
            channel = 514
        else:
            channel = 524
        self.query_merge = nn.Sequential(
            nn.Conv2d(channel, 256, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.query_merge1 = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )

        self.annotation_root = args.annotation_root
        self.clip_model, _ = clip.load(args.clip_path)
        if self.dataset == 'pascal':
            self.bg_text_features = zeroshot_classifier(new_class_names, ['a photo without {},a type of aircraft skin defects..'],
                                                        self.clip_model)
            self.fg_text_features = zeroshot_classifier(new_class_names, ['a photo of {},a type of aircraft skin defects.'],
                                                        self.clip_model)
        elif self.dataset == 'coco':
            self.bg_text_features = zeroshot_classifier(new_class_names_coco, ['a photo without {}.'],
                                                        self.clip_model)
            self.fg_text_features = zeroshot_classifier(new_class_names_coco, ['a photo of {}.'],
                                                        self.clip_model)
        reduce_dim = 256
        classes=2

        self.ASPP_meta = ASPP(reduce_dim)
        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True))

        self.res1_1 = nn.Sequential(
            nn.Conv2d(reduce_dim * 5, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        self.res2_1 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True))
        # self.transformer = Transformer(shot=self.shot)

        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, classes, kernel_size=1)
        )

        self.cls_1 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, classes, kernel_size=1)
        )

        self.focus1 = Focus(256, 256)
        self.focus2 = Focus(256, 256)
        self.focus3 = Focus(256, 256)
        self.FEM   =FEM(256)
        self.cr4 = nn.Sequential(nn.Conv2d(2, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())



    def forward(self, x, x_cv2, que_name, class_name, y_m=None, y_b=None, s_x=None, s_y=None, cat_idx=None):
        mask = rearrange(s_y, "b n h w -> (b n) 1 h w")
        mask = (mask == 1).float()
        h, w = x.shape[-2:]

        # s_x = rearrange(s_x, "b n c h w -> (b n) c h w")

        with torch.no_grad():
            # x=torch.Size([4, 3, 473, 473])
            query_feat_0 = self.layer0(x)  # torch.Size([4, 128, 119, 119])
            query_feat_1 = self.layer1(query_feat_0)  # torch.Size([4, 256, 119, 119])
            query_feat_2 = self.layer2(query_feat_1)  # torch.Size([4, 512, 60, 60])
            query_feat_3 = self.layer3(query_feat_2)  # torch.Size([4, 1024, 60, 60])
            query_feat_4 = self.layer4(query_feat_3)  # torch.Size([4, 2048, 60, 60])

            if self.vgg:
                query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2), query_feat_3.size(3)),
                                             mode='bilinear', align_corners=True)

        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat_cnn = self.down_query(query_feat)  #

        supp_feat_list = []
        final_supp_list = []
        mask_list = []
        mask_img_list=[]
        for i in range(self.shot):
            mask = (s_y[:, i, :, :] == 1).float().unsqueeze(1)
            mask_list.append(mask)
            img = s_x[:, i, :, :, :]
            masked_img = img * mask
            mask_img_list.append(masked_img)
        mask_img2 = torch.cat(mask_img_list, dim=0)
        b, shot, c, h, w, = s_x.shape
        mask_img2 = rearrange(mask_img2, "(b n) c h w -> b n c h w", b=b, n=shot)
        mask_img2 = mask_img2.mean(dim=1)

        for i in range(self.shot):
            #--------------------------------------------------------------------------#
            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:, i, :, :, :])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear',
                                     align_corners=True)
                supp_feat_4 = self.layer4(supp_feat_3 * mask)  # XS
                final_supp_list.append(supp_feat_4)
                if self.vgg:
                    supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2), supp_feat_3.size(3)),
                                                mode='bilinear', align_corners=True)
                # --------------------------------------------------------------------------#
                mask_feat_0 = self.layer0(mask_img2)
                mask_feat_1 = self.layer1(mask_feat_0)
                mask_feat_2 = self.layer2(mask_feat_1)
                mask_feat_3 = self.layer3(mask_feat_2)
                mask_feat_4 = self.layer4(mask_feat_3)
                if self.vgg:
                    query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2), query_feat_3.size(3)),
                                                 mode='bilinear', align_corners=True)
            # --------------------------------------------------------------------------#

        supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
        supp_feat_cnn = self.down_supp(supp_feat)
        supp_feat = Weighted_GAP(supp_feat, mask)
        supp_feat_list.append(supp_feat)
        # --------------------------------------------------------------------------#
        supp_test_mask = Weighted_GAP(supp_feat_cnn,F.interpolate(mask, size=(supp_feat_cnn.size(2), supp_feat_cnn.size(3)),
                                              mode='bilinear', align_corners=True))
        supp_test_feat = supp_test_mask.repeat(1, 1, supp_feat_cnn.shape[-2], supp_feat_cnn.shape[-1])
        # print(supp_test_feat.shape)
        mask_feat = torch.cat([mask_feat_3, mask_feat_2], 1)
        mask_feat = self.down_mask(mask_feat)

        cosin_feat=self.FEM(query_feat_cnn,supp_test_feat,mask_feat)

        target_layers = [self.clip_model.visual.transformer.resblocks[-1].ln_1]
        cam = GradCAM(model=self.clip_model, target_layers=target_layers, reshape_transform=reshape_transform)
        img_cam_list = get_img_cam(x_cv2, que_name, class_name, self.clip_model, self.bg_text_features, self.fg_text_features, cam, self.annotation_root, self.training)
        img_cam_list = [F.interpolate(t_img_cam.unsqueeze(0).unsqueeze(0), size=(supp_feat_cnn.shape[2], supp_feat_cnn.shape[3]), mode='bilinear',
                                      align_corners=True) for t_img_cam in img_cam_list]
        img_cam = torch.cat(img_cam_list, 0)
        img_cam = img_cam.repeat(1,2,1,1)
        query_feat = self.query_merge(torch.cat([query_feat_cnn,cosin_feat, img_cam * 10], dim=1))   #1*64*60*60

        meta_out = self.res1(query_feat)   # 1080->256
        meta_out = self.res2(meta_out) + meta_out
        out = self.cls(meta_out)

        out_1 = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

        a4=img_cam.clone().detach()
        a4=self.cr4(a4)

        refine2, out_2 = self.focus1(a4,  meta_out, out)
        refine3, out_3 = self.focus2(a4, refine2,out_2)
        refine4, out_4 = self.focus3(a4, refine3,out_3)
        out_2 = F.interpolate(out_2, size=(h, w), mode='bilinear', align_corners=True)
        out_3 = F.interpolate(out_3, size=(h, w), mode='bilinear', align_corners=True)
        out_4 = F.interpolate(out_4, size=(h, w), mode='bilinear', align_corners=True)

        meta_out_2 = self.ASPP_meta(refine4)
        meta_out_2 = self.res1_1(meta_out_2)  # 1080->256
        meta_out_2 = self.res2_1(meta_out_2) + meta_out_2
        final_out = self.cls_1(meta_out_2)

        final_out = F.interpolate(final_out, size=(h, w), mode='bilinear', align_corners=True)
        # Loss
        if self.training:
            main_loss = self.criterion(final_out, y_m.long())
            aux_loss1 = self.criterion(out_1, y_m.long())
            aux_loss2 = self.criterion(out_2, y_m.long())
            aux_loss3 = self.criterion(out_3, y_m.long())
            aux_loss4 = self.criterion(out_4, y_m.long())

            return final_out.max(1)[1], main_loss , (aux_loss1+aux_loss2+aux_loss3+aux_loss4)/4
        else:
            return final_out, out


    def get_optim(self, model, args, LR):
        optimizer = torch.optim.AdamW(
            [
                {'params': model.query_merge.parameters(), "lr": LR * 10},
                {'params': model.down_supp.parameters(), "lr": LR * 10},
                {'params': model.down_query.parameters(), "lr": LR * 10},
                {'params': model.cr4.parameters()},
                {'params': model.res1_1.parameters()},
                {'params': model.res2_1.parameters()},
                {'params': model.cls_1.parameters()},
                {'params': model.focus1.parameters()},
                {'params': model.focus2.parameters()},
                {'params': model.focus3.parameters()},
                {'params': model.FEM.parameters(), "lr": LR * 5},
                {'params': model.res1.parameters()},
                {'params': model.res2.parameters()},
                {'params': model.cls.parameters()}
            ], lr=LR, weight_decay=args.weight_decay, betas=(0.9, 0.999))
        return optimizer
