
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms
from utils.anchors import _enumerate_shifted_anchor, generate_anchor_base
from utils.utils_bbox import loc2bbox


class ProposalCreator():
    def __init__(
        self, 
        mode, 
        nms_iou          = 0.7,  #IoU(交并比)：IoU_A=(A&B)/(A+B)
        n_train_pre_nms  = 12000,  #训练/测试时最初生成和最终保留的建议框数量，可适当调整，下同
        n_train_post_nms = 600,
        n_test_pre_nms   = 3000,
        n_test_post_nms  = 300,
        min_size         = 16  #建议框的宽高的最小值
    ):
        self.mode             = mode
        self.nms_iou          = nms_iou
        self.n_train_pre_nms  = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms   = n_test_pre_nms
        self.n_test_post_nms  = n_test_post_nms
        self.min_size         = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.):
        if self.mode == "training":
            n_pre_nms  = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms  = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        # 将先验框转换成tensor
        anchor = torch.from_numpy(anchor).type_as(loc)
        # 将RPN网络预测结果转化成建议框————loc2bbox函数此处是将先验框转化成建议框，主要过程是对先验框的大小根据先验框调整参数进行调整；详见untils/untils_bbox.py
        roi = loc2bbox(anchor, loc)
       
        # 防止建议框超出图像边缘
        roi[:, [0, 2]] = torch.clamp(roi[:, [0, 2]], min = 0, max = img_size[1])
        roi[:, [1, 3]] = torch.clamp(roi[:, [1, 3]], min = 0, max = img_size[0])
        # 建议框的宽高的最小值不可以小于16
        min_size = self.min_size * scale
        keep = torch.where(((roi[:, 2] - roi[:, 0]) >= min_size) & ((roi[:, 3] - roi[:, 1]) >= min_size))[0]
        # 将对应的建议框保留下来
        roi = roi[keep, :]
        score = score[keep]

        # 根据得分进行排序，取出建议框————对建议框根据先验框分类参数(前景)进行降序，取出前train_pre_nms/test_pre_nms个建议框
        order = torch.argsort(score, descending=True)
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]

        # 对建议框进行非极大抑制————nms()方法，在一定区域内只保留其中先验框分类参数(前景)极大的一个建议框，防止建议框过多、重叠；最终保留n_train_post_nms/n_test_post_nms个建议框
        keep    = nms(roi, score, self.nms_iou)
        if len(keep) < n_post_nms:
            index_extra = np.random.choice(range(len(keep)), size=(n_post_nms - len(keep)), replace=True)
            keep = torch.cat([keep, keep[index_extra]])
        keep = keep[:n_post_nms]
        roi = roi[keep]
        return roi


class RegionProposalNetwork(nn.Module):
    def __init__(
        self, 
        in_channels   = 512, 
        mid_channels  = 512, 
        ratios        = [0.5, 1, 2],
        anchor_scales = [8, 16, 32], 
        feat_stride   = 16,
        mode          = "training",
    ):
        super(RegionProposalNetwork, self).__init__()
        # 生成基础先验框，shape为[9, 4]
        self.anchor_base = generate_anchor_base(anchor_scales = anchor_scales, ratios = ratios)
        n_anchor = self.anchor_base.shape[0]

        # 先进行一个3x3的卷积，可理解为特征整合
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        # 分类预测先验框内部是否包含物体————进行1*1的卷积，生成18个[先验框个数*2(先验框中物体为前景、背景的概率)]先验框分类参数————也有将此处的“先验框”替换为“建议框”的说法，但都是一样的，下同
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        # 回归预测对先验框进行调整————进行1*1的卷积，生成36个[基础先验框个数*4(先验框的坐标调整参数，包括中心点xy坐标，高宽hw)]先验框调整参数，亦称回归参数
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        # 特征点间距步长
        self.feat_stride    = feat_stride
        # 用于对建议框解码并进行非极大抑制————由先验框获得建议框，并进行非极大抑制
        self.proposal_layer = ProposalCreator(mode)
        # 对FPN的网络部分进行权值初始化
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        n, _, h, w = x.shape
        # 先进行一个3x3的卷积，可理解为特征整合————调用实例
        x = F.relu(self.conv1(x))
        # 回归预测对先验框进行调整————调用实例
        rpn_locs = self.loc(x)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        # 分类预测先验框内部是否包含物体————调用实例
        rpn_scores = self.score(x)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)
        
        # 进行softmax概率计算，每个先验框只有两个判别结果————“回归预测对先验框进行调整————调用实例”这一步的数学计算
        # 内部包含物体或者内部不包含物体，rpn_softmax_scores[:, :, 1]的内容为包含物体的概率
        rpn_softmax_scores  = F.softmax(rpn_scores, dim=-1)
        rpn_fg_scores       = rpn_softmax_scores[:, :, 1].contiguous()
        rpn_fg_scores       = rpn_fg_scores.view(n, -1)

        # 生成先验框，此时获得的anchor是布满网格点的，当输入图片为600,600,3的时候，shape为(12996, 4)
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, h, w)
        rois        = list()
        roi_indices = list()
        for i in range(n):
            roi         = self.proposal_layer(rpn_locs[i], rpn_fg_scores[i], anchor, img_size, scale = scale)
            batch_index = i * torch.ones((len(roi),))
            rois.append(roi.unsqueeze(0))
            roi_indices.append(batch_index.unsqueeze(0))

        rois        = torch.cat(rois, dim=0).type_as(x)
        roi_indices = torch.cat(roi_indices, dim=0).type_as(x)
        anchor      = torch.from_numpy(anchor).unsqueeze(0).float().to(x.device)
        
        return rpn_locs, rpn_scores, rois, roi_indices, anchor

def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
