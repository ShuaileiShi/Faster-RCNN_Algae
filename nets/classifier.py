import warnings

import torch
from torch import nn
from torchvision.ops import RoIPool

warnings.filterwarnings("ignore")

class Resnet50RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(Resnet50RoIHead, self).__init__()
        # classifier指向resnet50的后半部分，即第五次特征提取、平均池化、展平和全连接
        self.classifier = classifier
        # 对ROIPooling后的的结果进行回归预测————(num_classes)*4全连接用于对相应的建议框进行调整
        self.cls_loc = nn.Linear(2048, n_class * 4)
        # 对ROIPooling后的的结果进行分类————num_classes的全连接用于对最后获得的框进行分类
        self.score = nn.Linear(2048, n_class)
        # 权值初始化
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)
        # RoIPooling层将建议框映射到共享特征层上并截取，并将截取结果调整到固定大小
        self.roi = RoIPool((roi_size, roi_size), spatial_scale)
    
    def forward(self, x, rois, roi_indices, img_size):
        n, _, _, _ = x.shape
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()
        rois        = torch.flatten(rois, 0, 1)
        roi_indices = torch.flatten(roi_indices, 0, 1)
        
        rois_feature_map = torch.zeros_like(rois)
        rois_feature_map[:, [0,2]] = rois[:, [0,2]] / img_size[1] * x.size()[3]
        rois_feature_map[:, [1,3]] = rois[:, [1,3]] / img_size[0] * x.size()[2]

        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim=1)
        # 利用建议框对共享特征层进行截取————调用RoIPooling层实例
        pool = self.roi(x, indices_and_rois)
        # 利用classifier网络进行特征提取————调用classifier网络实例
        fc7 = self.classifier(pool)

        fc7 = fc7.view(fc7.size(0), -1)

        roi_cls_locs    = self.cls_loc(fc7)
        roi_scores      = self.score(fc7)
        roi_cls_locs    = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))
        roi_scores      = roi_scores.view(n, -1, roi_scores.size(1))
        return roi_cls_locs, roi_scores
    # 最终得到建议框的预测结果roi_scores(每个预测框对每个类，但数值未归一化)和回归参数roi_cls_locs(后续再进行解码生成预测框)

def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
