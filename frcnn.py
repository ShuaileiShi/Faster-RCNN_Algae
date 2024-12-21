import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from nets.FasterRCNN import FasterRCNN
from utils.utils import (cvtColor, get_classes, get_new_img_size, resize_image,preprocess_input, show_config)
from utils.utils_bbox import DecodeBox



class FRCNN(object):
    _defaults = {
        # 进行训练或预测前一定要修改model_path和classes_path！
        # 训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        # 验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        # 如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        "model_path"    : 'logs\ep100-loss0.261-val_loss0.300.pth',  #指向logs文件夹下的训练权值文件
        "classes_path"  : 'data_model/Algae2024.txt',  #指向data_model下的类别txt
        "backbone" : "resnet50",  #主干特征提取网络
        "confidence" : 0.5,  #预测框信度
        "nms_iou" : 0.3,  #非极大抑制所用的交并比
        'anchors_size' : [4, 16, 32],  #先验框的面积大小比例
        "cuda" : True,  #是否使用Cuda
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"


    # 初始化Faster RCNN
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 
        # 获得种类和先验框的数量
        self.class_names, self.num_classes  = get_classes(self.classes_path)

        self.std = torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(self.num_classes + 1)[None]
        if self.cuda:
            self.std = self.std.cuda()
        self.bbox_util = DecodeBox(self.std, self.num_classes)

        # 画框设置不同的颜色
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

        show_config(**self._defaults)
    
    # 载入模型与权值
    def generate(self):
        self.net = FasterRCNN(self.num_classes, "predict", anchor_scales = self.anchors_size, backbone = self.backbone)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        
        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()


    # 检测图片
    def detect_image(self, image, crop = False, count = False):
        # 计算输入图片的高和宽
        image_shape = np.array(np.shape(image)[0:2])
        # 预resize后的图片短边为600，计算预resize后的图片长边大小
        input_shape = get_new_img_size(image_shape[0], image_shape[1])
        # 将所有其它类型的图像转换成RGB图像
        image       = cvtColor(image)
        # 给原图像进行resize
        image_data  = resize_image(image, [input_shape[1], input_shape[0]])
        # 添加batch_size维度
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        # 将图片转换成tensor形式
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # roi_cls_locs  建议框的调整参数
            # roi_scores    建议框的种类得分
            # rois          建议框的坐标
            roi_cls_locs, roi_scores, rois, _ = self.net(images)
            # 利用classifier的预测结果对建议框进行解码，获得预测框
            results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape, nms_iou = self.nms_iou, confidence = self.confidence)
            # 如果没有检测出物体，返回原图           
            if len(results[0]) <= 0:
                return image
                
            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]
        
        # 设置字体与边框厚度
        font      = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(input_shape), 1))
        # 计数
        if count:
            print("top_label:", top_label)
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        # 是否进行目标的裁剪
        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]
                top    = max(0, np.floor(top).astype('int32'))
                left   = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right  = min(image.size[0], np.floor(right).astype('int32'))
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        # 图像绘制
        for i, c in enumerate(top_label):
            predicted_class = self.class_names[int(c)]
            box   = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top    = max(0, np.floor(top).astype('int32'))
            left   = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right  = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textbbox((10, 10), label, font=font) # 如果使用的是python3.8及对应版本的pillow，这里的textbbox方法需要更改为textsize方法
            label_width = label_size[2] - label_size[0]
            label_height = label_size[3] - label_size[1]

            text_origin = np.array([left, top - label_height]) if top - label_height >= 0 else np.array([left, top + 1])

            # 绘制矩形框
            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            # 绘制标签背景框
            draw.rectangle([tuple(text_origin), tuple(text_origin + [label_width, label_height])], fill=self.colors[c])
            # 绘制标签文字
            draw.text(tuple(text_origin), label, fill=(0, 0, 0), font=font)
            del draw

        return image
    
    def statistic(self, dir_origin_path, dir_save_path, crop = False, count = True):
        dir_classes_nums = np.zeros([self.num_classes], dtype = int)
        
        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                # 计算输入图片的高和宽
                image_shape = np.array(np.shape(image)[0:2])
                # 预resize后的图片短边为600，计算预resize后的图片长边大小
                input_shape = get_new_img_size(image_shape[0], image_shape[1])
                # 将所有其它类型的图像转换成RGB图像
                image       = cvtColor(image)
                # 给原图像进行resize
                image_data  = resize_image(image, [input_shape[1], input_shape[0]])
                # 添加batch_size维度
                image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

                # 将图片转换成tensor形式
                with torch.no_grad():
                    images = torch.from_numpy(image_data)
                    if self.cuda:
                        images = images.cuda()
                    # roi_cls_locs  建议框的调整参数
                    # roi_scores    建议框的种类得分
                    # rois          建议框的坐标
                    roi_cls_locs, roi_scores, rois, _ = self.net(images)
                    # 利用classifier的预测结果对建议框进行解码，获得预测框
                    results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape, nms_iou = self.nms_iou, confidence = self.confidence)
                    # 如果没有检测出物体，返回原图           
                    if len(results[0]) <= 0:
                        continue
                        
                    top_label   = np.array(results[0][:, 5], dtype = 'int32')
                    top_conf    = results[0][:, 4]
                    top_boxes   = results[0][:, :4]
            
                # 设置字体与边框厚度
                font      = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
                thickness = int(max((image.size[0] + image.size[1]) // np.mean(input_shape), 1))

                # 计数
                if count:
                    for i in range(self.num_classes):
                        num = np.sum(top_label == i)
                        dir_classes_nums[i] += num
                
                # 是否进行目标的裁剪
                if crop:
                    for i, c in list(enumerate(top_label)):
                        top, left, bottom, right = top_boxes[i]
                        top    = max(0, np.floor(top).astype('int32'))
                        left   = max(0, np.floor(left).astype('int32'))
                        bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                        right  = min(image.size[0], np.floor(right).astype('int32'))
                        
                        dir_save_path = "img_crop"
                        if not os.path.exists(dir_save_path):
                            os.makedirs(dir_save_path)
                        crop_image = image.crop([left, top, right, bottom])
                        crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                        print("save crop_" + str(i) + ".png to " + dir_save_path)

                # 图像绘制
                for i, c in enumerate(top_label):
                    predicted_class = self.class_names[int(c)]
                    box   = top_boxes[i]
                    score = top_conf[i]

                    top, left, bottom, right = box

                    top    = max(0, np.floor(top).astype('int32'))
                    left   = max(0, np.floor(left).astype('int32'))
                    bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                    right  = min(image.size[0], np.floor(right).astype('int32'))

                    label = '{} {:.2f}'.format(predicted_class, score)
                    draw = ImageDraw.Draw(image)
                    label_size = draw.textbbox((10, 10), label, font=font) # 如果使用的是python3.8及对应版本的pillow，这里的textbbox方法需要更改为textsize方法
                    label_width = label_size[2] - label_size[0]
                    label_height = label_size[3] - label_size[1]

                    text_origin = np.array([left, top - label_height]) if top - label_height >= 0 else np.array([left, top + 1])

                    # 绘制矩形框
                    for i in range(thickness):
                        draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
                    # 绘制标签背景框
                    draw.rectangle([tuple(text_origin), tuple(text_origin + [label_width, label_height])], fill=self.colors[c])
                    # 绘制标签文字
                    draw.text(tuple(text_origin), label, fill=(0, 0, 0), font=font)
                    del draw

                    if not os.path.exists(dir_save_path):
                        os.makedirs(dir_save_path)
                    image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

        return dir_classes_nums

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        input_shape = get_new_img_size(image_shape[0], image_shape[1])
        image       = cvtColor(image)
        image_data  = resize_image(image, [input_shape[1], input_shape[0]])
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            roi_cls_locs, roi_scores, rois, _ = self.net(images)
            results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape, nms_iou = self.nms_iou, confidence = self.confidence)
        
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                roi_cls_locs, roi_scores, rois, _ = self.net(images)
                results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape, nms_iou = self.nms_iou, confidence = self.confidence) 
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time
    
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w")
        image_shape = np.array(np.shape(image)[0:2])
        input_shape = get_new_img_size(image_shape[0], image_shape[1])
        image       = cvtColor(image)
        image_data  = resize_image(image, [input_shape[1], input_shape[0]])
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            roi_cls_locs, roi_scores, rois, _ = self.net(images)
            results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape, nms_iou = self.nms_iou, confidence = self.confidence)
            if len(results[0]) <= 0:
                return
            
            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]
        
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])
            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue
            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))
        f.close()
        return