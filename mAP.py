import os
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
from utils.utils import get_classes
from utils.utils_map import get_map
from frcnn import FRCNN

if __name__ == "__main__":
    # map_mode用于指定该文件运行时计算的内容：
    # map_mode为0代表整个计算流程，包括获得预测结果、获得真实框、计算VOC_map。
    # map_mode为1代表仅获得预测结果。
    # map_mode为2代表仅获得真实框。
    # map_mode为3代表仅计算VOC_map。
    map_mode = 3

    classes_path = 'data_model/Algae2024.txt'  #用于指定需要测量VOC_map的类别，一般情况下与训练和预测所用的classes_path一致即可
    VOCdevkit_path = 'VOCdevkit/Algae2024'  #用于指定VOC数据集所在的文件夹
    map_out_path = 'map'  #用于指定结果输出的文件夹

    map_vis = False  #用于指定是否开启VOC_map计算的可视化
    nms_iou = 0.5  #预测时使用到的非极大抑制值的大小，越大表示非极大抑制越不严格
    confidence = 0.02  #门限置信度设置得尽可能小进而获得全部可能的预测框，用于计算mAP，不可随意修改

    # MINOVERLAP用于指定想要获得的mAP0.x，mAP0.x的意义是在IoU阈值为0.x下的均值平均精度值
    # 比如计算mAP0.75，可以设定MINOVERLAP = 0.75
    # 当某一预测框与真实框重合度大于MINOVERLAP时，该预测框被认为是正样本，否则为负样本
    # 因此MINOVERLAP的值越大，预测框要预测的越准确才能被认为是正样本，此时算出来的mAP值越低，
    MINOVERLAP      = 0.5
    
    # 想要获得不同门限值下的Recall和Precision值，请修改score_threhold
    # Recall和Precision不像AP是一个面积的概念，因此在门限值不同时，网络的Recall和Precision值是不同的
    # 默认情况下，本代码计算的Recall和Precision值代表的是当门限值为0.5（此处定义为score_threhold）时所对应的值
    # 因为计算mAP需要获得近乎所有的预测框，上面定义的confidence不能随便更改
    # 这里专门定义一个score_threhold用于代表门限值，进而在计算mAP时找到门限值对应的Recall和Precision值
    score_threhold  = 0.5

    image_ids = open(os.path.join(VOCdevkit_path, "Sets/Main/test.txt")).read().strip().split()

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        frcnn = FRCNN(confidence = confidence, nms_iou = nms_iou)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "Images/"+image_id+".jpg")
            image       = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
            frcnn.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")
        
    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                root = ET.parse(os.path.join(VOCdevkit_path, "Annotations/"+image_id+".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult')!=None:
                        difficult = obj.find('difficult').text
                        if int(difficult)==1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox = obj.find('bndbox')
                    left   = bndbox.find('xmin').text
                    top    = bndbox.find('ymin').text
                    right  = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, False, score_threhold = score_threhold, path = map_out_path)
        print("Get map done.")
