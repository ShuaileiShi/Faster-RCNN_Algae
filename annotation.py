import os
import random
import xml.etree.ElementTree as ET
import numpy as np
from utils.utils import get_classes


# annotation_mode用于指定该文件运行时计算的内容
# annotation_mode为0代表整个标签处理过程
# annotation_mode为1代表获得VOCdevkit/XXXX/Sets里面的txt
# annotation_mode为2代表获得训练用的data_image中的_train.txt、_val.txt
annotation_mode = 0

# 必须要修改，用于生成_train.txt、_val.txt的目标信息，与训练和预测所用的classes_path一致
# 仅在annotation_mode为0和2的时候有效
# classes_path = 'data_model/BJFUHJZooPlankton.txt'
classes_path = 'data_model/Algae2024.txt'

# trainval_percent用于指定(训练集+验证集)与测试集的比例
# train_percent用于指定(训练集+验证集)中训练集与验证集的比例
# 仅在annotation_mode为0和1的时候有效
trainval_percent = 1
train_percent    = 0.01

# 指向VOC数据集所在的文件夹
VOCdevkit_path = 'VOCdevkit/test'  # 使用不同VOC数据集请修改此处路径
VOCdevkit_sets = [('test', 'train'), ('test', 'val')]  # 使用不同VOC数据集请修改此处编号
classes, _     = get_classes(classes_path)


photo_nums  = np.zeros(len(VOCdevkit_sets))
nums        = np.zeros(len(classes))
def convert_annotation(ref, image_id, list_file):
    in_file = open(os.path.join(VOCdevkit_path, 'Annotations/%s.xml'%(image_id)), encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),\
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        
        nums[classes.index(cls)] = nums[classes.index(cls)] + 1
        
if __name__ == "__main__":
    random.seed(0)
    if " " in os.path.abspath(VOCdevkit_path):
        raise ValueError("数据集存放的文件夹路径与图片名称中不可以存在空格，否则会影响正常的模型训练，请注意修改。")

    if annotation_mode == 0 or annotation_mode == 1:
        xmlfilepath     = os.path.join(VOCdevkit_path, 'Annotations')
        saveBasePath    = os.path.join(VOCdevkit_path, 'Sets')
        temp_xml        = os.listdir(xmlfilepath)
        total_xml       = []
        for xml in temp_xml:
            if xml.endswith(".xml"):
                total_xml.append(xml)

        num      = len(total_xml)  
        list     = range(num)  
        tv       = int(num*trainval_percent)  
        tr       = int(tv*train_percent)  
        trainval = random.sample(list,tv)  
        train    = random.sample(trainval,tr)  
        
        print("train and val size",tv)
        print("train size",tr)
        print("val size",tv-tr)
        ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
        ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
        ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
        fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
        
        for i in list:  
            name=total_xml[i][:-4]+'\n'  
            if i in trainval:  
                ftrainval.write(name)  
                if i in train:  
                    ftrain.write(name)  
                else:  
                    fval.write(name)  
            else:  
                ftest.write(name)  
        
        ftrainval.close()  
        ftrain.close()  
        fval.close()  
        ftest.close()
        print("已在Sets生成trainval.txt")

    if annotation_mode == 0 or annotation_mode == 2:
        type_index = 0
        for ref, image_set in VOCdevkit_sets:
            image_ids = open(os.path.join(VOCdevkit_path, 'Sets/%s.txt'%(image_set)), encoding='utf-8').read().strip().split()
            list_file = open('%s_%s.txt'%(ref, image_set), 'w', encoding='utf-8')
            for image_id in image_ids:
                list_file.write('%s/Images/%s.jpg'%(os.path.abspath(VOCdevkit_path), image_id))

                convert_annotation(ref, image_id, list_file)
                list_file.write('\n')
            photo_nums[type_index] = len(image_ids)
            type_index += 1
            list_file.close()
        print("已生成_train.txt和_val.txt")
        
        def printTable(List1, List2):
            for i in range(len(List1[0])):
                print("|", end=' ')
                for j in range(len(List1)):
                    print(List1[j][i].rjust(int(List2[j])), end=' ')
                    print("|", end=' ')
                print()

        str_nums = [str(int(x)) for x in nums]
        tableData = [
            classes, str_nums
        ]
        colWidths = [0]*len(tableData)
        len1 = 0
        for i in range(len(tableData)):
            for j in range(len(tableData[i])):
                if len(tableData[i][j]) > colWidths[i]:
                    colWidths[i] = len(tableData[i][j])
        printTable(tableData, colWidths)

        if photo_nums[0] <= 500:
            print("训练集数量小于500，属于较小的数据量，请注意设置较大的训练世代（Epoch）以满足足够的梯度下降次数（Step）。")

        if np.sum(nums) == 0:
            print("在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")