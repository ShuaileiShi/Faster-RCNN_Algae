import os
import xml.etree.ElementTree as ET

# Annotations文件夹的路径
annotations_folder = 'VOCdevkit\PMID_2019\Annotations - 副本'
# 更改前的标签名
label1 = 'ceratium furca'
# 更改后的标签名
label2 = 'Ceratium'

# 遍历Annotations文件夹中的所有XML文件
for filename in os.listdir(annotations_folder):
    if filename.endswith('.xml'):
        file_path = os.path.join(annotations_folder, filename)
        try:
            # 解析XML文件
            tree = ET.parse(file_path)
            root = tree.getroot()

            # 检查每个<object>标签
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name == label1:
                    obj.find('name').text = label2  # 更新名称
                    tree.write(file_path)  # 保存修改后的XML文件

        except Exception as e:
            print(f"处理文件 {filename} 时发生错误: {e}")

print("XML文件更新完成。")
