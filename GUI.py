import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
from tqdm import tqdm
from frcnn import FRCNN


# 创建主窗口
root = tk.Tk()
root.title("藻类目标检测")

# 设置窗口的大小
root.geometry("1280x720")
root.resizable(False, False)

# 显示加载图片的地址
describe_label = tk.Label(root, text="本项目是一个藻类目标检测系统，可以对藻类图像中的常见藻类进行识别和定位。", font=("Arial", 24))
describe_label.pack(pady=10)

# 用于展示图片的Label
image_display = tk.Label(root)
image_display.pack(pady=10)


# 选择单张图片的按钮
def select_image():
    # 打开图片选择对话框
    image_path = filedialog.askopenfilename(title="请选择图片", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
    if image_path:
        # 显示图片的路径
        describe_label.config(text=f"图片路径是: {image_path}")
        # 打开并显示图片
        image = Image.open(image_path)
        image = image.resize((600, 400))  # 调整图片大小
        photo = ImageTk.PhotoImage(image)
        image_display.config(image=photo)
        image_display.image = photo  # 保持引用，防止图片被垃圾回收
        return image_path
    else:
        print("图片文件错误！")
        describe_label.config(text=f"图片文件错误！")

# 选择图片文件夹的按钮
def select_folder():
    # 打开文件夹选择对话框
    folder_path = filedialog.askdirectory(title="请选择图片文件夹")
    if folder_path:
        describe_label.config(text=f"图片文件夹路径是：{folder_path}")
        return folder_path
    else:
        print("图片文件夹错误！")
        describe_label.config(text=f"图片文件夹错误！")

# 目标检测按钮
def run_image_predict():
    # 调用目标检测函数，返回处理后的图片并显示
    frcnn = FRCNN()
    image_path = select_image()
    image = Image.open(image_path)
    processed_image = frcnn.detect_image(image = image, crop = False, count = False)
    processed_image.show()
    processed_image = processed_image.resize((600, 400))  # 调整图片大小
    photo = ImageTk.PhotoImage(processed_image)
    image_display.config(image=photo)
    image_display.image = photo
    describe_label.config(text=f"预测结果")

def run_folder_predict():
    frcnn = FRCNN()
    folder_path = select_folder()
    img_names = os.listdir(folder_path)
    for img_name in tqdm(img_names):
        if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path = os.path.join(folder_path, img_name)
            image = Image.open(image_path)
            processed_image = frcnn.detect_image(image)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            processed_image.save(os.path.join(folder_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)
    print("预测完成！")

# 按钮1：运行单张图片的目标检测
run_predict_button = tk.Button(root, text="预测单张图片，请选择图片", command=run_image_predict, font=("Arial", 12))
run_predict_button.pack(pady=10)
# 按钮2：运行多张图片的目标检测
run_folder_predict_button = tk.Button(root, text="预测多张图片，请选择图片文件夹", command=run_folder_predict, font=("Arial", 12))
run_folder_predict_button.pack(pady=10)

# 运行主循环
root.mainloop()