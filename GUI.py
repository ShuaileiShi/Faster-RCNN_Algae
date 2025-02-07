import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy
import os
import pandas
import threading
from tqdm import tqdm
from PIL import Image, ImageTk

from frcnn import FRCNN


# 单张预测窗口
def open_image_window():
    # 选择单张图片
    def select_image():
        image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        if image_path:
            select_label.config(text=f"选择的图片路径: {image_path}")
            image = Image.open(image_path)
            image.thumbnail((600, 400))  # 调整图片大小
            photo = ImageTk.PhotoImage(image)
            image_display_label.config(image=photo)
            image_display_label.image = photo  # 保存引用，避免被垃圾回收

    # 运行预测
    def predict_image():
        image_path = select_label.cget("text").replace("选择的图片路径: ", "")
        if image_path:  
            predict_button.config(state=tk.DISABLED)  # 禁用预测按钮，避免重复点击
            progress_bar = ttk.Progressbar(new_window, orient="horizontal", length=400, mode="indeterminate")  # 创建进度条
            progress_bar.place(x=200, y=500)
            progress_bar.start()
            # 使用多线程运行模型
            def run_model():
                frcnn = FRCNN()
                image = Image.open(image_path)
                processed_image = frcnn.detect_image(image=image, crop=False, count=False)
                processed_image.show()
                processed_image = processed_image.resize((600, 400))  # 调整图片大小
                photo = ImageTk.PhotoImage(processed_image)
                image_display_label.config(image=photo)
                image_display_label.image = photo  # 保存引用，避免被垃圾回收

                progress_bar.stop()  # 停止并关闭进度条
                progress_bar.destroy()
                predict_button.config(state=tk.NORMAL)  # 重新启用预测按钮

            threading.Thread(target=run_model).start()
        else:
            messagebox.showerror("错误", "请先选择图片！")

    # 返回主窗口
    def return_to_root():
        new_window.destroy()
        root.deiconify()

    # 创建新窗口
    new_window = tk.Toplevel(root)
    new_window.title("单张预测")
    new_window.geometry("800x600")
    '''new_window.attributes('-topmost', True)'''

    select_button = tk.Button(new_window, text="选择图片", font=("宋体", 16), command=select_image)
    select_button.place(x=375, y=10)
    select_label = tk.Label(new_window, text="选择的图片路径: ", font=("宋体", 12))
    select_label.place(x=15, y=50)

    image_display_label = tk.Label(new_window)
    image_display_label.place(x=120, y=100)

    predict_button = tk.Button(new_window, text="开始预测", font=("宋体", 16), command=predict_image)
    predict_button.place(x=375, y=550)

    return_button = tk.Button(new_window,text="返回",font=("宋体", 16),command=return_to_root)
    return_button.place(x=10,y=10)
    
    # 绑定子窗口关闭事件
    new_window.protocol("WM_DELETE_WINDOW", return_to_root)


# 多张预测窗口
def open_folder_window():
    # 选择预测文件夹
    def select_folder():
        select_path = filedialog.askdirectory()
        if select_path:
            select_label.config(text=f"预测文件夹路径: {select_path}")
    
    # 选择存储文件夹
    def save_folder():
        save_path = filedialog.askdirectory()
        if save_path:
            save_label.config(text=f"存储文件夹路径: {save_path}")

    # 运行预测
    def predict_folder():
        select_path = select_label.cget("text").replace("预测文件夹路径: ", "")
        save_path = save_label.cget("text").replace("存储文件夹路径: ", "")
        if select_path and save_path:
            predict_button.config(state=tk.DISABLED)  # 禁用预测按钮，避免重复点击
            progress_bar = ttk.Progressbar(new_window, orient="horizontal", length=600, mode="determinate")  # 创建进度条
            progress_bar.place(x=100, y=400)
            progress_bar["value"] = 0  # 初始进度为0

            # 使用多线程运行模型
            def run_model():
                frcnn = FRCNN()
                img_names = os.listdir(select_path)
                total_images = len([name for name in img_names if name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))])
                progress_bar["maximum"] = total_images  # 最大进度为总图像数

                for i, img_name in enumerate(tqdm(img_names)):
                    if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                        image_path = os.path.join(select_path, img_name)
                        image = Image.open(image_path)
                        processed_image = frcnn.detect_image(image)
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        processed_image.save(os.path.join(save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)
                        
                        progress_bar["value"] = i + 1  # 进度增加
                        new_window.update_idletasks()  # 更新 GUI
                
                messagebox.showinfo("预测完成！", "预测结果已保存于存储文件夹中。")
                progress_bar.destroy()  # 关闭进度条
                predict_button.config(state=tk.NORMAL)  # 重新启用预测按钮

            threading.Thread(target=run_model).start()
        else:
            messagebox.showerror("错误", "请先选择文件夹！")

    # 返回主窗口
    def return_to_root():
        new_window.destroy()
        root.deiconify()

    # 创建新窗口
    new_window = tk.Toplevel(root)
    new_window.title("多张预测")
    new_window.geometry("800x600")
    '''new_window.attributes('-topmost', True)'''

    select_button = tk.Button(new_window, text="请选择含需要预测图片的文件夹", font=("宋体", 16), command=select_folder)
    select_button.place(x=250, y=20)
    select_label = tk.Label(new_window, text="预测文件夹路径: ", font=("宋体", 12))
    select_label.place(x=15, y=70)

    save_button = tk.Button(new_window, text="请选择用于存储预测结果的文件夹", font=("宋体", 16), command=save_folder)
    save_button.place(x=250, y=150)
    save_label = tk.Label(new_window, text="存储文件夹路径: ", font=("宋体", 12))
    save_label.place(x=15, y=200)

    predict_button = tk.Button(new_window, text="开始预测", font=("宋体", 16), command=predict_folder)
    predict_button.place(x=375, y=300)

    return_button = tk.Button(new_window,text="返回",font=("宋体", 16),command=return_to_root)
    return_button.place(x=10,y=10)

    # 绑定子窗口关闭事件
    new_window.protocol("WM_DELETE_WINDOW", return_to_root)


# 统计分析窗口
def open_statistics_window():
    # 选择预测文件夹
    def select_folder():
        select_path = filedialog.askdirectory()
        if select_path:
            select_label.config(text=f"预测文件夹路径: {select_path}")
    
    # 选择存储文件夹
    def save_folder():
        save_path = filedialog.askdirectory()
        if save_path:
            save_label.config(text=f"存储文件夹路径: {save_path}")
    
    # 运行预测
    def predict_folder():
        select_path = select_label.cget("text").replace("预测文件夹路径: ", "")
        save_path = save_label.cget("text").replace("存储文件夹路径: ", "")
        if select_path and save_path:
            predict_button.config(state=tk.DISABLED)  # 禁用预测按钮，避免重复点击
            progress_bar = ttk.Progressbar(new_window, orient="horizontal", length=600, mode="indeterminate")  # 创建进度条
            progress_bar.place(x=100, y=400)
            progress_bar.start()  # 暂时只能用indeterminate（进度条循环，无法反映真实处理进度）
            
            # 使用多线程运行模型
            def run_model():
                frcnn = FRCNN()
                dir_classes_nums = frcnn.statistic(select_path)
                # 计算百分比
                Total = sum(dir_classes_nums)
                dir_classes_percentage = numpy.around(dir_classes_nums/Total, 4)
                condition = dir_classes_percentage > 0
                dir_classes_percentage_Nzero = numpy.extract(condition, dir_classes_percentage)
                # 计算物种丰度（S）
                S = len(dir_classes_percentage_Nzero)
                print("物种丰度S =", S)
                # 计算香农-威纳指数（H）
                H = -1*sum(dir_classes_percentage_Nzero * numpy.log(dir_classes_percentage_Nzero))
                print("香农-威纳指数H =", round(H,3))
                # 计算辛普森多样性指数（D）
                D = 1 - sum(numpy.power(dir_classes_percentage_Nzero,2))
                print("辛普森多样性指数D =", round(D,3))
                # 计算物种均匀度（J）
                J = H/numpy.log(S)
                print("物种均匀度J =", round(J,3))

                # 创建物种数据字典
                species_dict = {
                    '属名': frcnn.class_names,
                    '个数': list(dir_classes_nums),
                    '百分比': list(dir_classes_percentage * 100),
                }
                df_species = pandas.DataFrame(species_dict)  # 创建 DataFrame 来保存物种信息
                
                # 创建生物多样性指数字典
                community_dict = {
                    '物种丰度 (S)': [S],
                    '香农-威纳指数 (H)': [round(H, 3)],
                    '辛普森多样性指数 (D)': [round(D, 3)],
                    '物种均匀度 (J)': [round(J, 3)],
                }
                df_community = pandas.DataFrame(community_dict)  # 创建 DataFrame 来保存生物多样性指数
                
                # 将群落多样性指标添加到物种数据之后（合并成一个Excel文件的两个sheet）
                output_xlsx_path = os.path.join(save_path, 'output.xlsx')
                with pandas.ExcelWriter(output_xlsx_path) as writer:
                    df_species.to_excel(writer, sheet_name='物种数据', index=False)
                    df_community.to_excel(writer, sheet_name='生物多样性指数', index=False)

                messagebox.showinfo("预测完成！", f"统计分析结果已成功保存至{output_xlsx_path}中。")
                progress_bar.destroy()  # 关闭进度条
                predict_button.config(state=tk.NORMAL)  # 重新启用预测按钮

            threading.Thread(target=run_model).start()
        else:
            messagebox.showerror("错误", "请先选择文件夹！")
    
    # 返回主窗口
    def return_to_root():
        new_window.destroy()
        root.deiconify()

    # 创建新窗口
    new_window = tk.Toplevel(root)
    new_window.title("统计分析")
    new_window.geometry("800x600")
    '''new_window.attributes('-topmost', True)'''

    select_button = tk.Button(new_window, text="请选择含需要预测图片的文件夹", font=("宋体", 16), command=select_folder)
    select_button.place(x=250, y=20)
    select_label = tk.Label(new_window, text="预测文件夹路径: ", font=("宋体", 12))
    select_label.place(x=15, y=70)

    save_button = tk.Button(new_window, text="请选择用于存储预测结果的文件夹", font=("宋体", 16), command=save_folder)
    save_button.place(x=250, y=150)
    save_label = tk.Label(new_window, text="存储文件夹路径: ", font=("宋体", 12))
    save_label.place(x=15, y=200)

    predict_button = tk.Button(new_window, text="开始预测", font=("宋体", 16), command=predict_folder)
    predict_button.place(x=375, y=300)

    return_button = tk.Button(new_window,text="返回",font=("宋体", 16),command=return_to_root)
    return_button.place(x=10,y=10)

    # 绑定子窗口关闭事件
    new_window.protocol("WM_DELETE_WINDOW", return_to_root)


# 关闭主窗口
def destroy():
    sure = messagebox.askokcancel('关闭','确定关闭？')
    if sure :
        root.destroy()
    else :
        pass

def on_combobox_select(event):
    selected_option = combobox.get()
    if selected_option == "单张预测":
        open_image_window()
        root.withdraw()
    elif selected_option == "多张预测":
        open_folder_window()
        root.withdraw()
    elif selected_option == "统计分析":
        open_statistics_window()
        root.withdraw()

# 主窗口
root = tk.Tk()
root.title("藻类目标检测系统")
root.geometry("800x600")
root.resizable(False, False)
describe_1 = tk.Label(root, text="藻类目标检测系统", font=("宋体", 24)).place(x=250, y=50)
describe_2 = tk.Label(root, text="本系统基于 Faster RCNN 目标检测模型，可对常见藻类进行图像识别", font=("宋体", 16)).place(x=80, y=100)
describe_3 = tk.Label(root, text="功能说明：", font=("宋体", 16)).place(x=150, y=300)
describe_3 = tk.Label(root, text="单张预测：选择单个图片文件，对其进行预测，返回预测后的图片；", font=("宋体", 12)).place(x=150, y=330)
describe_4 = tk.Label(root, text="多张预测：选择图片文件夹，对其进行遍历预测，返回预测后的图片；", font=("宋体", 12)).place(x=150, y=360)
describe_5 = tk.Label(root, text="统计分析：选择图片文件夹，对其进行遍历预测和统计分析，返回统计分析结果。", font=("宋体", 12)).place(x=150, y=390)
root.protocol('WM_DELETE_WINDOW', destroy)

# 创建 Combobox
combobox_label = tk.Label(root, text="请选择目标检测功能", font=("宋体", 24))
combobox_label.place(x=250, y=160)
combobox = ttk.Combobox(root, values=["单张预测", "多张预测", "统计分析"], font=("宋体", 20), state="readonly")
combobox.place(x=250, y=200)
combobox.bind("<<ComboboxSelected>>", on_combobox_select)

root.mainloop()
