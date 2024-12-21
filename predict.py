import os
import time
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from frcnn import FRCNN

if __name__ == "__main__":
    frcnn = FRCNN()
    # mode用于指定测试的模式：
    # 'predict'      表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    # 'video'        表示视频预测，可调用摄像头或者视频进行预测，详情查看对应参数及注释。
    # 'fps'          表示测试fps，使用的图片是img里面的street.jpg，详情查看对应参数及注释。
    # 'dir_predict'  表示遍历文件夹进行预测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看对应参数及注释。
    # 'statistic'    表示遍历文件夹进行预测并保存。在dir_save_path中生成一个统计生物多样性的Excel表格。
    mode = "predict"
    
    if mode == "predict":
        crop  = False  #指定是否在单张图片预测后对目标进行截取
        count = True   #指定是否进行目标的计数

        img = input('请指定需要预测的图片路径:')
        try:
            image = Image.open(img)
        except:
            print('错误！图片不存在！')
        else:
            r_image = frcnn.detect_image(image, crop = crop, count = count)
            r_image.show()


    elif mode == "video":
        # 保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
        video_path      = 0   #指定原视频的路径，当video_path=0时表示检测摄像头，="指定视频路径"
        video_save_path = ""  #指定视频保存的路径，当video_save_path=""时表示不保存，="视频保存路径"
        video_fps       = 30  #指定保存的视频的fps

        capture=cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        fps = 0.0
        while(True):
            t1 = time.time()
            ref,frame=capture.read()  #读取某一帧
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)  #格式转变，BGRtoRGB
            frame = Image.fromarray(np.uint8(frame))  #转变成Image
            frame = np.array(frcnn.detect_image(frame))  #进行预测
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)  #RGBtoBGR满足opencv显示格式
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)
            if c==27:
                capture.release()
                break
        capture.release()
        out.release()
        cv2.destroyAllWindows()


    elif mode == "fps":
        test_interval  = 100               #指定测量fps的时候，图片预测的次数。理论上test_interval越大，fps越准确
        fps_image_path = "img/street.jpg"  #指定测试的fps图片

        img = Image.open(fps_image_path)
        tact_time = frcnn.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        dir_origin_path = "c:/Users/86137/Desktop/111"  #指定需要预测的图片文件夹路径
        dir_save_path   = "c:/Users/86137/Desktop/222"  #指定预测后图片的保存路径

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = frcnn.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    elif mode == "statistic":
        count = True
        dir_origin_path = "D:/PMID2019/dir_origin_path"  #指定需要预测的图片文件夹路径
        dir_save_path   = "D:/PMID2019/dir_save_path"  #指定预测后图片的保存路径

        dir_classes_nums = frcnn.statistic(dir_origin_path, dir_save_path)

        # 计算百分比
        Total = sum(dir_classes_nums)
        dir_classes_percentage = np.around(dir_classes_nums/Total, 4)

        condition = dir_classes_percentage > 0
        dir_classes_percentage_Nzero = np.extract(condition, dir_classes_percentage)

        # 计算物种丰富度（S）
        S = len(dir_classes_percentage_Nzero)
        print("物种丰富度S =", S)

        # 计算香农-威纳指数（H）
        H = -1*sum(dir_classes_percentage_Nzero * np.log(dir_classes_percentage_Nzero))
        print("香农-威纳指数H =", round(H,3))

        # 计算辛普森多样性指数（1-D）
        D = 1 - sum(np.power(dir_classes_percentage_Nzero,2))
        print("辛普森多样性指数D =", round(D,3))

        # 计算物种均匀度
        J = H/np.log(S)
        print("物种均匀度J =", round(J,3))

        stat_dict = {'Names':frcnn.class_names, 'Count':list(dir_classes_nums), 'Percentage':list(dir_classes_percentage*100)}
        df = pd.DataFrame(stat_dict)
        print(df)
        df.to_excel('D:/PMID2019/dir_save_path/output.xlsx')

    else:
        raise AssertionError("请指定正确的模式: 'predict', 'video', 'fps', 'dir_predict' or 'statistic'.")
