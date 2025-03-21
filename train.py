import datetime
import os
from functools import partial
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.FasterRCNN import FasterRCNN
from nets.FasterRCNN_training import (FasterRCNNTrainer, get_lr_scheduler,set_optimizer_lr, weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import FRCNNDataset, frcnn_dataset_collate
from utils.utils import (get_classes, seed_everything, show_config,worker_init_fn)
from utils.utils_fit import fit_one_epoch


if __name__ == "__main__":

    # 迁移训练 ———— model_path = '预训练权重路径'; 预训练权重为data_model/voc_weights_resnet.pth
    # 继续训练 ———— model_path = 'logs文件夹下相应权值文件路径'; 修改下方的 冻结阶段 或者 解冻阶段 的参数，来保证模型epoch的连续性
    # 主干训练 ———— model_path = ''; 下方的 pretrain = True
    # 从零训练 ———— model_path = ''; 下方的pretrain = Fasle，Freeze_Train = Fasle
    # model_path = 'logs/train_2025_03_08weights.pth'
    model_path = 'logs/train_2025_02_23weights.pth'

    #classes_path = 'data_model/BJFUHJZooPlankton.txt'  
    classes_path = 'data_model/Algae2024.txt'  # 指向data_model下的分类目标txt
    # train_annotation_path = 'data_image/test_train.txt'  
    train_annotation_path = 'data_image/Algae2024_train.txt'  # 指向image_data下的由annotation生成的文件训练txt
    # val_annotation_path = 'data_image/test_val.txt'  
    val_annotation_path = 'data_image/Algae2024_val.txt'  # 指向image_data下的由annotation生成的文件检测txt
    backbone = "resnet50"  # 主干特征提取网络

    Cuda         = True  # 是否使用Cuda
    seed         = 11  # 用于固定随机种子，使得每次独立训练都可以获得一样的结果
    train_gpu    = [0,]  # 训练用到的GPU，默认为第一张卡、双卡为[0, 1]、三卡为[0, 1, 2];在使用多GPU时，每个卡上的batch为总batch除以卡的数量
    fp16         = True  # 是否使用混合精度训练
    input_shape  = [600, 600]  # 输入的shape大小
    anchors_size = [4, 16, 32]  # 用于设定先验框的大小，每个数对应3个先验框。详见anchors.py

    # 训练模式参数
    pretrained   = False  # 是否使用主干网络的预训练权重，此处使用的是主干的权重，因此是在模型构建的时候进行加载的。
    Freeze_Train = True   # 是否进行冻结训练，默认先冻结训练后解冻训练。如果设置Freeze_Train=False，建议使用优化器为sgd
    # 如果设置了model_path，则主干的权值无需加载，pretrained的值无意义。
    # 如果不设置model_path，pretrained = True，此时仅加载主干开始训练。
    # 如果不设置model_path，pretrained = False，Freeze_Train = Fasle，此时从零训练，且没有冻结主干的过程。

    # 冻结阶段训练参数
    # Init_Epoch         模型当前开始的训练世代，其值可以大于Freeze_Epoch，如此则会跳过冻结阶段，直接从设置的世代开始，并调整对应的学习率。
    # Freeze_Epoch       模型冻结训练的Freeze_Epoch
    # Freeze_batch_size  模型冻结训练的batch_size
    Init_Epoch          = 0  # 继续训练时可使用
    Freeze_Epoch        = 50  # 根据需求改
    Freeze_batch_size   = 8

    # 解冻阶段训练参数
    # UnFreeze_Epoch       模型总共训练的世代。SGD推荐设置较大的UnFreeze_Epoch，Adam可以设置较小的UnFreeze_Epoch
    # Unfreeze_batch_size  模型在解冻后的batch_size
    UnFreeze_Epoch      = 100  # 根据需求改
    Unfreeze_batch_size = 4
    

    optimizer_type = "adam"  # 使用到的优化器种类，可选的有adam、sgd
    momentum       = 0.9  # 优化器内部使用到的momentum参数
    weight_decay   = 0  # 权值衰减，可防止过拟合。当使用adam时建议设置为0
    Init_lr        = 1e-4  # 模型的最大学习率。当使用Adam优化器时建议=1e-4，当使用SGD优化器时建议=1e-2
    Min_lr         = Init_lr * 0.01  # 模型的最小学习率。建议设置为最大学习率的0.01
    lr_decay_type  = 'cos'  # 使用到的学习率下降方式，可选的有step、cos
    save_period    = 5  # 每隔若干世代保存一次权值
    eval_flag      = True  # 是否在训练时进行评估，评估对象为验证集。此处获得的mAP会与get_map.py获得的会有所不同。原因：此处获得的mAP为验证集的mAP；此处设置评估参数较为保守，目的是加快评估速度
    eval_period    = 5  # 每隔若干世代评估一次，频繁评估会增加训练耗时
    num_workers    = 4  # 用于设置是否使用多线程读取数据，1代表关闭多线程。开启后会加快数据读取速度，但是会占用更多内存
    save_dir       = 'logs'  # 权值与日志文件保存的文件夹


    # 获取classes和anchor
    class_names, num_classes = get_classes(classes_path)
    # 设置显卡
    os.environ["CUDA_VISIBLE_DEVICES"]  = ','.join(str(x) for x in train_gpu)
    ngpus_per_node                      = len(train_gpu)
    print('Number of devices: {}'.format(ngpus_per_node))
    seed_everything(seed)
    
    model = FasterRCNN(num_classes, anchor_scales = anchors_size, backbone = backbone, pretrained = pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        
        # 根据预训练权重的Key和模型的Key进行加载
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # 显示没有匹配上的Key
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    # 记录Loss
    time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history    = LossHistory(log_dir, model, input_shape=input_shape)

    # torch 1.2不支持amp，显示"could not be resolve"，建议使用torch 1.7.1及以上正确使用fp16
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model_train)
        cudnn.benchmark = True
        model_train = model_train.cuda()
    
    # 读取数据集对应的txt
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
    
    show_config(
        classes_path = classes_path, model_path = model_path, input_shape = input_shape, \
        Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
        Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
        save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
    )

    # 总训练世代指的是遍历全部数据的总次数
    # 总训练步长指的是梯度下降的总次数 
    # 每个训练世代包含若干训练步长，每个训练步长进行一次梯度下降。
    # 此处仅建议最低训练世代，上不封顶，计算时只考虑了解冻部分
    wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
    total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
    if total_step <= wanted_step:
        if num_train // Unfreeze_batch_size == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
        wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
        print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m"%(optimizer_type, wanted_step))
        print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
        print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m"%(total_step, wanted_step, wanted_epoch))

    # 主干特征提取网络特征通用，冻结训练可以加快训练速度，也可以在训练初期防止权值被破坏。
    # 提示OOM或者显存不足请调小Batch_size
    if True:
        UnFreeze_flag = False
        # 冻结一定部分训练
        if Freeze_Train:
            for param in model.extractor.parameters():
                param.requires_grad = False
        # 冻结主干特征提取网络层
        model.freeze_bn()

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        # 判断当前batch_size，自适应调整学习率
        nbs             = 16
        lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 5e-2
        lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        
        # 根据optimizer_type选择优化器
        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        # 获得学习率下降的公式
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        # 判断每一个世代的长度
        epoch_step     = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        train_dataset = FRCNNDataset(train_lines, input_shape, train = True)
        val_dataset   = FRCNNDataset(val_lines, input_shape, train = False)

        gen           = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=frcnn_dataset_collate, 
                                    worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
        gen_val       = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=frcnn_dataset_collate, 
                                    worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))

        train_util    = FasterRCNNTrainer(model_train, optimizer)

        # 记录eval的map曲线
        eval_callback = EvalCallback(model_train, input_shape, class_names, num_classes, val_lines, log_dir, Cuda, \
                                        eval_flag=eval_flag, period=eval_period)

        # 开始模型训练
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            # 如果模型有冻结学习部分则解冻，并设置参数
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                # 判断当前batch_size，自适应调整学习率
                nbs          = 16
                lr_limit_max = 1e-4 if optimizer_type == 'adam' else 5e-2
                lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit  = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit   = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                # 获得学习率下降的公式
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                
                for param in model.extractor.parameters():
                    param.requires_grad = True
                # 冻结主干特征提取网络层
                model.freeze_bn()

                epoch_step     = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                gen     = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=frcnn_dataset_collate, worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
                gen_val = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=frcnn_dataset_collate, worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))

                UnFreeze_flag = True
                
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            
            fit_one_epoch(model, train_util, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir)
            
        loss_history.writer.close()


'''
训练目标检测模型一定需要注意以下几点：
1、训练前仔细检查自己的格式是否满足要求，该库要求数据集格式为VOC格式，需要准备好的内容有输入图片和标签
   输入图片为.jpg图片，无需固定大小，传入训练前会自动进行resize。
   灰度图会自动转成RGB图片进行训练，无需自己修改。
   输入图片如果后缀非jpg，需要自己批量转成jpg后再开始训练。
   标签为.xml格式，文件中会有需要检测的目标信息，标签文件和输入图片文件相对应。
2、损失值的大小用于判断是否收敛，比较重要的是有收敛的趋势，即验证集损失不断下降，如果验证集损失基本上不改变的话，模型基本上就收敛了。
   损失值的具体大小并没有什么意义，大和小只在于损失的计算方式，并不是接近于0才好。如果想要让损失好看点，可以直接到对应的损失函数里面除上10000。
   训练过程中的损失值会保存在logs文件夹下的loss_%Y_%m_%d_%H_%M_%S文件夹中
3、训练好的权值文件保存在logs文件夹中，每个训练世代（Epoch）包含若干训练步长（Step），每个训练步长（Step）进行一次梯度下降。
   如果只是训练了几个Step是不会保存的，Epoch和Step的概念要捋清楚一下。
'''
'''
训练分为两个阶段，分别是冻结阶段和解冻阶段：
冻结阶段：backbone被冻结，不发生改变;此时主要训练RPN网络和ROI网络。占用的显存较小。
解冻阶段：backbone被解冻，网络所有参数都会发生改变。占用的显存较大。
若干参数设置建议：
（一）从整个模型的预训练权重开始训练： 
    Adam：
        Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 1e-4。（冻结）
        Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 1e-4。（不冻结）
    SGD：
        Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 150，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 1e-2。（冻结）
        Init_Epoch = 0，UnFreeze_Epoch = 150，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2。（不冻结）
    其中：UnFreeze_Epoch可以在100-300之间调整。
（二）从主干网络的预训练权重开始训练：
    Adam：
        Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 1e-4。（冻结）
        Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 1e-4。（不冻结）
    SGD：
        Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 150，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 1e-2。（冻结）
        Init_Epoch = 0，UnFreeze_Epoch = 150，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2。（不冻结）
    其中：由于从主干网络的预训练权重开始训练，主干的权值不一定适合目标检测，需要更多的训练跳出局部最优解。
            UnFreeze_Epoch可以在150-300之间调整，YOLOV5和YOLOX均推荐使用300。
            Adam相较于SGD收敛的快一些。因此UnFreeze_Epoch理论上可以小一点，但依然推荐更多的Epoch。
（三）batch_size的设置：
    在显卡能够接受的范围内，以大为好。显存不足与数据集大小无关，提示显存不足（OOM或者CUDA out of memory）请调小batch_size。
faster rcnn的Batch BatchNormalization层已经冻结，batch_size可以为1
'''