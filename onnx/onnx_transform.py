import torch
import onnx
from torchvision import models
from nets.FasterRCNN import FasterRCNN

# 使用可用的 GPU，否则使用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 导入训练好的模型(.pth 格式)
model = FasterRCNN(num_classes=24) # 注意修改num_classes(分类类别数)和下面model_path(模型地址)
model_path = 'logs/ep010-loss0.633-val_loss0.435.pth'

# 将模型设置为评估模式,并移动到设备
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)

# 构造输入图像张量，并确保在同一设备上
input = torch.randn(1, 3, 600, 600).to(device)

# 运行模型以获得输出
with torch.no_grad():
    output = model(input)

# 将 PyTorch 模型转换为 ONNX 模型
onnx_path = model_path.replace('.pth', '.onnx')

torch.onnx.export(
    model,
    input,
    onnx_path,
    opset_version=15,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    do_constant_folding=False)

print("ONNX 模型导出成功。")

# 加载并检查 ONNX 模型
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print('无报错，onnx模型载入成功')