import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# onnx onnxsim onnxruntime onnxruntime-gpu

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    model.export(format='onnx', simplify=True, opset=13)

# import torch
#
# print(torch.__version__)
# print(torch.version.cuda)
# print(torch.backends.cudnn.version())