
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from ultralytics import RTDETR

if __name__ == '__main__':
    model = YOLO('./ultralytics/cfg/models/v11/yolov11.yaml')
    # model = YOLO('./ultralytics/cfg/models/v11/yolov11n.yaml')
    # model = YOLO('./ultralytics/cfg/models/v11/yolov11-star-HSPAN.yaml')
    # model = YOLO('./ultralytics/cfg/models/v11/yolov11-HSPAN.yaml')
    # model = YOLO('./ultralytics/cfg/models/v11/yolov11-ghost.yaml')
    # model = YOLO('./ultralytics/cfg/models/v11/yolov11-starEMA.yaml')
    # model = YOLO('./ultralytics/cfg/models/rt-detr/rtdetr-l.yaml')
    model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='E:/桌面/Lab/项目/病虫害检测/v8tov11/finaldataset/finaldataset.yaml',
                cache=True,
                imgsz=640,
                epochs=120,  # 快速验证用20轮，正式训练改回120
                batch=2,  # 6GB显存稳妥值
                close_mosaic=10,
                workers=0,
                device='0',
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )
# if __name__ == '__main__':
#     model = RTDETR(
#     './ultralytics/cfg/models/rt-detr/rtdetr-l.yaml')
#     model.train(data='E:/桌面/Lab/项目/病虫害检测/v8tov11/finaldataset/finaldataset.yaml',
#                 cache=True,
#                 imgsz=640,
#                 epochs=300,
#                 batch=2,  # 6GB显存稳妥值
#                 close_mosaic=10,
#                 workers=0,
#                 device='0',
#                 optimizer='SGD', # using SGD
#                 # resume='', # last.pt path
#                 # amp=False, # close amp
#                 # fraction=0.2,
#                 project='runs/train',
#                 name='rt',
#             )