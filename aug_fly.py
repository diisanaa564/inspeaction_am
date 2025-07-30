#yolo+aug
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='C:/datasets/split_sampledv3/dataset.yaml',
    epochs=100,
    imgsz=640,
    degrees=15,        # random rotation up to ±15°
    translate=0.1,     # shift image up to 10%
    scale=0.8,         # random scale 80% to 120%
    shear=2,           # random shear up to 2°
    perspective=0.001, # perspective transform
    flipud=0.3,        # vertical flip probability
    fliplr=0.5,        # horizontal flip probability
    mosaic=1.0,        # mosaic ON (default: 1.0)
    mixup=0.0          # mixup OFF (set >0 if you want to try)
)
