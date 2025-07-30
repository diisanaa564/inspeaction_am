from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='C:/datasets/split_sampledv3/dataset.yaml',
    epochs=100,
    imgsz=640
)
