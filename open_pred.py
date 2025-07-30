from ultralytics import YOLO

model = YOLO('C:/Users/Disa Nabila/yolov5/runs/detect/train10/weights/best.pt') #loads trained YOLOv8 

model.predict(
    #source='C:/datasets/split_sampledv3/test/images', # prev problem
    source='C:/datasets/inspectionam_sampled/sampledv3/images',
    save=True,
    save_txt=True  #saves bounding box labels
)
