import os
import cv2

# Set paths
image_dir = r'C:/Users/Disa Nabila/yolov5/runs/detect/predict3'
label_dir = r'C:/Users/Disa Nabila/yolov5/runs/detect/predict3/labels'

# New output directory to save cropped images, moved to C:/datasets
output_dir = r'C:/datasets/crops_split'

# Class names as per YOLOv8 training order
class_names = ['crack', 'over extrusion', 'under extrusion', 'warping']

# ========== CREATE OUTPUT FOLDERS ==========
for cls in class_names:
    os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

# ========== LOOP THROUGH LABEL FILES ==========
for label_file in os.listdir(label_dir):
    if not label_file.endswith('.txt'):
        continue

    image_name = label_file.replace('.txt', '.jpg')
    image_path = os.path.join(image_dir, image_name)
    label_path = os.path.join(label_dir, label_file)

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Image not found or unreadable: {image_path}")
        continue

    h, w = img.shape[:2]

    # Read bounding box label
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        cls_id, x, y, bw, bh = map(float, line.strip().split())
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)

        crop = img[y1:y2, x1:x2]

        # Skip invalid/empty crops
        if crop.size == 0:
            print(f"⚠️ Skipped empty crop: {image_path} [{x1},{y1},{x2},{y2}]")
            continue

        # Save cropped image to the corresponding class folder
        save_path = os.path.join(output_dir, class_names[int(cls_id)], f"{label_file[:-4]}_{i}.jpg")
        cv2.imwrite(save_path, crop)

print("✅ Cropping complete. Files saved to:", output_dir)

# Define class names based on your training
class_names = ['crack', 'over extrusion', 'under extrusion', 'warping']

# Make folders for each class
for cls in class_names:
    os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

# Loop over all label files
for label_file in os.listdir(label_dir):
    if not label_file.endswith('.txt'):
        continue

    image_name = label_file.replace('.txt', '.jpg')
    image_path = os.path.join(image_dir, image_name)
    label_path = os.path.join(label_dir, label_file)

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image not found: {image_path}")
        continue

    h, w = img.shape[:2]

    # Read label lines
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        cls_id, x, y, bw, bh = map(float, line.strip().split())
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)

        crop = img[y1:y2, x1:x2]

        save_path = os.path.join(output_dir, class_names[int(cls_id)], f"{label_file[:-4]}_{i}.jpg")
        cv2.imwrite(save_path, crop)
