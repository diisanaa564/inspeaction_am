import os
import shutil
import random

# Input folder
input_dir = r'C:/datasets/crops_balanced'
# Output root
output_dir = r'C:/datasets/crops_balanced_split'

# Split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Class folders
class_names = ['crack', 'over extrusion', 'under extrusion', 'warping']

for cls in class_names:
    cls_path = os.path.join(input_dir, cls)
    images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.png'))]
    random.shuffle(images)

    total = len(images)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)

    splits = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }

    for split, files in splits.items():
        split_dir = os.path.join(output_dir, split, cls)
        os.makedirs(split_dir, exist_ok=True)
        for f in files:
            src = os.path.join(cls_path, f)
            dst = os.path.join(split_dir, f)
            shutil.copy(src, dst)

        print(f"✅ {cls} → {split}: {len(files)} images")
