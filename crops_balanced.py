import os
import shutil
import random

# Your input folder: where your current cropped images live
source_root = r'C:/datasets/crops_split'

# New output folder
output_root = r'C:/datasets/crops_balanced'
os.makedirs(output_root, exist_ok=True)

# Target count per class
target_count = 730

# List of class folders
class_names = ['crack', 'over extrusion', 'under extrusion', 'warping']

for cls in class_names:
    src_folder = os.path.join(source_root, cls)
    dst_folder = os.path.join(output_root, cls)
    os.makedirs(dst_folder, exist_ok=True)

    all_images = [f for f in os.listdir(src_folder) if f.lower().endswith(('.jpg', '.png'))]

    # Shuffle and pick 730 only
    random.shuffle(all_images)
    selected = all_images[:target_count]

    # Copy selected files to balanced directory
    for fname in selected:
        shutil.copy(os.path.join(src_folder, fname), os.path.join(dst_folder, fname))

    print(f"✅ {cls}: {len(selected)} images copied.")
