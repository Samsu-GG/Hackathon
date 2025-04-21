import os
import shutil


source_dir = r"D:\Hackathon\data\trashnet-master\data\dataset-resized"
target_dir = "processed_dataset"


recycle_classes = ["cardboard", "glass", "metal", "paper", "plastic"]


trash_classes = ["trash"]


os.makedirs(os.path.join(target_dir, "recycle"), exist_ok=True)
os.makedirs(os.path.join(target_dir, "trash"), exist_ok=True)
os.makedirs(os.path.join(target_dir, "compost"), exist_ok=True)  # এখন খালি থাকবে


for category in os.listdir(source_dir):
    category_path = os.path.join(source_dir, category)

    if not os.path.isdir(category_path):
        continue

    if category in recycle_classes:
        target_category = "recycle"
    elif category in trash_classes:
        target_category = "trash"
    else:
        continue

    for file in os.listdir(category_path):
        src_file = os.path.join(category_path, file)
        dst_file = os.path.join(target_dir, target_category, file)
        shutil.copy(src_file, dst_file)