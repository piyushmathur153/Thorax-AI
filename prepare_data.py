import os
import shutil
from sklearn.model_selection import train_test_split

raw_dir = "data/raw"
processed_dir = "data/processed"

categories = ['Normal', 'Tuberculosis']
splits = ['train', 'val', 'test']

# Creating Destination Folder
for split in splits:
    for cat in ['normal', 'tb']:
        os.makedirs(os.path.join(processed_dir, split, cat), exist_ok=True)

# Loading Normal Scans
normal_dir = os.path.join(raw_dir, 'Normal')
normal_files = [
    os.path.join(normal_dir, f) for f in os.listdir(normal_dir) 
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
]

# Loading TB Scans
tb_dir = os.path.join(raw_dir, 'Tuberculosis')
tb_files = [
    os.path.join(tb_dir, f) for f in os.listdir(tb_dir) 
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
]

all_files = normal_files + tb_files
all_labels = [0] * len(normal_files) + [1] * len(tb_files)

print(f"Total Images Found: {len(all_files)}")
print(f"Normal: {len(normal_files)} | TB: {len(tb_files)}")

# 80% Train, 20% Temp (Val + Test)
train_files, temp_files, train_labels, temp_labels = train_test_split(
    all_files, all_labels, test_size=0.20, random_state=42, stratify=all_labels
)

# 10% Val, 10% Test (50-50 Split of Temp)
val_files, test_files, val_labels, test_labels = train_test_split(
    temp_files, temp_labels, test_size=0.50, random_state=42, stratify=temp_labels
)

def copy_files(file_list, label_list, split_name):
    print(f"\nProcessing {split_name} split...")
    for fpath, label in zip(file_list, label_list):
        subfolder = 'tb' if label == 1 else 'normal'
        fname = os.path.basename(fpath)
        dest = os.path.join(processed_dir, split_name, subfolder, fname)
        shutil.copyfile(fpath, dest)
    print(f"--> {split_name.capitalize()} completed: {len(file_list)} images (TB: {label_list.count(1)}, Normal: {label_list.count(0)})")

copy_files(train_files, train_labels, 'train')
copy_files(val_files, val_labels, 'val')
copy_files(test_files, test_labels, 'test')

print("\nAll datasets split and copied successfully into data/processed/!")