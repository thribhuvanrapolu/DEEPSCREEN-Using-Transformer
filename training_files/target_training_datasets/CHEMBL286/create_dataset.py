import os
import shutil
import json

# Load the JSON file
with open('train_val_test_dict.json', 'r') as f:
    data = json.load(f)

# Create folders for train, test, and validate
folders = ['training', 'test', 'validation']
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    for class_label in ['class_0', 'class_1']:
        os.makedirs(os.path.join(folder, class_label), exist_ok=True)

# Move images to respective folders
for dataset, images in data.items():
    for image, label in images:
        src_path = os.path.join('imgs', f'{image}.png')  # Assuming images have .jpg extension
        dest_folder = os.path.join(dataset, f'class_{label}')
        dest_path = os.path.join(dest_folder, f'{image}.png')
        if os.path.exists(src_path):  # Check if the image exists before moving
            shutil.copy(src_path, dest_path)
        else:
            print(f"Warning: Image {image} not found in the source folder.")
