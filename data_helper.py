import os
import shutil

val_dir = './data/tiny-imagenet-200/val'
val_img_dir = os.path.join(val_dir, 'images')
val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')

# Read mapping from filename -> class
with open(val_annotations_file, 'r') as f:
    lines = f.readlines()
    val_img_dict = {line.split('\t')[0]: line.split('\t')[1] for line in lines}

# Create subfolders and move images
for img, cls in val_img_dict.items():
    cls_dir = os.path.join(val_dir, cls)
    os.makedirs(cls_dir, exist_ok=True)
    src = os.path.join(val_img_dir, img)
    dst = os.path.join(cls_dir, img)
    if os.path.exists(src):
        shutil.move(src, dst)

# Clean up
shutil.rmtree(val_img_dir)
print("✅ Validation set reorganized successfully.")

