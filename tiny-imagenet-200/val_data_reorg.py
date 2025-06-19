import os
import shutil

val_dir = "tiny-imagenet-200/val"
img_dir = os.path.join(val_dir, "images")
anno_file = os.path.join(val_dir, "val_annotations.txt")

# 创建类别文件夹
with open(anno_file, "r") as f:
    for line in f:
        img, cls, *_ = line.strip().split("\t")
        cls_folder = os.path.join(val_dir, cls)
        if not os.path.exists(cls_folder):
            os.mkdir(cls_folder)
        src = os.path.join(img_dir, img)
        dst = os.path.join(cls_folder, img)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.move(src, dst)
print("val/ 目录已整理为 ImageFolder 可用格式。")
