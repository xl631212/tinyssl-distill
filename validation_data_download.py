import os
import urllib.request
import zipfile
import tarfile
from torchvision.datasets import Caltech101, Flowers102

'''
# ---- Tiny-ImageNet-200 ----
if not os.path.exists("tiny-imagenet-200"):
    print("Downloading tiny-imagenet-200...")
    urllib.request.urlretrieve("http://cs231n.stanford.edu/tiny-imagenet-200.zip", "tiny-imagenet-200.zip")
    print("Extracting tiny-imagenet-200...")
    with zipfile.ZipFile("tiny-imagenet-200.zip", 'r') as zip_ref:
        zip_ref.extractall(".")
    os.remove("tiny-imagenet-200.zip")
    print("Done.")
else:
    print("tiny-imagenet-200 already exists.")

# ---- CIFAR-10-C ----
if not os.path.exists("cifar10c"):
    print("Downloading CIFAR-10-C...")
    urllib.request.urlretrieve("https://zenodo.org/record/2535967/files/CIFAR-10-C.tar", "CIFAR-10-C.tar")
    print("Extracting CIFAR-10-C...")
    with tarfile.open("CIFAR-10-C.tar", 'r') as tar_ref:
        tar_ref.extractall(".")
    os.rename('./CIFAR-10-C', './cifar10c')
    os.remove("CIFAR-10-C.tar")
    print("Done.")
else:
    print("cifar10c already exists.")

    # ---- Flowers102 (替代 ImageNetV2/LAION-400M，适合zero-shot) ----
try:
    print("Downloading Oxford Flowers102 via torchvision...")
    Flowers102(root='./', split="test", download=True)
    print("Done.")
except Exception as e:
    print(f"Flowers102 download failed: {e}")

'''
# ---- Caltech101 (use torchvision, will auto download, ignore HTTP errors) ----

import os
import zipfile
import tarfile

# Step 1: 解压 caltech-101.zip（如已解压可跳过）
zip_path = "/home/xuyingl/iccv_workshop/caltech-101.zip"
unzip_dir = "/home/xuyingl/iccv_workshop/caltech-101"
if not os.path.exists(unzip_dir):
    print("Unzipping caltech-101.zip ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_dir)
    print("Unzipped.")
else:
    print("Already unzipped.")

# Step 2: 解压 101_ObjectCategories.tar.gz（注意这里的路径）
tar_gz_path = "/home/xuyingl/iccv_workshop/caltech-101/caltech-101/101_ObjectCategories.tar.gz"
caltech101_dir = "/home/xuyingl/iccv_workshop/caltech101/101_ObjectCategories"
os.makedirs("/home/xuyingl/iccv_workshop/caltech101", exist_ok=True)
if os.path.exists(tar_gz_path) and not os.path.exists(caltech101_dir):
    print("Extracting 101_ObjectCategories.tar.gz ...")
    with tarfile.open(tar_gz_path, 'r:gz') as tar_ref:
        tar_ref.extractall("/home/xuyingl/iccv_workshop/caltech101")
    print("Extracted.")
else:
    print("Already extracted or .tar.gz not found.")

# Step 3: 检查
if os.path.exists(caltech101_dir):
    print(f"Success! Caltech101 extracted at: {caltech101_dir}")
    print("Example classes:", os.listdir(caltech101_dir)[:5])
else:
    print("Extraction failed, please check paths.")



# 你的 zero-shot 评测 eval_dataset_root 填 "/home/xuyingl/iccv_workshop/caltech101"


# 3. 你的零样本脚本参数就填：
#    eval_dataset_root="/home/xuyingl/iccv_workshop/caltech101"
