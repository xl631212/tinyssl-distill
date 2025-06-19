import os
import random
import json
from shutil import copyfile

src_dir = '/home/xuyingl/.cache/kagglehub/datasets/deeptrial/miniimagenet/versions/2/ImageNet-Mini/images'
json_path = '/home/xuyingl/.cache/kagglehub/datasets/deeptrial/miniimagenet/versions/2/ImageNet-Mini/imagenet_class_index.json'
out_dir = './data/real'
num_per_class = 5

os.makedirs(out_dir, exist_ok=True)

# 读取class index映射表
with open(json_path, 'r') as f:
    class_map = json.load(f)
# 构建 {wnid: readable_name} 字典
wnid2name = {v[0]: v[1] for k, v in class_map.items()}

class_names = sorted([d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))])

total = 0
for cname in class_names:
    class_dir = os.path.join(src_dir, cname)
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(images) < num_per_class:
        continue
    # 用可读类别名作为输出文件夹
    readable_name = wnid2name.get(cname, cname)
    out_cdir = os.path.join(out_dir, readable_name)
    os.makedirs(out_cdir, exist_ok=True)
    selected = random.sample(images, num_per_class)
    for i, fname in enumerate(selected):
        src_img = os.path.join(class_dir, fname)
        dst_img = os.path.join(out_cdir, f'{i}.jpg')
        copyfile(src_img, dst_img)
        total += 1
    print(f'类别 {cname}（{readable_name}） 完成采样 {len(selected)} 张')

print(f'采样完成，总计 {total} 张图片。')
