import os
import torch
import numpy as np
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import clip
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from diffmix_mae_patch_train import SimCLR_MAE_Patch

# -------- 数据采样 --------
root = './data/all'
n_per_class = 20  # 每类采样
labels, imgs = [], []

for ci, cname in enumerate(sorted(os.listdir(root))):
    cdir = os.path.join(root, cname)
    cimgs = sorted([os.path.join(cdir, f) for f in os.listdir(cdir) if f.lower().endswith(('.jpg','.png'))])
    imgs += cimgs[:n_per_class]
    labels += [cname]*min(n_per_class, len(cimgs))

# -------- 模型/变换 --------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clip_model, clip_pre = clip.load("ViT-B/32", device=device)
clip_model.eval()

student = SimCLR_MAE_Patch(models.resnet18(weights=None), use_mae=False).to(device)
student.load_state_dict(torch.load('/home/xuyingl/iccv_workshop/student_stronghead_l1_final.pth', map_location=device), strict=False)
student.eval()

student_transform = transforms.Compose([
    transforms.Resize(224), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
])

# -------- 提取特征 --------
clip_feats, stu_feats, y = [], [], []

with torch.no_grad():
    for path, label in zip(imgs, labels):
        img = Image.open(path).convert('RGB')
        # CLIP
        img_clip = clip_pre(img).unsqueeze(0).to(device)
        cf = clip_model.encode_image(img_clip).cpu().numpy().flatten()
        clip_feats.append(cf)
        # Student
        img_stu = student_transform(img).unsqueeze(0).to(device)
        sf, _, _, _ = student(img_stu)
        stu_feats.append(sf.cpu().numpy().flatten())
        y.append(label)

clip_feats, stu_feats = np.array(clip_feats), np.array(stu_feats)

# -------- t-SNE 可视化 --------
def plot_embeds(feats, y, title, save_path):
    tsne = TSNE(n_components=2, random_state=0)
    embeds = tsne.fit_transform(feats)
    plt.figure(figsize=(9, 7))
    for cname in sorted(set(y)):
        idx = [i for i, lab in enumerate(y) if lab == cname]
        plt.scatter(embeds[idx,0], embeds[idx,1], label=cname, alpha=0.7, s=24)
    plt.title(title)
    plt.legend(fontsize=10, loc="best", bbox_to_anchor=(1.05, 1.0))
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

plot_embeds(clip_feats, y, "CLIP Teacher Features (t-SNE)", "tsne_clip_teacher.png")
plot_embeds(stu_feats, y, "Student Features (t-SNE)", "tsne_student.png")

# -------- PCA 可视化 --------
def plot_pca(feats, y, title, save_path):
    pca = PCA(n_components=2)
    embeds = pca.fit_transform(feats)
    plt.figure(figsize=(9, 7))
    for cname in sorted(set(y)):
        idx = [i for i, lab in enumerate(y) if lab == cname]
        plt.scatter(embeds[idx,0], embeds[idx,1], label=cname, alpha=0.7, s=24)
    plt.title(title)
    plt.legend(fontsize=10, loc="best", bbox_to_anchor=(1.05, 1.0))
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

plot_pca(clip_feats, y, "CLIP Teacher Features (PCA)", "pca_clip_teacher.png")
plot_pca(stu_feats, y, "Student Features (PCA)", "pca_student.png")

print("特征分布可视化已自动保存为: tsne_clip_teacher.png, tsne_student.png, pca_clip_teacher.png, pca_student.png")
