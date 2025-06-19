import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import clip

from diffmix_mae_patch_train import SimCLR_MAE_Patch

# ====== 0. 只用有的类别（65类） ======
ALL_DATA = "/home/xuyingl/iccv_workshop/data/all"
eval_classes = sorted([d for d in os.listdir(ALL_DATA) if os.path.isdir(os.path.join(ALL_DATA, d))])
class_to_idx = {c: i for i, c in enumerate(eval_classes)}
num_classes = len(eval_classes)
print(f"只用这 {num_classes} 类：{eval_classes}")

# ====== 1. 通用ImageFolder（65类线性探针/全量合并数据） ======
class CustomImageFolder(Dataset):
    def __init__(self, root, class_to_idx, transform=None):
        self.samples = []
        self.transform = transform
        for c in class_to_idx:
            cdir = os.path.join(root, c)
            if not os.path.isdir(cdir): continue
            for f in os.listdir(cdir):
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.samples.append((os.path.join(cdir, f), class_to_idx[c]))
    def __getitem__(self, idx):
        img, label = self.samples[idx]
        img = Image.open(img).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
    def __len__(self): return len(self.samples)

def linear_probe_evaluate(student_ckpt, proj_dim=512, all_data=ALL_DATA, class_to_idx=class_to_idx):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimCLR_MAE_Patch(models.resnet18(weights=None), use_mae=False, proj_dim=proj_dim)
    model.load_state_dict(torch.load(student_ckpt, map_location=device), strict=False)
    model.to(device)
    model.eval()
    for param in model.parameters(): param.requires_grad = False
    clf = nn.Linear(proj_dim, len(class_to_idx)).to(device)
    optimizer = optim.Adam(clf.parameters(), lr=1e-2)
    tfm = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    allset = CustomImageFolder(all_data, class_to_idx, tfm)
    N = len(allset)
    train_size = int(0.8 * N)
    val_size = N - train_size
    trainset, valset = torch.utils.data.random_split(allset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
    valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=4)
    for epoch in range(10):
        model.eval(); clf.train()
        for imgs, labels in tqdm(trainloader, desc=f"Linear probe Epoch {epoch+1}"):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                feats, _, _, _ = model(imgs)
            preds = clf(feats)
            loss = F.cross_entropy(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # 保存权重
    linear_probe_path = student_ckpt.replace(".pth", "_linear_probe.pth")
    torch.save(clf.state_dict(), linear_probe_path)
    # 验证
    clf.eval(); all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in valloader:
            imgs, labels = imgs.to(device), labels.to(device)
            feats, _, _, _ = model(imgs)
            logits = clf(feats)
            all_preds.append(logits.cpu())
            all_labels.append(labels.cpu())
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    top1 = (preds.argmax(1) == labels).float().mean().item()
    top5 = (preds.topk(5, 1)[1] == labels.unsqueeze(1)).any(1).float().mean().item()
    print(f"[LINEAR PROBE] Top-1: {top1:.4f}, Top-5: {top5:.4f}")
    return top1, top5

# ====== 2. Caltech101 Folder Dataset ======
class Caltech101Folder(Dataset):
    def __init__(self, root, class_names, transform=None):
        self.samples = []
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        for cname in class_names:
            cdir = os.path.join(root, cname)
            if not os.path.isdir(cdir):
                continue
            for f in os.listdir(cdir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(cdir, f), self.class_to_idx[cname]))
    def __getitem__(self, idx):
        img, label = self.samples[idx]
        img = Image.open(img).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
    def __len__(self):
        return len(self.samples)

def zero_shot_clip_transfer(model_ckpt, eval_dataset_root="/home/xuyingl/iccv_workshop/caltech101", proj_dim=512):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_transform = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    cat_dir = os.path.join(eval_dataset_root, "101_ObjectCategories")
    class_names = sorted([d for d in os.listdir(cat_dir) if os.path.isdir(os.path.join(cat_dir, d))])
    text_prompts = [f"a photo of a {name}" for name in class_names]
    clip_model, _ = clip.load("ViT-B/32", device=device)
    with torch.no_grad():
        text_tokens = clip.tokenize(text_prompts).to(device)
        text_features = clip_model.encode_text(text_tokens).float()
        text_features = F.normalize(text_features, dim=1)
    student = SimCLR_MAE_Patch(models.resnet18(weights=None), use_mae=False, proj_dim=proj_dim).to(device)
    student.load_state_dict(torch.load(model_ckpt, map_location=device), strict=False)
    student.eval()
    testset = Caltech101Folder(cat_dir, class_names, data_transform)
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)
    correct = 0
    total = 0
    for imgs, labels in tqdm(testloader, desc="Zero-shot eval"):
        imgs = imgs.to(device)
        with torch.no_grad():
            feats, _, _, _ = student(imgs)
            feats = F.normalize(feats, dim=1)
            sim = feats @ text_features.T
            pred = sim.argmax(1).cpu()
        correct += (pred == labels).sum().item()
        total += len(labels)
    acc = correct / total
    print(f"[ZERO-SHOT Caltech101] Top-1 acc: {acc:.4f}")
    return acc

# ====== 3. CIFAR-10线性头训练&自动适配所有CIFAR-10-C扰动shape ======
def train_cifar10_linear_head(student_ckpt, proj_dim=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimCLR_MAE_Patch(models.resnet18(weights=None), use_mae=False, proj_dim=proj_dim)
    model.load_state_dict(torch.load(student_ckpt, map_location=device), strict=False)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    num_classes = 10
    clf = nn.Linear(proj_dim, num_classes).to(device)
    optimizer = optim.Adam(clf.parameters(), lr=1e-2)
    tfm = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    from torchvision import datasets as tv_datasets
    cifar10_train = tv_datasets.CIFAR10(root='./cifar10_data', train=True, download=True, transform=tfm)
    cifar10_test  = tv_datasets.CIFAR10(root='./cifar10_data', train=False, download=True, transform=tfm)
    trainloader = DataLoader(cifar10_train, batch_size=64, shuffle=True, num_workers=4)
    testloader  = DataLoader(cifar10_test,  batch_size=64, shuffle=False, num_workers=4)
    for epoch in range(10):
        model.eval(); clf.train()
        for imgs, labels in tqdm(trainloader, desc=f"CIFAR10 linear Epoch {epoch+1}"):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                feats, _, _, _ = model(imgs)
            preds = clf(feats)
            loss = F.cross_entropy(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # 保存头权重
    save_path = student_ckpt.replace('.pth', '_cifar10_linear.pth')
    torch.save(clf.state_dict(), save_path)
    # 测试
    clf.eval(); all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in testloader:
            imgs, labels = imgs.to(device), labels.to(device)
            feats, _, _, _ = model(imgs)
            logits = clf(feats)
            all_preds.append(logits.cpu())
            all_labels.append(labels.cpu())
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    top1 = (preds.argmax(1) == labels).float().mean().item()
    print(f"[CIFAR10 Linear Probe] Test Top-1: {top1:.4f}, 权重已存: {save_path}")
    return top1, save_path

from tqdm import tqdm

def cifar10c_robustness(student_ckpt, proj_dim=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = SimCLR_MAE_Patch(models.resnet18(weights=None), use_mae=False, proj_dim=proj_dim).to(device)
    student.load_state_dict(torch.load(student_ckpt, map_location=device), strict=False)
    student.eval()
    cifar10_head_path = student_ckpt.replace(".pth", "_cifar10_linear.pth")
    linear_head = nn.Linear(proj_dim, 10).to(device)
    if os.path.exists(cifar10_head_path):
        linear_head.load_state_dict(torch.load(cifar10_head_path, map_location=device), strict=True)
        print(f"[CIFAR10C] 用CIFAR10线性头: {cifar10_head_path}")
    else:
        print("[CIFAR10C] 没有cifar10_linear头，用随机头")
    norm = transforms.Normalize([0.5]*3, [0.5]*3)
    corrs = sorted([f for f in os.listdir("cifar10c") if f.endswith('.npy') and 'label' not in f])
    all_acc = []
    for corr in tqdm(corrs, desc="CIFAR-10-C"):
        arr = np.load(f"cifar10c/{corr}")
        # shape自适应修正
        if arr.ndim == 4 and arr.shape[-1] == 3:  # [N, H, W, C]
            arr = arr.transpose(0,3,1,2)
        elif arr.ndim == 3:  # 灰度
            arr = np.repeat(arr[:,None,:,:], 3, axis=1)
        elif arr.ndim == 4 and arr.shape[1] == 3:  # [N, C, H, W]
            pass
        else:
            raise ValueError(f"未知shape: {arr.shape} in {corr}")
        arr = torch.tensor(arr).float() / 255
        correct, total = 0, 0
        for i in range(0, len(arr), 128):
            imgs = arr[i:i+128].to(device)
            imgs = F.interpolate(imgs, (224,224))
            imgs = norm(imgs)
            with torch.no_grad():
                feats, _, _, _ = student(imgs)
                logits = linear_head(feats)
                pred = logits.argmax(1)
            labels = torch.arange(i, i+len(imgs)) % 10
            correct += (pred.cpu() == labels).sum().item()
            total += len(imgs)
        acc = correct / total if total > 0 else 0.0
        print(f"[CIFAR10C] {corr}: {acc:.4f}")
        all_acc.append(acc)
    robust_score = np.mean(all_acc)
    print(f"[CIFAR10C] Mean Robust Accuracy: {robust_score:.4f}")
    return robust_score

# ====== 4. 参数统计/推理速度 ======
def model_stat_and_speed(model_ckpt, proj_dim=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimCLR_MAE_Patch(models.resnet18(weights=None), use_mae=False, proj_dim=proj_dim).to(device)
    model.load_state_dict(torch.load(model_ckpt, map_location=device), strict=False)
    model.eval()
    dummy = torch.randn(8, 3, 224, 224).to(device)
    total = sum(p.numel() for p in model.parameters())
    t0 = time.time()
    for _ in range(10):
        with torch.no_grad():
            model(dummy)
    t1 = time.time()
    infer_speed = (t1-t0)/10
    print(f"[MODEL STAT] Params: {total}, Inference Speed (8x224x224): {infer_speed:.4f} s")
    return total, infer_speed

# ====== 5. 消融对比全流程 ======
all_methods = {
    "SimCLR-RealOnly": "simclr_realonly.pth",
    "DiffMix-SSL": "diffmix_mae_patch_final.pth",
    "Distill-NoCompress": "student_distill_cosine_l1_final.pth",
    "Distill+Quant": "student_distill_cosine_quantized.pth"
}
result_table = []
for method, ckpt in all_methods.items():
    print(f"\n=== Evaluating: {method} ===")
    top1, top5 = linear_probe_evaluate(ckpt)
    cifar10_top1, _ = train_cifar10_linear_head(ckpt)
    zs_acc = zero_shot_clip_transfer(ckpt)
    robust = cifar10c_robustness(ckpt)
    param, spd = model_stat_and_speed(ckpt)
    result_table.append({
        "Method": method,
        "Params": param,
        "Top1": top1,
        "Top5": top5,
        "CIFAR10": cifar10_top1,
        "ZeroShot": zs_acc,
        "Robust": robust,
        "Speed(s)": spd
    })
df = pd.DataFrame(result_table)
df.to_csv("day4_result.csv", index=False)
print(df)
fig, ax = plt.subplots(figsize=(10,3))
ax.axis('off')
table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc='center',
    loc='center'
)
plt.tight_layout()
plt.savefig("day4_result_table.png")
print("Table and csv saved.")
