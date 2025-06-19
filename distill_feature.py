import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import clip

# --------- 非线性头定义（建议直接写进 diffmix_mae_patch_train.py）---------
class StrongProjHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.ReLU(),
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

# ---- 用你的 backbone + 新头 -----
from diffmix_mae_patch_train import SimCLR_MAE_Patch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 1. 数据集 ---
class SingleImageDataset(Dataset):
    def __init__(self, root, transform):
        self.paths = []
        for cname in sorted(os.listdir(root)):
            cdir = os.path.join(root, cname)
            for fname in os.listdir(cdir):
                if fname.lower().endswith(('.jpg', '.png')):
                    self.paths.append(os.path.join(cdir, fname))
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        return img

# --- 2. 教师特征 ---
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()
def extract_teacher_feature(img_tensor):
    img_pil = transforms.ToPILImage()(img_tensor.cpu())
    img_clip = clip_preprocess(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = clip_model.encode_image(img_clip)
    return feat.squeeze(0).float()

# --- 3. 学生模型 ---
student = SimCLR_MAE_Patch(models.resnet18(weights=None), use_mae=False).to(device)
# 换投影头
student.proj_head = StrongProjHead(student.embed_dim, 512).to(device)
student.load_state_dict(torch.load("diffmix_mae_patch_final.pth"), strict=False)

# 解冻 layer4 + 投影头，其它冻结
for name, param in student.named_parameters():
    if name.startswith("proj_head") or "layer4" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# --- 4. Dataloader ---
transform = transforms.Compose([
    transforms.Resize(224), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
])
dataset = SingleImageDataset('./data/all', transform)
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

# --- 5. 训练：Cosine + L2 (可选) ---
optimizer = optim.Adam(filter(lambda p: p.requires_grad, student.parameters()), lr=1e-3)
cos_criterion = nn.CosineEmbeddingLoss()
l2_criterion = nn.MSELoss()
l1_lambda = 0.001
use_mse_loss = True  # 改 True/False 切换

for epoch in range(50):
    student.train()
    epoch_loss = 0
    for imgs in tqdm(loader):
        imgs = imgs.to(device)
        with torch.no_grad():
            t_feats = torch.stack([extract_teacher_feature(img) for img in imgs])
        s_feats, _, _, _ = student(imgs)
        s_feats = s_feats.float()
        t_feats = t_feats.float()
        target = torch.ones(imgs.shape[0], device=imgs.device)
        cos_loss = cos_criterion(s_feats, t_feats, target)
        # 混合 L2（可选）
        if use_mse_loss:
            mse_loss = l2_criterion(s_feats, t_feats)
            loss = cos_loss + 0.5 * mse_loss
        else:
            loss = cos_loss
        l1_reg = sum(p.abs().sum() for p in student.proj_head.parameters())
        total_loss = loss + l1_lambda * l1_reg
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss.item()
    print(f'Epoch [{epoch+1}/50] Distill Loss: {epoch_loss/len(loader):.4f}')
    if (epoch+1)%10 == 0:
        torch.save(student.state_dict(), f"student_stronghead_l1_epoch{epoch+1}.pth")

torch.save(student.state_dict(), "student_stronghead_l1_final.pth")
print("已保存模型 student_stronghead_l1_final.pth")

# -------- 6. 8-bit 动态量化 ---------
import torch.quantization as tq
student.eval()
quantized_model = tq.quantize_dynamic(student, {nn.Linear}, dtype=torch.qint8)
torch.save(quantized_model.state_dict(), "student_stronghead_quantized.pth")
print("已保存8-bit量化模型 student_stronghead_quantized.pth")
