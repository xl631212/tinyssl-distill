import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np

class RealOnlyDataset(Dataset):
    def __init__(self, root, num_per_class=20, transform=None):
        self.imgs = []
        for cname in sorted(os.listdir(root)):
            cdir = os.path.join(root, cname)
            if not os.path.isdir(cdir): continue
            real_imgs = [f for f in os.listdir(cdir) if f.lower().endswith(('.jpg', '.png')) and '_var' not in f]
            random.shuffle(real_imgs)
            real_imgs = real_imgs[:num_per_class]
            self.imgs += [os.path.join(cdir, f) for f in real_imgs]
        self.transform = transform
    def __len__(self): return len(self.imgs)
    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

class SimCLR_Linear(nn.Module):
    def __init__(self, backbone, proj_dim=512):
        super().__init__()
        self.encoder = backbone
        self.encoder.fc = nn.Identity()
        embed_dim = self.encoder.inplanes if hasattr(self.encoder, "inplanes") else 512
        self.head = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
    def forward(self, x):
        feat = self.encoder(x)
        z = self.head(feat)
        return z

def simclr_loss(z1, z2, temperature=0.5):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    N = z1.size(0)
    labels = torch.arange(N, device=z1.device)
    sims = (z1 @ z2.T) / temperature
    loss = nn.CrossEntropyLoss()(sims, labels)
    return loss

def train_simclr_realonly(data_root='./data/all', out_ckpt="simclr_realonly.pth"):
    transform = transforms.Compose([
        transforms.Resize(224), transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    dataset = RealOnlyDataset(data_root, num_per_class=20, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimCLR_Linear(models.resnet18(weights=None), proj_dim=512).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(50):
        model.train()
        losses = []
        for imgs in loader:
            imgs = imgs.to(device)
            # 生成两组增强视图
            imgs2 = torch.stack([transform(transforms.ToPILImage()(img.cpu())) for img in imgs])
            imgs2 = imgs2.to(device)
            z1 = model(imgs)
            z2 = model(imgs2)
            loss = simclr_loss(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f'Epoch {epoch+1}/50, SimCLR RealOnly Loss: {np.mean(losses):.4f}')
    torch.save(model.state_dict(), out_ckpt)
    print(f"Saved: {out_ckpt}")

if __name__ == '__main__':
    train_simclr_realonly()
