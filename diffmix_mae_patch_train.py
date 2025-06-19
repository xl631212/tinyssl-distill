import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# ========== 1. Patchify/Unpatchify ==========

def patchify(imgs, patch_size=16):
    B, C, H, W = imgs.shape
    assert H % patch_size == 0 and W % patch_size == 0
    h = H // patch_size
    w = W // patch_size
    patches = imgs.reshape(B, C, h, patch_size, w, patch_size)
    patches = patches.permute(0,2,4,1,3,5).reshape(B, h*w, C*patch_size*patch_size)
    return patches

def unpatchify(patches, patch_size=16, img_size=224):
    B, N, D = patches.shape
    h = w = img_size // patch_size
    patches = patches.reshape(B, h, w, 3, patch_size, patch_size)
    patches = patches.permute(0,3,1,4,2,5).reshape(B, 3, img_size, img_size)
    return patches

# ========== 2. DiffMix Pair Dataset ==========

class DiffMixPairDataset(Dataset):
    def __init__(self, root, transform=None):
        self.pairs = []
        self.negatives = []
        self.transform = transform
        for cname in sorted(os.listdir(root)):
            cdir = os.path.join(root, cname)
            if not os.path.isdir(cdir): continue
            imgs = sorted([f for f in os.listdir(cdir) if f.lower().endswith(('.jpg', '.png'))])
            real_imgs = [f for f in imgs if '_var' not in f]
            synth_imgs = [f for f in imgs if '_var' in f]
            for real in real_imgs:
                related_synth = [f for f in synth_imgs if f.startswith(real.replace('.jpg','').replace('.png',''))]
                if not related_synth: continue
                synth = random.choice(related_synth)
                self.pairs.append((os.path.join(cdir, real), os.path.join(cdir, synth)))
            self.negatives += [os.path.join(cdir, f) for f in imgs]

    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        real_path, synth_path = self.pairs[idx]
        img1 = Image.open(real_path).convert("RGB")
        img2 = Image.open(synth_path).convert("RGB")
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        while True:
            neg_path = random.choice(self.negatives)
            if neg_path != real_path and neg_path != synth_path:
                break
        img_neg = Image.open(neg_path).convert("RGB")
        if self.transform:
            img_neg = self.transform(img_neg)
        return img1, img2, img_neg

# ========== 3. Transformer MAE Patch解码头 ==========

class MAETransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim*mlp_ratio), dim)
        )
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class MAEDecoder(nn.Module):
    def __init__(self, embed_dim, num_patches, patch_dim, num_layers=4, num_heads=8, decoder_embed_dim=256):
        super().__init__()
        self.proj = nn.Linear(embed_dim, decoder_embed_dim, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.blocks = nn.ModuleList([
            MAETransformerBlock(decoder_embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(decoder_embed_dim)
        self.pred = nn.Linear(decoder_embed_dim, patch_dim, bias=True)
        self.num_patches = num_patches
        nn.init.trunc_normal_(self.mask_token, std=.02)

    def forward(self, x, mask):
        B = x.shape[0]
        device = x.device
        decoder_embed = self.proj(x)
        out = torch.zeros(B, self.num_patches, decoder_embed.shape[-1], device=device)
        for b in range(B):
            out[b, ~mask[b]] = decoder_embed[b, ~mask[b]]
            out[b, mask[b]] = self.mask_token
        for blk in self.blocks:
            out = blk(out)
        out = self.norm(out)
        pred = self.pred(out)
        return pred

# ========== 强化版投影头（建议与蒸馏脚本一致）==========
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

# ========== 4. 主模型 ==========

class SimCLR_MAE_Patch(nn.Module):
    def __init__(self, backbone, img_size=224, patch_size=16, proj_dim=512, use_mae=True):
        super().__init__()
        self.encoder = backbone
        self.encoder.fc = nn.Identity()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size
        self.embed_dim = self.encoder.inplanes if hasattr(self.encoder, "inplanes") else 512
        # 更强非线性头（和蒸馏阶段一致！）
        self.proj_head = StrongProjHead(self.embed_dim, proj_dim)
        self.use_mae = use_mae
        if use_mae:
            self.mae_decoder = MAEDecoder(
                self.embed_dim, self.num_patches, self.patch_dim,
                num_layers=4, num_heads=8, decoder_embed_dim=256
            )

    def forward(self, x, mask=None, for_mae=False):
        B = x.size(0)
        patches = patchify(x, patch_size=self.patch_size)
        imgs_patch = patches.view(-1, 3, self.patch_size, self.patch_size)
        feats_all = self.encoder(imgs_patch.to(x.device))
        feats = feats_all.view(B, self.num_patches, -1)
        z = self.proj_head(feats.mean(dim=1))
        if self.use_mae and for_mae and mask is not None:
            pred = self.mae_decoder(feats, mask)
            return z, pred, mask, patches
        return z, None, None, None

# ========== 5. 对比损失 ==========

def info_nce_loss(z1, z2, z_neg, temperature=0.5):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    z_neg = nn.functional.normalize(z_neg, dim=1)
    pos_sim = (z1 * z2).sum(-1) / temperature
    neg_sim = torch.matmul(z1, z_neg.T) / temperature
    pos_loss = -pos_sim.mean()
    neg_loss = torch.logsumexp(neg_sim, dim=1).mean()
    return pos_loss + neg_loss

# ========== 6. 数据增强 ==========

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ========== 7. 训练主流程 ==========

def train():
    dataset = DiffMixPairDataset('./data/all', transform=transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimCLR_MAE_Patch(models.resnet18(pretrained=False), use_mae=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 100

    # 冻结除 layer4/proj_head 以外层
    for name, param in model.named_parameters():
        if ("layer4" in name) or ("proj_head" in name):
            param.requires_grad = True
        else:
            param.requires_grad = False

    for epoch in range(epochs):
        model.train()
        losses = []
        for img1, img2, img_neg in loader:
            img1, img2, img_neg = img1.to(device), img2.to(device), img_neg.to(device)
            optimizer.zero_grad()
            if epoch < 70:
                z1, _, _, _ = model(img1)
                z2, _, _, _ = model(img2)
                zneg, _, _, _ = model(img_neg)
                loss = info_nce_loss(z1, z2, zneg)
            else:
                # Patch掩码（每张图片mask掉60% patch）
                B, C, H, W = img1.shape
                patch_size = 16
                num_patches = (H // patch_size) ** 2
                mask = torch.zeros(B, num_patches, dtype=torch.bool, device=img1.device)
                for b in range(B):
                    mask_idx = torch.randperm(num_patches)[:int(num_patches * 0.6)]
                    mask[b, mask_idx] = True
                z1, recon, mask_out, patches = model(img1, mask=mask, for_mae=True)
                rec_loss = nn.functional.mse_loss(recon[mask], patches[mask])
                z2, _, _, _ = model(img2)
                zneg, _, _, _ = model(img_neg)
                contrast_loss = info_nce_loss(z1, z2, zneg)
                loss = contrast_loss + 0.5 * rec_loss
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        avg_loss = sum(losses) / len(losses)
        print(f'Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}')
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f'diffmix_mae_patch_epoch{epoch+1}.pth')
    torch.save(model.state_dict(), 'diffmix_mae_patch_final.pth')
    print('训练完成！')

if __name__ == '__main__':
    train()
