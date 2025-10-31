import os
import random
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models as tv_models
import albumentations as A
import kornia


# =========================================
# UTILITY FUNCTIONS
# =========================================
def read_img(path):
    img = cv2.imread(str(path))  # Reads as BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    return img

def frequency_split_rgb(rgb_tensor, kernel_size=11, sigma=5.0):
    """Split RGB image into low and high frequency components"""
    k = kernel_size if kernel_size%2==1 else kernel_size+1
    rgb_low = kornia.filters.gaussian_blur2d(rgb_tensor, (k,k), (sigma, sigma))
    H = rgb_tensor - rgb_low
    return rgb_low, H

def blend_with_mask(original, prediction, mask, epsilon=1e-6):
    """Only apply prediction in masked regions, keep original elsewhere"""
    mask_expanded = mask.expand_as(prediction)  # Explicit expansion
    return mask_expanded * prediction + (1 - mask_expanded) * original

# =========================================
# DATASET
# =========================================
class PairedClothDatasetWithMask(Dataset):
    def __init__(self, wrinkled_dir, clean_dir, mask_dir, size=512, augment=True):
        self.wr_files = sorted(glob(os.path.join(wrinkled_dir, '*')))
        self.cl_files = sorted(glob(os.path.join(clean_dir, '*')))
        self.mask_files = sorted(glob(os.path.join(mask_dir, '*')))
        assert len(self.wr_files) == len(self.cl_files) == len(self.mask_files)
        self.size = size
        self.augment = augment
        self.alb_aug = A.Compose([
            A.RandomCrop(size, size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.4),
        ], additional_targets={'image0': 'image', 'mask0': 'mask'})  # âœ… Correct mapping

    def __len__(self):
        return len(self.wr_files)

    def __getitem__(self, idx):
        wr = read_img(self.wr_files[idx])
        cl = read_img(self.cl_files[idx])
        mask = cv2.imread(self.mask_files[idx], cv2.IMREAD_GRAYSCALE)
        
        H,W,_ = wr.shape
        if H < self.size or W < self.size:
            new_w = max(self.size, W)
            new_h = max(self.size, H)
            wr = cv2.resize(wr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            cl = cv2.resize(cl, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if self.augment:
            aug = self.alb_aug(image=wr, image0=cl, mask0=mask)
            wr = aug['image']
            cl = aug['image0']
            mask = aug['mask0']
        else:
            h,w,_ = wr.shape
            sy = (h - self.size)//2
            sx = (w - self.size)//2
            wr = wr[sy:sy+self.size, sx:sx+self.size]
            cl = cl[sy:sy+self.size, sx:sx+self.size]
            mask = mask[sy:sy+self.size, sx:sx+self.size]

        # Normalize RGB to [0, 1]
        wr_norm = (wr.astype(np.float32) / 255.0).transpose(2, 0, 1)
        cl_norm = (cl.astype(np.float32) / 255.0).transpose(2, 0, 1)
        mask_t = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0)
        
        return torch.from_numpy(wr_norm).float(), torch.from_numpy(cl_norm).float(), mask_t

class ResNetAutoencoder(nn.Module):
    """
    ResNet101 encoder -> simple ConvTranspose decoder.
    - in_channels: number of input channels (use 4 for RGB+mask)
    - out_channels: number of output channels (3 for RGB residual)
    """
    def __init__(self, in_channels=4, out_channels=3, pretrained=True):
        super().__init__()
        # Load resnet101
        resnet = tv_models.resnet101(pretrained=pretrained)
        # adapt first conv to accept in_channels
        if in_channels != 3:
            self.encoder_conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # copy weights for first 3 channels if pretrained
            if pretrained:
                with torch.no_grad():
                    self.encoder_conv1.weight[:, :3, :, :].copy_(resnet.conv1.weight)
                    if in_channels > 3:
                        # initialize extra channels
                        nn.init.kaiming_normal_(self.encoder_conv1.weight[:, 3:, :, :])
        else:
            self.encoder_conv1 = resnet.conv1

        # use rest of resnet layers
        self.encoder_bn1 = resnet.bn1
        self.encoder_relu = resnet.relu
        self.encoder_maxpool = resnet.maxpool
        self.encoder_layer1 = resnet.layer1  # 64
        self.encoder_layer2 = resnet.layer2  # 128
        self.encoder_layer3 = resnet.layer3  # 256
        self.encoder_layer4 = resnet.layer4  # 512

        # freeze encoder layers
        for m in [
            self.encoder_conv1,
            self.encoder_bn1,
            self.encoder_layer1,
            self.encoder_layer2,
            self.encoder_layer3,
            self.encoder_layer4,
        ]:
            for param in m.parameters():
                param.requires_grad = False



        # Decoder: progressively upsample back to input resolution
        # ResNet downscales by factor 32 => we need 5 upsample stages (x2 each)
        self.dec_t1 = nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1)  # /16
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.dec_t2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # /8
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.dec_t3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # /4
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.dec_t4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)    # /2
        self.dec_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # final upsample to original resolution (if needed) and output conv
        self.dec_t5 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)    # /1
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x = self.encoder_conv1(x)
        x = self.encoder_bn1(x)
        x = self.encoder_relu(x)
        x = self.encoder_maxpool(x)

        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)
        x = self.encoder_layer3(x)
        x = self.encoder_layer4(x)  # shape: (B, 512, H/32, W/32)

        # Decoder
        x = self.dec_t1(x); x = self.dec_conv1(x)
        x = self.dec_t2(x); x = self.dec_conv2(x)
        x = self.dec_t3(x); x = self.dec_conv3(x)
        x = self.dec_t4(x); x = self.dec_conv4(x)
        x = self.dec_t5(x)
        out = self.final_conv(x)
        # no activation â€” your training script expects raw residuals
        return out

# =========================================
# LOSS MODULES
# =========================================
def gram_matrix(feat):
    b, c, h, w = feat.size()
    f = feat.view(b, c, h * w)
    G = torch.bmm(f, f.transpose(1, 2))
    return G / (c * h * w)

# Gram Matrix for Style Loss
class GramStyleLoss(nn.Module):
    """Texture preservation using Gram matrices"""
    def __init__(self, layers=('3', '8', '15', '22'), device='cuda'):
        super().__init__()
        vgg = tv_models.vgg19(pretrained=True).features.to(device).eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.layer_idxs = [int(l) for l in layers]
        self.criterion = nn.L1Loss()
        self.device = device
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def extract_feats(self, x):
        feats = []
        h = x
        for idx, layer in enumerate(self.vgg):
            h = layer(h)
            if idx in self.layer_idxs:
                feats.append(h)
        return feats

    def forward(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Normalize inputs for VGG
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        y = (y - self.mean.to(y.device)) / self.std.to(y.device)

        fx = self.extract_feats(x)
        fy = self.extract_feats(y)
        loss = 0.0
        for a, b in zip(fx, fy):
            Ga = gram_matrix(a)
            Gb = gram_matrix(b)
            loss += self.criterion(Ga, Gb)
        return loss


class VGGPerceptual(nn.Module):
    """Standard VGG perceptual loss"""
    def __init__(self, layers=['relu2_2','relu3_3'], device='cuda'):
        super().__init__()
        vgg = tv_models.vgg16(pretrained=True).features.eval()
        for p in vgg.parameters(): 
            p.requires_grad = False
        self.vgg = vgg.to(device)
        self.device = device
        self.layer_map = {
            'relu1_2': 3, 'relu2_2': 8, 'relu3_3': 15, 'relu4_3': 22
        }
        self.selected_idxs = [self.layer_map[l] for l in layers]
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Normalize inputs for VGG
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        y = (y - self.mean.to(y.device)) / self.std.to(y.device)

        def extract(inp):
            feats = []
            xi = inp
            for idx, layer in enumerate(self.vgg):
                xi = layer(xi)
                if idx in self.selected_idxs:
                    feats.append(xi)
            return feats
        fx = extract(x)
        fy = extract(y)
        loss = 0.0
        for a,b in zip(fx,fy):
            loss += F.l1_loss(a,b)
        return loss


def validate_model(dataloader, resnet_low, resnet_high, perc, style_loss_module, device,
                   w_residual_low, w_residual_high, w_rgb_final, w_perc, w_style, w_ssim):

    resnet_low.eval()
    resnet_high.eval()
    
    pbar = tqdm(dataloader, desc="Validation")
    epoch_loss_val = 0.0
    batch_count = 0
    with torch.no_grad():
        for wr_rgb, cl_rgb, mask in pbar:
            wr_rgb = wr_rgb.to(device)  # (B, 3, H, W)
            cl_rgb = cl_rgb.to(device)  # (B, 3, H, W)
            mask = mask.to(device)      # (B, 1, H, W)

            # Frequency split on BOTH wrinkled and clean RGB
            rgb_low_wr, H_wr = frequency_split_rgb(wr_rgb, kernel_size=11, sigma=5.0)
            rgb_low_cl, H_cl = frequency_split_rgb(cl_rgb, kernel_size=11, sigma=5.0)

            # Smooth mask
            mask_smooth = kornia.filters.gaussian_blur2d(mask, (9, 9), (3.0, 3.0))
            mask_smooth = mask_smooth / (mask_smooth.max() + 1e-6)
            mask_smooth = torch.clamp(mask_smooth, 0.0, 1.0)

            # ============================================================
            # RESIDUAL LEARNING: Networks predict corrections (deltas)
            # ============================================================
            
            # Ground truth residuals
            residual_low_gt = rgb_low_cl - rgb_low_wr   # What to add to wrinkled low freq
            residual_high_gt = H_cl - H_wr              # What to add to wrinkled high freq
            
            # Low frequency network - predicts RESIDUAL
            in_low = torch.cat([rgb_low_wr, mask_smooth], dim=1)  # (B, 4, H, W)
            residual_low_pred =  resnet_low(in_low)  # (B, 3, H, W)
            rgb_low_pred = rgb_low_wr + residual_low_pred  # Apply correction
            
            # High frequency network - predicts RESIDUAL
            in_high = torch.cat([H_wr, mask_smooth], dim=1)  # (B, 4, H, W)
            residual_high_pred = resnet_high(in_high)  # (B, 3, H, W)
            H_pred = H_wr + residual_high_pred  # Apply correction
            
            # ============================================================
            # Blend each component with mask BEFORE combining
            # ============================================================
            rgb_low_blended = blend_with_mask(rgb_low_wr, rgb_low_pred, mask_smooth)
            H_blended = blend_with_mask(H_wr, H_pred, mask_smooth)
            
            # Reconstruct RGB by adding the two components
            rgb_pred = rgb_low_blended + H_blended
            
            # Clamp to valid range
            rgb_pred = torch.clamp(rgb_pred, 0.0, 1.0)

            # ===================================================
            # COMPUTE LOSSES - Supervise the RESIDUALS directly!
            # ===================================================
            
            mask_count = mask_smooth.sum() + 1e-6
                    
            # Direct supervision on low frequency RESIDUAL
            loss_residual_low = (mask_smooth * (residual_low_pred - residual_low_gt).abs()).sum() / mask_count
            
            # Direct supervision on high frequency RESIDUAL
            loss_residual_high = (mask_smooth * (residual_high_pred - residual_high_gt).abs()).sum() / mask_count
            
            # Loss on final reconstructed RGB
            loss_rgb_final = (mask_smooth * (rgb_pred - cl_rgb).abs()).sum() / mask_count
            
            # Perceptual losses
            loss_perc = perc(rgb_pred, cl_rgb)
            loss_style = style_loss_module(rgb_pred, cl_rgb)
            loss_ssim = kornia.losses.ssim_loss(rgb_pred, cl_rgb, window_size=11)

            # ===================================================
            # TOTAL LOSS
            # ===================================================
            loss = (w_residual_low * loss_residual_low +
                    w_residual_high * loss_residual_high +
                    w_rgb_final * loss_rgb_final +
                    w_perc * loss_perc +
                    w_style * loss_style +
                    w_ssim * loss_ssim)

            epoch_loss_val += loss.item()
            batch_count += 1

            pbar.set_postfix({
                'val_loss': f'{loss.item():.4f}',
                'res_low': f'{loss_residual_low.item():.3f}',
                'res_high': f'{loss_residual_high.item():.3f}',
                'perc': f'{loss_perc.item():.3f}',
                'style': f'{loss_style.item():.3f}'
            })


    avg_loss_val = epoch_loss_val / batch_count

    return avg_loss_val




# =========================================
# TRAINING FUNCTION
# =========================================
def train_dual_resnet():
    """Train two separate U-Nets predicting RESIDUALS for low and high frequency in RGB"""
    print(f"\n{'='*70}")
    print(f"ðŸš€ Starting Dual U-Net Training - RESIDUAL LEARNING (RGB)")
    print(f"{'='*70}\n")

    dataset_root = '/home/ml2/Documents/experiments_6_oct/wrinkle_removal/frequency_seperation/best_working/test_lama/train'
    wrinkled_dir = os.path.join(dataset_root, 'image')
    clean_dir = os.path.join(dataset_root, 'target')
    mask_dir = os.path.join(dataset_root, 'mask')
    
    dataset_root_val = '/home/ml2/Documents/experiments_6_oct/wrinkle_removal/frequency_seperation/best_working/test_lama/val'
    wrinkled_dir_val = os.path.join(dataset_root_val, 'image')
    clean_dir_val = os.path.join(dataset_root_val, 'target')
    mask_dir_val = os.path.join(dataset_root_val, 'mask')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    size = 1024
    batch_size = 1
    epochs = 1000
    checkpoint_dir = 'checkpoints_dual_unet_residual_rgb_modified'
    os.makedirs(checkpoint_dir, exist_ok=True)

    ds = PairedClothDatasetWithMask(wrinkled_dir, clean_dir, mask_dir, size=size, augment=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)


    val_dataset = PairedClothDatasetWithMask(wrinkled_dir_val, clean_dir_val, mask_dir_val, size=size, augment=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)

    # Two separate U-Nets - predict RESIDUALS (input: 3 RGB + 1 mask = 4 channels)
    resnet_low = ResNetAutoencoder(in_channels=4, out_channels=3).to(device)  # RGB input + mask
    resnet_high = ResNetAutoencoder(in_channels=4, out_channels=3).to(device)
    
    def count_parameters(model, name="Model"):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        total_m = total / 1e6
        trainable_m = trainable / 1e6
        non_trainable_m = (total - trainable) / 1e6

        print(f"\n{name} Parameters:")
        print(f"  Total params: {total_m:.2f}M")
        print(f"  Trainable params: {trainable_m:.2f}M")
        print(f"  Non-trainable params: {non_trainable_m:.2f}M")

        return total, trainable


    # Count parameters for each
    count_parameters(resnet_low, "ResNet101Autoencoder (Low-Freq)")
    count_parameters(resnet_high, "ResNet101Autoencoder (High-Freq)")

    # Combined totals
    total_low, train_low = count_parameters(resnet_low)
    total_high, train_high = count_parameters(resnet_high)

    total_combined_m = (total_low + total_high) / 1e6
    train_combined_m = (train_low + train_high) / 1e6

    print(f"\nCombined:")
    print(f"  Total parameters: {total_combined_m:.2f}M")
    print(f"  Trainable parameters: {train_combined_m:.2f}M")

    optG = torch.optim.Adam(
        list( resnet_low.parameters()) + list( resnet_high.parameters()), 
        lr=1e-4, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optG, T_max=epochs)

    # Initialize loss modules
    perc = VGGPerceptual(device=device)
    style_loss_module = GramStyleLoss(device=device)
    
    # Loss weights
    w_residual_low = 1.0    # Loss on low freq residual
    w_residual_high = 1.0   # Loss on high freq residual
    w_rgb_final = 2.0       # Final reconstruction
    w_perc = 0.3
    w_style = 0.05
    w_ssim = 0.5

    best_loss = float('inf')

    for epoch in range(epochs):
        resnet_low.train()
        resnet_high.train()
        
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0.0
        batch_count = 0

        for wr_rgb, cl_rgb, mask in pbar:
            wr_rgb = wr_rgb.to(device)  # (B, 3, H, W)
            cl_rgb = cl_rgb.to(device)  # (B, 3, H, W)
            mask = mask.to(device)      # (B, 1, H, W)

            # Frequency split on BOTH wrinkled and clean RGB
            rgb_low_wr, H_wr = frequency_split_rgb(wr_rgb, kernel_size=11, sigma=5.0)
            rgb_low_cl, H_cl = frequency_split_rgb(cl_rgb, kernel_size=11, sigma=5.0)

            # Smooth mask
            mask_smooth = kornia.filters.gaussian_blur2d(mask, (9, 9), (3.0, 3.0))
            mask_smooth = mask_smooth / (mask_smooth.max() + 1e-6)
            mask_smooth = torch.clamp(mask_smooth, 0.0, 1.0)

            # ============================================================
            # RESIDUAL LEARNING: Networks predict corrections (deltas)
            # ============================================================
            
            # Ground truth residuals
            residual_low_gt = rgb_low_cl - rgb_low_wr   # What to add to wrinkled low freq
            residual_high_gt = H_cl - H_wr              # What to add to wrinkled high freq
            
            # Low frequency network - predicts RESIDUAL
            in_low = torch.cat([rgb_low_wr, mask_smooth], dim=1)  # (B, 4, H, W)
            residual_low_pred =  resnet_low(in_low)  # (B, 3, H, W)
            rgb_low_pred = rgb_low_wr + residual_low_pred  # Apply correction
            
            # High frequency network - predicts RESIDUAL
            in_high = torch.cat([H_wr, mask_smooth], dim=1)  # (B, 4, H, W)
            residual_high_pred =  resnet_high(in_high)  # (B, 3, H, W)
            H_pred = H_wr + residual_high_pred  # Apply correction
            
            # ============================================================
            # Blend each component with mask BEFORE combining
            # ============================================================
            rgb_low_blended = blend_with_mask(rgb_low_wr, rgb_low_pred, mask_smooth)
            H_blended = blend_with_mask(H_wr, H_pred, mask_smooth)
            
            # Reconstruct RGB by adding the two components
            rgb_pred = rgb_low_blended + H_blended
            
            # Clamp to valid range
            rgb_pred = torch.clamp(rgb_pred, 0.0, 1.0)

            # ===================================================
            # COMPUTE LOSSES - Supervise the RESIDUALS directly!
            # ===================================================
            
            mask_count = mask_smooth.sum() + 1e-6
            
            # Direct supervision on low frequency RESIDUAL
            loss_residual_low = (mask_smooth * (residual_low_pred - residual_low_gt).abs()).sum() / mask_count
            
            # Direct supervision on high frequency RESIDUAL
            loss_residual_high = (mask_smooth * (residual_high_pred - residual_high_gt).abs()).sum() / mask_count
            
            # Loss on final reconstructed RGB
            loss_rgb_final = (mask_smooth * (rgb_pred - cl_rgb).abs()).sum() / mask_count
            
            # Perceptual losses
            loss_perc = perc(rgb_pred, cl_rgb)
            loss_style = style_loss_module(rgb_pred, cl_rgb)
            loss_ssim = kornia.losses.ssim_loss(rgb_pred, cl_rgb, window_size=11)

            # ===================================================
            # TOTAL LOSS
            # ===================================================
            loss = (w_residual_low * loss_residual_low +
                    w_residual_high * loss_residual_high +
                    w_rgb_final * loss_rgb_final +
                    w_perc * loss_perc +
                    w_style * loss_style +
                    w_ssim * loss_ssim)

            optG.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list( resnet_low.parameters()) + list( resnet_high.parameters()), 
                max_norm=5.0
            )
            optG.step()

            epoch_loss += loss.item()
            batch_count += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'res_low': f'{loss_residual_low.item():.3f}',
                'res_high': f'{loss_residual_high.item():.3f}',
                'rgb_fin': f'{loss_rgb_final.item():.3f}'
            })

        scheduler.step()
        avg_loss = epoch_loss / batch_count

        avg_val_loss = validate_model(val_dataloader, resnet_low, resnet_high, perc, style_loss_module, device,
                                w_residual_low, w_residual_high, w_rgb_final, w_perc, w_style, w_ssim)



        # ===================================================
        # CHECKPOINTING
        # ===================================================
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_dir = os.path.join(checkpoint_dir, 'best')
            os.makedirs(best_dir, exist_ok=True)
            save_path = os.path.join(best_dir, f'best_epoch_{epoch+1}.pt')
            torch.save({
                'resnet_low':  resnet_low.state_dict(),
                'resnet_high':  resnet_high.state_dict(),
                'optG': optG.state_dict(),
                'epoch': epoch
            }, save_path)
            print(f"\nðŸ† New best model (epoch {epoch+1}, loss {best_loss:.4f})")
            print(f"   Saved to: {save_path}")

        if (epoch + 1) % 50 == 0:
            checkpoint_dir_epochs = os.path.join(checkpoint_dir, 'checkpoints')
            os.makedirs(checkpoint_dir_epochs, exist_ok=True)
            ckpt_path = os.path.join(checkpoint_dir_epochs, f'epoch_{epoch+1}.pt')
            torch.save({
                'resnet_low':  resnet_low.state_dict(),
                'resnet_high':  resnet_high.state_dict(),
                'optG': optG.state_dict(),
                'epoch': epoch
            }, ckpt_path)
            print(f"\nâœ… Saved checkpoint at epoch {epoch+1}")

            print(f"""
            Epoch {epoch+1} Loss Breakdown:
            =====================================
            RESIDUAL LOSSES:
            - Residual Low:   {loss_residual_low.item():.4f}
            - Residual High:  {loss_residual_high.item():.4f}
            
            RECONSTRUCTION:
            - RGB_final:      {loss_rgb_final.item():.4f}
            
            PERCEPTUAL:
            - Perceptual:     {loss_perc.item():.4f}
            - Style:          {loss_style.item():.4f}
            - SSIM:           {loss_ssim.item():.4f}
            =====================================
            TOTAL:            {loss.item():.4f}
            =====================================
            """)

    print(f"\n{'='*70}")
    print("âœ… Training completed with RESIDUAL LEARNING in RGB!")
    print(f"{'='*70}")

if __name__ == '__main__':
    train_dual_resnet()