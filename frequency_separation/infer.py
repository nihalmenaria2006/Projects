import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from glob import glob

import torch
import torch.nn as nn
import torchvision.models as tv_models

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

# ===================================
# INFERENCE FUNCTION - RESIDUAL LEARNING (RGB)
# =========================================
def inference_dual_resnet_residual_rgb(
    wrinkled_dir,
    mask_dir,
    output_dir,
    checkpoint_path,
    device='cuda'
):
    """
    Inference using two separate U-Nets with RESIDUAL LEARNING in RGB
    
    Args:
        wrinkled_dir: Directory containing wrinkled images
        mask_dir: Directory containing mask images
        output_dir: Directory to save output images
        checkpoint_path: Path to checkpoint file containing both models
        device: 'cuda' or 'cpu'
    """
    print(f"\n{'='*70}")
    print("🚀 Starting Dual U-Net Inference - RESIDUAL LEARNING (RGB)")
    print(f"{'='*70}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize both models (RGB: 3 channels + 1 mask = 4 input channels)
    resnet_low = ResNetAutoencoder(in_channels=4, out_channels=3).to(device)
    resnet_high = ResNetAutoencoder(in_channels=4, out_channels=3).to(device)
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    resnet_low.load_state_dict(checkpoint['resnet_low'])
    resnet_high.load_state_dict(checkpoint['resnet_high'])
    
    resnet_low.eval()
    resnet_high.eval()
    print("✅ Models loaded successfully\n")
    
    # Get image files
    wrinkled_files = sorted(glob(os.path.join(wrinkled_dir, '*')))
    mask_files = sorted(glob(os.path.join(mask_dir, '*')))
    
    assert len(wrinkled_files) == len(mask_files), "Mismatch in number of images and masks"
    
    print(f"Found {len(wrinkled_files)} images to process\n")
    
    with torch.no_grad():
        for idx, (wr_path, mask_path) in enumerate(zip(wrinkled_files, mask_files)):
            print(f"Processing [{idx+1}/{len(wrinkled_files)}]: {os.path.basename(wr_path)}")
            
            # Read images
            wr_rgb = read_img(wr_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Resize both to 1024x1024
            wr_rgb = cv2.resize(wr_rgb, (1024, 1024), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            
            # Normalize RGB to [0, 1] and convert to tensor
            wr_rgb_norm = (wr_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)  # (3, H, W)
            wr_rgb_tensor = torch.from_numpy(wr_rgb_norm).unsqueeze(0).float().to(device)  # (1, 3, H, W)
            
            # Frequency split on RGB
            rgb_low_wr, H_wr = frequency_split_rgb(
                wr_rgb_tensor,
                kernel_size=11,
                sigma=5.0
            )
            
            # Prepare mask
            mask_t = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
            mask_smooth = kornia.filters.gaussian_blur2d(mask_t, (9, 9), (3.0, 3.0))
            mask_smooth = mask_smooth / (mask_smooth.max() + 1e-6)
            mask_smooth = torch.clamp(mask_smooth, 0.0, 1.0)
            
            # ============================================================
            # RESIDUAL LEARNING INFERENCE
            # Networks predict RESIDUALS (corrections) to apply
            # ============================================================
            
            # Low frequency network - predicts RESIDUAL
            in_low = torch.cat([rgb_low_wr, mask_smooth], dim=1)  # (1, 4, H, W)
            residual_low_pred = resnet_low(in_low)  # (1, 3, H, W)
            rgb_low_pred = rgb_low_wr + residual_low_pred  # Apply correction
            
            # High frequency network - predicts RESIDUAL
            in_high = torch.cat([H_wr, mask_smooth], dim=1)  # (1, 4, H, W)
            residual_high_pred = resnet_high(in_high)  # (1, 3, H, W)
            H_pred = H_wr + residual_high_pred  # Apply correction
            
            # Blend each component with mask BEFORE combining
            rgb_low_blended = blend_with_mask(rgb_low_wr, rgb_low_pred, mask_smooth)
            H_blended = blend_with_mask(H_wr, H_pred, mask_smooth)
            
            # Reconstruct RGB by adding components
            rgb_pred = rgb_low_blended + H_blended
            
            # Clamp to valid range
            rgb_pred = torch.clamp(rgb_pred, 0.0, 1.0)
            
            # Convert back to numpy
            rgb_pred_np = rgb_pred.squeeze().cpu().numpy()  # (3, H, W)
            rgb_pred_np = (rgb_pred_np.transpose(1, 2, 0) * 255.0).astype(np.uint8)  # (H, W, 3)
            
            # Save output
            output_path = os.path.join(output_dir, os.path.basename(wr_path))
            cv2.imwrite(output_path, rgb_pred_np[:,:,::-1])  # Convert RGB to BGR for cv2
            print(f"   ✅ Saved to: {output_path}")
            
            # Optional: Print residual statistics for debugging
            if idx == 0:  # Only for first image
                res_low_mean = residual_low_pred.abs().mean().item()
                res_high_mean = residual_high_pred.abs().mean().item()
                print(f"   📊 Avg residual magnitudes - Low: {res_low_mean:.4f}, High: {res_high_mean:.4f}")
    
    print(f"\n{'='*70}")
    print("✅ Dual U-Net residual inference completed (RGB)!")
    print(f"{'='*70}")


if __name__ == '__main__':
    # Configuration
    wrinkled_dir = '/home/ml2/Documents/experiments_6_oct/wrinkle_removal/frequency_seperation/best_working/resnet_modified_new/data/val/image'
    mask_dir = '/home/ml2/Documents/experiments_6_oct/wrinkle_removal/frequency_seperation/best_working/resnet_modified_new/data/val/mask'
    output_dir = '/home/ml2/Documents/experiments_6_oct/wrinkle_removal/frequency_seperation/best_working/resnet_modified_new/checkpoints_dual_unet_residual_rgb_modified/preds'
    
    # Path to your trained checkpoint (use RGB residual training checkpoint!)
    checkpoint_path = '/home/ml2/Documents/experiments_6_oct/wrinkle_removal/frequency_seperation/best_working/resnet_modified_new/checkpoints_dual_unet_residual_rgb_modified/best/best_epoch_1.pt'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    inference_dual_resnet_residual_rgb(
        wrinkled_dir=wrinkled_dir,
        mask_dir=mask_dir,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        device=device
    )
# =========================================
# MAIN FUNCTION
# =========================================
