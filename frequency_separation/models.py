import torch
import torch.nn as nn
from torchvision import models as tv_models
class ResNetAutoencoder101(nn.Module):
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

class ResNetAutoencoder34(nn.Module):
    """
    ResNet34 encoder -> simple ConvTranspose decoder.
    - in_channels: number of input channels (use 4 for RGB+mask)
    - out_channels: number of output channels (3 for RGB residual)
    """
    def __init__(self, in_channels=4, out_channels=3, pretrained=True):
        super().__init__()
        # Load resnet34
        resnet = tv_models.resnet34(pretrained=pretrained)
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
        self.dec_t1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # /16
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.dec_t2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # /8
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
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
        # no activation — your training script expects raw residuals
        return out
