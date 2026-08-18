import torch
import torch.nn as nn
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw_channel = c * dw_expand
        
        self.conv1 = nn.Conv2d(c, dw_channel, 1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel)
        self.sg1 = SimpleGate()
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, 1)
        )
        
        # Feed-Forward Network
        ffn_channel = c * ffn_expand
        self.conv4 = nn.Conv2d(c, ffn_channel, 1)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1)
        
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x):
        input_x = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg1(x)
        x = self.conv3(x)
        x = x * self.sca(x)
        y = input_x + x * self.beta
        
        # FFN Path
        z = self.conv4(y)
        z = self.sg2(z)
        z = self.conv5(z)
        return y + z * self.gamma

# --- 2. RCAN Attention Block ---
class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(num_features, num_features // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
class NAFNetRCANPipeline(nn.Module):
    def __init__(self, in_channels=1, num_features=64, scale_factor=2):
        super().__init__()
        
        # Step 1: Base Smoothening / Denoising (NAFNet Backbone)
        self.head = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.naf_denoiser = nn.Sequential(
            NAFBlock(num_features),
            NAFBlock(num_features)
        )
        self.denoise_tail = nn.Conv2d(num_features, in_channels, 3, padding=1)
        
        # Step 3: Residual Detail Rectification
        self.residual_head = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.naf_detail = nn.Sequential(
            NAFBlock(num_features),
            NAFBlock(num_features)
        )
        self.detail_tail = nn.Conv2d(num_features, in_channels, 3, padding=1)
        self.sr_head = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.ca_block = ChannelAttention(num_features)
        self.upscaler = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(scale_factor),
            NAFBlock(num_features),
            NAFBlock(num_features),
            nn.Conv2d(num_features, in_channels, 3, padding=1)
        )

    def forward(self, noisy_lr):
        # Step 1: Base Smoothening via NAFNet
        feat_base = self.head(noisy_lr)
        feat_denoised = self.naf_denoiser(feat_base)
        base_img = self.denoise_tail(feat_denoised) + noisy_lr
        
        # Step 2: Extract Residual
        raw_residual = noisy_lr - base_img
        
        # Step 3: Rectify Residual Map
        feat_res = self.residual_head(raw_residual)
        feat_rectified = self.naf_detail(feat_res)
        cleaned_residual = self.detail_tail(feat_rectified)
        
        # Step 4: Re-inject Recovered Details
        sharp_lr = base_img + cleaned_residual
        
        # Step 5: Upscale to High Resolution
        sr_feat = self.sr_head(sharp_lr)
        sr_feat = self.ca_block(sr_feat)
        final_hr = self.upscaler(sr_feat)
        
        return final_hr