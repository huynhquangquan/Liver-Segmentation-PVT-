import os
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import timm
import cv2  # Dùng để resize ảnh
import torch.nn.functional as F

# Kiến trúc cải tiến giống training
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_block = ConvBlock(in_channels + skip_channels, out_channels)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)

class PVT_LiverSegImproved(nn.Module):
    def __init__(self, backbone_name="pvt_v2_b2", num_classes=1):
        super(PVT_LiverSegImproved, self).__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=True)
        self.encoder_channels = [feat['num_chs'] for feat in self.backbone.feature_info][-3:]
        
        self.center = ConvBlock(self.encoder_channels[2], 256)
        self.up1 = UpBlock(256, self.encoder_channels[1], 128)
        self.up2 = UpBlock(128, self.encoder_channels[0], 64)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        if C == 1:
            x = x.repeat(1, 3, 1, 1)
        
        features = self.backbone(x)
        # Lấy 3 tầng cuối từ backbone
        f1 = features[-3]  # độ phân giải cao
        f2 = features[-2]
        f3 = features[-1]  # độ phân giải thấp, đặc trưng mạnh
        
        center = self.center(f3)
        d1 = self.up1(center, f2)
        d2 = self.up2(d1, f1)
        d3 = self.up3(d2)
        out = self.out_conv(d3)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load model cải tiến
    model = PVT_LiverSegImproved(backbone_name="pvt_v2_b2", num_classes=1).to(device)
    model.load_state_dict(torch.load("weights/pvt_liver_seg_preprocess.pth", map_location=device))  
    model.eval()
    
    # 2. Folder test
    test_root = "taskliver_03/test"
    volume_files = sorted([
        f for f in os.listdir(test_root)
        if f.startswith("test-volume-") and f.endswith(".nii")
    ])
    
    os.makedirs("plots", exist_ok=True)
    
    for vol_file in volume_files:
        volume_path = os.path.join(test_root, vol_file)
        volume_nib = nib.load(volume_path)
        volume_data = volume_nib.get_fdata().astype(np.float32)
        
        mid_slice = volume_data.shape[2] // 2
        image_2d = volume_data[:, :, mid_slice]
        
        image_resized = cv2.resize(image_2d, (256, 256), interpolation=cv2.INTER_LINEAR)
        image_tensor = torch.from_numpy(image_resized).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(image_tensor)
            pred_sigmoid = torch.sigmoid(pred)
            pred_mask = (pred_sigmoid > 0.5).float().cpu().numpy()[0, 0]
        
        plt.figure(figsize=(12, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(image_resized, cmap='gray', aspect='equal')
        plt.title("Input Slice")
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.imshow(image_resized, cmap='gray', aspect='equal')
        plt.imshow(pred_mask, alpha=0.5, cmap='jet')
        plt.title("Predicted Mask")
        plt.axis("off")
        
        save_path = os.path.join("plots", f"pred_{vol_file}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"Đã lưu kết quả dự đoán cho {vol_file} tại {save_path}")

if __name__ == "__main__":
    main()
