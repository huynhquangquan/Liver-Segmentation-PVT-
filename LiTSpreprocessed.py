import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import timm  # Sử dụng backbone PVT từ timm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
import random
from sklearn.model_selection import train_test_split  # Sử dụng split đơn giản

# ---------------------------------------------------------
# Transform: Augmentation (flip, rotation) và Resize (256x256)
# ---------------------------------------------------------
class AugmentResizeTransform:
    def __init__(self, size=(256, 256), flip_prob=0.5, rotation_range=10):
        """
        size: tuple (width, height) để resize đến.
        flip_prob: xác suất lật ngang.
        rotation_range: biên độ góc xoay ngẫu nhiên (-rotation_range, rotation_range).
        """
        self.size = size
        self.flip_prob = flip_prob
        self.rotation_range = rotation_range

    def __call__(self, image, mask):
        # image, mask có kích thước (1, H, W)
        image_2d = image[0]
        mask_2d = mask[0]

        # Random horizontal flip
        if random.random() < self.flip_prob:
            image_2d = np.fliplr(image_2d)
            mask_2d = np.fliplr(mask_2d)

        # Random rotation
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        (h, w) = image_2d.shape[:2]
        center = (w // 2, h // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image_2d = cv2.warpAffine(image_2d, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)
        mask_2d = cv2.warpAffine(mask_2d, rot_matrix, (w, h), flags=cv2.INTER_NEAREST)

        # Resize: sử dụng nội suy tuyến tính cho image, nearest cho mask
        image_resized = cv2.resize(image_2d, self.size, interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask_2d, self.size, interpolation=cv2.INTER_NEAREST)

        # Thêm lại kênh: (1, H, W)
        image_resized = np.expand_dims(image_resized, axis=0)
        mask_resized = np.expand_dims(mask_resized, axis=0)

        return image_resized, mask_resized

# ---------------------------------------------------------
# Dataset: Load dữ liệu đã được preprocess từ folder preprocess_data
# ---------------------------------------------------------
class PreprocessedLiverDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        data_dir: folder chứa các file .npz đã được preprocess.
        transform: (tùy chọn) transform để áp dụng augmentation.
        """
        self.data_dir = data_dir
        self.file_list = sorted([f for f in os.listdir(data_dir) if f.endswith(".npz")])
        self.transform = transform
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        filepath = os.path.join(self.data_dir, self.file_list[idx])
        data = np.load(filepath)
        image = data['image']   # shape: (1, H, W)
        mask = data['mask']     # shape: (1, H, W)
        
        # Kiểm tra dữ liệu
        if np.isnan(image).any() or np.isinf(image).any():
            raise ValueError(f"Invalid image data in file {filepath}.")
        if np.isnan(mask).any() or np.isinf(mask).any():
            raise ValueError(f"Invalid mask data in file {filepath}.")
        if mask.min() < 0 or mask.max() > 1:
            print(f"Warning: Mask values out of [0,1] in file {filepath}. Clipping performed.")
            mask = np.clip(mask, 0, 1)
        
        # Áp dụng augmentation nếu transform được cung cấp
        if self.transform is not None:
            image, mask = self.transform(image, mask)
            
        # Chuyển thành tensor
        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask).float()
        return image_tensor, mask_tensor

# ---------------------------------------------------------
# Model segmentation sử dụng backbone PVT với kiến trúc U-Net style
# ---------------------------------------------------------
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
        f1 = features[-3]
        f2 = features[-2]
        f3 = features[-1]
        center = self.center(f3)
        d1 = self.up1(center, f2)
        d2 = self.up2(d1, f1)
        d3 = self.up3(d2)
        out = self.out_conv(d3)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out

# ---------------------------------------------------------
# Hàm tính loss (Dice + BCE)
# ---------------------------------------------------------
def dice_loss(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=[2, 3])
    union = pred.sum(dim=[2, 3]) + target.sum(dim=[2, 3])
    dice_coef = (2. * intersection + smooth) / (union + smooth)
    dice_coef = torch.clamp(dice_coef, max=1.0)
    dice = 1 - dice_coef
    return dice.mean()

def bce_loss(pred, target):
    pred = pred.squeeze(1)
    target = target.squeeze(1)
    return nn.functional.binary_cross_entropy_with_logits(pred, target)

def combined_loss(pred, target):
    return 0.5 * dice_loss(pred, target) + 0.5 * bce_loss(pred, target)

# ---------------------------------------------------------
# Hàm evaluate: Tính Dice coefficient và pixel accuracy trên tập dữ liệu
# ---------------------------------------------------------
def evaluate(model, dataloader, device):
    model.eval()
    dice_scores = []
    pixel_accuracies = []
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()
            
            # Tính Dice cho từng batch
            intersection = (preds * masks).sum(dim=[2,3])
            union = preds.sum(dim=[2,3]) + masks.sum(dim=[2,3])
            dice = (2. * intersection + 1e-5) / (union + 1e-5)
            dice_scores.append(dice.mean().item())
            
            # Tính pixel accuracy
            correct = (preds == masks).float().sum()
            total = torch.numel(preds)
            pixel_accuracy = correct / total
            pixel_accuracies.append(pixel_accuracy.item())
    avg_dice = np.mean(dice_scores)
    avg_accuracy = np.mean(pixel_accuracies)
    return avg_dice, avg_accuracy

# ---------------------------------------------------------
# Hàm evaluate_loss: Tính loss trên tập dữ liệu
# ---------------------------------------------------------
def evaluate_loss(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = combined_loss(outputs, masks)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# ---------------------------------------------------------
# Early Stopping dựa trên validation loss
# ---------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
    def step(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
            return False
        elif self.best_loss - loss > self.min_delta:
            self.best_loss = loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False

# ---------------------------------------------------------
# Vòng lặp train cho một epoch
# ---------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)
        if torch.isnan(images).any() or torch.isinf(images).any():
            print(f"WARNING: Invalid image tensor in batch {batch_idx}")
        if torch.isnan(masks).any() or torch.isinf(masks).any():
            print(f"WARNING: Invalid mask tensor in batch {batch_idx}")
        optimizer.zero_grad()
        outputs = model(images)
        loss = combined_loss(outputs, masks)
        if loss.item() < 0:
            print(f"WARNING: Negative loss encountered in batch {batch_idx}! Loss: {loss.item():.6f}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# ---------------------------------------------------------
# Main: Sử dụng một validation set để đánh giá hiệu năng, overfitting và vẽ các đồ thị metric
# ---------------------------------------------------------
def main():
    # Tham số huấn luyện
    batch_size = 2
    lr = 1e-4
    num_epochs = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_dir = "preprocess_data"  # Folder chứa file .npz đã được preprocess
    transform = AugmentResizeTransform(size=(256, 256), flip_prob=0.5, rotation_range=10)
    dataset = PreprocessedLiverDataset(data_dir, transform=transform)
    
    # Tách dữ liệu thành train và validation (80/20)
    indices = list(range(len(dataset)))
    train_indices, valid_indices = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)
    train_subset = Subset(dataset, train_indices)
    valid_subset = Subset(dataset, valid_indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Khởi tạo mô hình, optimizer, scheduler và early stopping dựa trên validation loss
    model = PVT_LiverSegImproved(backbone_name="pvt_v2_b2", num_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    early_stopper = EarlyStopping(patience=5, min_delta=1e-4)
    
    # Danh sách lưu lại các metric qua từng epoch
    train_losses = []
    valid_losses = []
    train_dice_list = []
    train_acc_list = []
    valid_dice_list = []
    valid_acc_list = []
    
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        valid_loss = evaluate_loss(model, valid_loader, device)
        scheduler.step(valid_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        avg_dice_train, avg_acc_train = evaluate(model, train_loader, device)
        avg_dice_valid, avg_acc_valid = evaluate(model, valid_loader, device)
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_dice_list.append(avg_dice_train)
        train_acc_list.append(avg_acc_train)
        valid_dice_list.append(avg_dice_valid)
        valid_acc_list.append(avg_acc_valid)
        
        # Tính hiệu số loss giữa train và validation để theo dõi overfitting
        loss_gap = valid_loss - train_loss
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Gap: {loss_gap:.4f}, LR: {current_lr:.6f}")
        print(f"Train Dice: {avg_dice_train:.4f}, Train Acc: {avg_acc_train:.4f} | Valid Dice: {avg_dice_valid:.4f}, Valid Acc: {avg_acc_valid:.4f}")
        
        if loss_gap > 0.05:
            print("Warning: Hiệu số loss giữa train và valid khá lớn, có thể mô hình đang overfitting!")
        
        if early_stopper.step(valid_loss):
            print("Early stopping triggered!")
            break

    # Lưu mô hình cuối cùng
    os.makedirs("weights", exist_ok=True)
    save_path = "weights/pvt_liver_seg_preprocess.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Đã lưu mô hình tại {save_path}")
    
    # Vẽ đồ thị Loss cho train và validation
    plt.figure()
    plt.plot(train_losses, marker='o', label='Train Loss')
    plt.plot(valid_losses, marker='o', label='Valid Loss')
    plt.title("Biểu đồ Loss qua các Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_plot.png")
    plt.show()
    
    # Vẽ đồ thị Dice coefficient cho train và validation
    plt.figure()
    plt.plot(train_dice_list, marker='o', label='Train Dice')
    plt.plot(valid_dice_list, marker='o', label='Valid Dice')
    plt.title("Biểu đồ Dice Coefficient qua các Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Coefficient")
    plt.legend()
    plt.grid(True)
    plt.savefig("dice_plot.png")
    plt.show()
    
    # Vẽ đồ thị Pixel Accuracy cho train và validation
    plt.figure()
    plt.plot(train_acc_list, marker='o', label='Train Pixel Acc')
    plt.plot(valid_acc_list, marker='o', label='Valid Pixel Acc')
    plt.title("Biểu đồ Pixel Accuracy qua các Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Pixel Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("pixel_acc_plot.png")
    plt.show()

if __name__ == "__main__":
    main()
