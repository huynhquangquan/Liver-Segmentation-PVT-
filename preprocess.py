# preprocess.py
import os
import numpy as np
import nibabel as nib
import cv2

def preprocess_volume(volume_path, seg_path, output_dir, slices_per_volume=5, target_size=(256, 256)):
    """
    Đọc volume và segmentation (.nii), trích xuất các lát cắt trung tâm,
    resize về target_size, chuẩn hóa và lưu thành file npz nén.
    """
    # Load volume và segmentation từ file raw (.nii)
    volume_nib = nib.load(volume_path)
    seg_nib = nib.load(seg_path)
    volume_data = volume_nib.get_fdata().astype(np.float32)
    seg_data = seg_nib.get_fdata().astype(np.float32)
    
    # Chuẩn hóa volume: min-max scaling về [0,1]
    vol_min, vol_max = volume_data.min(), volume_data.max()
    if vol_max - vol_min > 0:
        volume_data = (volume_data - vol_min) / (vol_max - vol_min)
    
    # Chuyển segmentation về nhị phân (threshold 0.5)
    seg_data = (seg_data > 0.5).astype(np.float32)
    
    D = volume_data.shape[2]  # Giả sử volume_data có shape (H, W, D)
    mid_slice = D // 2
    half = slices_per_volume // 2
    start = max(0, mid_slice - half)
    end = min(D, mid_slice + half + 1)
    
    base_filename = os.path.splitext(os.path.basename(volume_path))[0]  # ví dụ: volume-01
    for slice_idx in range(start, end):
        # Lấy lát cắt 2D từ volume và segmentation
        image_2d = volume_data[:, :, slice_idx]
        mask_2d = seg_data[:, :, slice_idx]
        
        # Resize: dùng cv2.resize (nội suy tuyến tính cho image, nearest cho mask)
        image_resized = cv2.resize(image_2d, target_size, interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask_2d, target_size, interpolation=cv2.INTER_NEAREST)
        
        # Thêm kênh: chuyển từ (H, W) thành (1, H, W)
        image_resized = np.expand_dims(image_resized, axis=0)
        mask_resized = np.expand_dims(mask_resized, axis=0)
        
        # Kiểm tra dữ liệu (NaN, vô hạn, mask ngoài [0,1])
        if np.isnan(image_resized).any() or np.isinf(image_resized).any():
            print(f"WARNING: Invalid image data in {volume_path}, slice {slice_idx}.")
        if np.isnan(mask_resized).any() or np.isinf(mask_resized).any():
            print(f"WARNING: Invalid mask data in {seg_path}, slice {slice_idx}.")
        if mask_resized.min() < 0 or mask_resized.max() > 1:
            print(f"Warning: Mask values out of [0,1] in {volume_path}, slice {slice_idx}. Clipping performed.")
            mask_resized = np.clip(mask_resized, 0, 1)
        
        # Lưu file dưới dạng npz nén
        filename = f"{base_filename}_slice-{slice_idx}.npz"
        filepath = os.path.join(output_dir, filename)
        np.savez_compressed(filepath, image=image_resized, mask=mask_resized)
        print(f"Saved {filepath}")

def main():
    # Đường dẫn dữ liệu raw (folder 'train' chứa 2 folder: 'volumes' và 'segmentations')
    train_root = "taskliver_03/train"
    volume_dir = os.path.join(train_root, "volumes")
    seg_dir = os.path.join(train_root, "segmentations")
    
    output_dir = "preprocess_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Lấy danh sách file volume (.nii)
    volume_files = sorted([f for f in os.listdir(volume_dir) if f.startswith("volume-") and f.endswith(".nii")])
    
    for vol_file in volume_files:
        seg_file = vol_file.replace("volume", "segmentation")
        volume_path = os.path.join(volume_dir, vol_file)
        seg_path = os.path.join(seg_dir, seg_file)
        print(f"Processing {vol_file} ...")
        preprocess_volume(volume_path, seg_path, output_dir, slices_per_volume=5, target_size=(256,256))

if __name__ == "__main__":
    main()