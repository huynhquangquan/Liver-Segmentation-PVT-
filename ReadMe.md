# Dataset:
* Tập train: https://drive.google.com/drive/folders/0B0vscETPGI1-Q1h1WFdEM2FHSUE?resourcekey=0-XIVV_7YUjB9TPTQ3NfM17A
* Tập Test: https://drive.google.com/drive/folders/0B0vscETPGI1-NDZNd3puMlZiNWM?resourcekey=0-dZUUwJiQnUVYVpRQvs_2tQ

Di chuyển các mẫu .nii vào tên folder tương ứng.
`
├── checkcpu.py                      # Kiểm tra số lõi CPU
├── inference.py                     # Dự đoán (predict)
├── LiTSpreprocessed.py              # Train model sử dụng dữ liệu .npz trong preprocess_data
├── LiTS_full.py                     # Train model với dữ liệu gốc .nii (full pipeline)
├── preprocess.py                    # Tiền xử lý dữ liệu cho LiTSpreprocessed.py
├── ReadMe.md                        # File mô tả dự án
├── requirements.txt                 # Các module và thư viện cần thiết
├── LiTSpreprocessed_nb_version/    # Phiên bản notebook của LiTSpreprocessed
│
├── plots/                           # Lưu kết quả plot từ inference.py hoặc notebook
│
├── preprocess_data/                # Chứa dữ liệu đã xử lý từ preprocess.py (.npz)
│
├── taskliver_03/                   # Chứa dữ liệu raw (.nii)
│   ├── test/
│   └── train/
│       ├── segmentations/
│       └── volumes/
│
└── weights/                         # Chứa các mô hình đã huấn luyện
    ├── pvt_liver_seg_preprocess.pth    # Model từ LiTSpreprocessed
    └── pvt_liver_seg.pth               # Model từ LiTS_full
`
