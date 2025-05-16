# Dataset:
* Tập train: https://drive.google.com/drive/folders/0B0vscETPGI1-Q1h1WFdEM2FHSUE?resourcekey=0-XIVV_7YUjB9TPTQ3NfM17A
* Tập Test: https://drive.google.com/drive/folders/0B0vscETPGI1-NDZNd3puMlZiNWM?resourcekey=0-dZUUwJiQnUVYVpRQvs_2tQ

Di chuyển các mẫu .nii vào tên folder tương ứng.
```
¦   checkcpu.py                         -> kiểm tra số lõi cpu
¦   inference.py                        -> dự đoán/ predict
¦   LiTSpreprocessed.py                 -> train model sử dụng dữ liệu .npz (đã được làm nhẹ và qua xử lý) trong preprocess_data
¦   LiTS_full.py                        -> train model với full code sử dụng dữ liệu .nii (bản gốc)
¦   preprocess.py                       -> tiền xử lý dữ liệu cho file LiTSpreprocessed.py
¦   ReadMe.md                           -> md
¦   requirements.txt                    -> module, libary cần thiết cho project
¦   LiTSpreprocessed_nb_version         -> LiTSpreprocessed phiên bản notebook
¦                   
+---plots                               -> folder chứa kết quả dự đoán plot từ inference.py hoặc LiTSpreprocessed_nb_version.ipynb
¦       
+---preprocess_data                     -> folder chứa dữ liệu đã được xử lý từ preprocess.py
¦       
+---taskliver_03                        -> folder chứa dữ liệu raw
¦   +---test
¦   ¦       
¦   +---train
¦       +---segmentations
¦       ¦       
¦       +---volumes
¦               
+---weights                             -> folder chứa model
    +---pvt_liver_seg_preprocess.pth    -> model của LiTSpreprocessed (phiên bản python và notebook, chỉ tồn tại 1)
    ¦       
    +---pvt_liver_seg.pth               -> model của LiTS_full
```
