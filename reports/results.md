# Results

Cập nhật kết quả sau mỗi lần train đáng tin cậy (Đề tài Hướng A: Học máy Nâng cao).

| Model | Branch | Input Size | Dice | IoU | Precision | Recall | Weight/Run | Note |
|---|---|---:|---:|---:|---:|---:|---|---|
| DeepLabV3+ ResNet34 (Baseline) | `Quang` | 256 | 0.8832 | 0.7909 | 0.8663 | 0.9009 | `weights/quang_best_deeplabv3plus_checkpoint.pth` | Threshold search best at 0.50 |
| DeepLabV3+ Fine-tune (BCE+Tversky) | `Quang` | 256 | 0.8841 | 0.7915 | 0.8680 | 0.9021 | `weights/quang_best_deeplabv3plus_checkpoint.pth` | Threshold sweep best at 0.65 |
| UNet++ EfficientNet-B4 (Nâng cao) | `Khoa` | 256 | 0.8960 | 0.8115 | 0.8892 | 0.9028 | `weights/Khoa_best_Unetplusplus_checkpoint.pth` | Threshold search best at 0.65 |
| Ensemble average (2 mô hình) | `src/inference/ensemble.py` | 256 | 0.9026 | 0.8225 | 0.8904 | 0.9151 | DeepLabV3+ & UNet++ | Best raw ensemble at threshold 0.50 |
| Ensemble + post-process | `src/inference/postprocess.py` | 256 | 0.9027 | 0.8226 | 0.8903 | 0.9154 | DeepLabV3+ & UNet++ | Morphology + remove components `<64` px |
