"""Generate model complexity and efficiency trade-off charts."""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

fig_dir = Path("reports/latex_oil_spill_report/figures")
fig_dir.mkdir(parents=True, exist_ok=True)

models = ["DeepLabV3+\n(ResNet34)", "UNet++\n(EfficientNet-B4)", "Ensemble\n(DeepLab+UNet)"]
model_names_flat = ["DeepLabV3+ (ResNet34)", "UNet++ (EfficientNet-B4)", "Ensemble (DeepLab+UNet)"]
dice = [0.8832, 0.8960, 0.9026]
flops = [15.77, 19.44, 35.21]  # GFLOPs
params = [22.44, 20.92, 43.36]  # Millions
latency_gpu = [8.19, 49.03, 57.22]  # ms
fps_gpu = [122.1, 20.4, 17.5]

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.2), dpi=200)

# Subplot 1: Dice vs FLOPs & Parameter size (bubble chart)
colors = ["#1f77b4", "#ff7f0e", "#d62728"]
sizes = [p * 35 for p in params]

for i, (m, d, f, s, c) in enumerate(zip(models, dice, flops, sizes, colors)):
    ax1.scatter(f, d, s=s, color=c, alpha=0.75, edgecolors="black", linewidth=1.5, zorder=3)
    offset_y = 0.0016 if i != 2 else -0.0028
    offset_x = 0.8 if i != 0 else -6.5
    label_txt = f"{model_names_flat[i]}\nDice={d:.4f} | {params[i]:.1f}M params"
    ax1.annotate(
        label_txt,
        (f, d),
        xytext=(f + offset_x, d + offset_y),
        fontsize=9.5,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.0),
    )

ax1.set_xlabel("Độ phức tạp tính toán (GFLOPs)", fontsize=11, fontweight="bold")
ax1.set_ylabel("Chỉ số Dice Score trên tập xác thực", fontsize=11, fontweight="bold")
ax1.set_title("Đánh đổi giữa Độ chính xác (Dice) và Độ phức tạp tính toán (FLOPs)\n(Kích thước bong bóng tỉ lệ thuận với số lượng tham số)", fontsize=11, fontweight="bold")
ax1.set_ylim(0.875, 0.908)
ax1.set_xlim(5, 42)
ax1.grid(True, linestyle="--", alpha=0.6)

# Subplot 2: Inference Latency vs FPS on GPU
x = np.arange(len(models))
width = 0.35

ax2_fps = ax2.twinx()
rects1 = ax2.bar(x - width/2, latency_gpu, width, label="Độ trễ GPU (ms)", color="#3b82f6", alpha=0.85, edgecolor="black")
rects2 = ax2_fps.bar(x + width/2, fps_gpu, width, label="Thông lượng (FPS)", color="#10b981", alpha=0.85, edgecolor="black")

ax2.set_xlabel("Kiến trúc mô hình", fontsize=11, fontweight="bold")
ax2.set_ylabel("Độ trễ GPU (ms/ảnh)", fontsize=11, fontweight="bold", color="#1d4ed8")
ax2_fps.set_ylabel("Thông lượng xử lý (FPS - Ảnh/giây)", fontsize=11, fontweight="bold", color="#047857")
ax2.set_title("Độ trễ suy luận và Thông lượng thời gian thực trên GPU RTX 2050\n(Batch size = 1, độ phân giải 256x256)", fontsize=11, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels([m.replace("\n", " ") for m in models], fontsize=9.5)
ax2.grid(True, linestyle="--", alpha=0.4)

# Value annotations
for rect in rects1:
    h = rect.get_height()
    ax2.annotate(f"{h:.1f} ms", (rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
for rect in rects2:
    h = rect.get_height()
    ax2_fps.annotate(f"{h:.1f} FPS", (rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

plt.tight_layout()
out_path = fig_dir / "model_complexity_tradeoff.png"
plt.savefig(out_path)
plt.savefig("reports/development_experiments/model_complexity_tradeoff.png")
print("Saved chart to:", out_path)
