"""Regenerate evaluation charts from validation CSV results (DeepLabV3+, UNet++, Ensemble)."""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    exp_dir = root / "reports" / "development_experiments"
    fig_dir = root / "reports" / "latex_oil_spill_report" / "figures"
    out_dirs = [d for d in [exp_dir, fig_dir] if d.exists()]

    valid_models = {"DeepLabV3+", "UNet++", "Ensemble"}

    # 1. Clean threshold_search_results.csv
    thresh_csv = exp_dir / "threshold_search_results.csv"
    with thresh_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [r for r in reader if r["model"] in valid_models]

    with thresh_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 2. Clean best_results.csv
    best_csv = exp_dir / "best_results.csv"
    with best_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames_best = list(reader.fieldnames or [])
        best_rows = [r for r in reader if r["model"] in valid_models]

    with best_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_best)
        writer.writeheader()
        writer.writerows(best_rows)

    # 3. Plot threshold_dice_curves.png
    models = ["DeepLabV3+", "UNet++", "Ensemble"]
    plt.figure(figsize=(10, 6))
    for model in models:
        model_rows = [r for r in rows if r["model"] == model and r["variant"] == "raw"]
        xs = [float(r["threshold"]) for r in model_rows]
        ys = [float(r["dice"]) for r in model_rows]
        plt.plot(xs, ys, marker="o", linewidth=2, label=model)
    plt.xlabel("Threshold")
    plt.ylabel("Dice")
    plt.title("Threshold search on validation set")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    for d in out_dirs:
        plt.savefig(d / "threshold_dice_curves.png", dpi=180)
    plt.close()

    # 4. Plot best_metrics_comparison.png
    labels = [f"{r['model']}\n{r['variant']}" for r in best_rows]
    dice = [float(r["dice"]) for r in best_rows]
    iou = [float(r["iou"]) for r in best_rows]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, dice, width, label="Dice")
    plt.bar(x + width / 2, iou, width, label="IoU")
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Best validation metrics")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    for d in out_dirs:
        plt.savefig(d / "best_metrics_comparison.png", dpi=180)
    plt.close()

    # 5. Selected rows for detailed comparison
    selected = []
    for m, v in [("DeepLabV3+", "raw"), ("UNet++", "raw"), ("Ensemble", "raw"), ("Ensemble", "postprocess")]:
        for r in best_rows:
            if r["model"] == m and r["variant"] == v:
                selected.append(r)
                break

    # Plot validation_metric_comparison.png
    plt.figure(figsize=(10, 5.5))
    x = np.arange(len(selected))
    w = 0.18
    for idx, (m_key, m_label) in enumerate([("dice", "Dice"), ("iou", "IoU"), ("precision", "Precision"), ("recall", "Recall")]):
        vals = [float(r[m_key]) for r in selected]
        plt.bar(x + (idx - 1.5) * w, vals, w, label=m_label)
    xlabels = [f"{r['model']}\n{r['variant']} @ {float(r['threshold']):.2f}" for r in selected]
    plt.xticks(x, xlabels, rotation=12, ha="right")
    plt.ylim(0.75, 0.94)
    plt.ylabel("Score")
    plt.title("Validation Metric Comparison")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(ncol=4, loc="upper center")
    plt.tight_layout()
    for d in out_dirs:
        plt.savefig(d / "validation_metric_comparison.png", dpi=180)
    plt.close()

    # Plot oil_error_composition.png
    plt.figure(figsize=(10, 5.5))
    x = np.arange(len(selected))
    tp_s, fp_s, fn_s = [], [], []
    for r in selected:
        tp, fp, fn = float(r["tp"]), float(r["fp"]), float(r["fn"])
        tot = tp + fp + fn
        tp_s.append(tp / tot)
        fp_s.append(fp / tot)
        fn_s.append(fn / tot)
    tp_arr = np.array(tp_s)
    fp_arr = np.array(fp_s)
    fn_arr = np.array(fn_s)
    plt.bar(x, tp_arr, width=0.55, color="#2ca05a", label="TP oil")
    plt.bar(x, fp_arr, width=0.55, bottom=tp_arr, color="#de2d26", label="FP false alarm")
    plt.bar(x, fn_arr, width=0.55, bottom=tp_arr + fp_arr, color="#2b83ba", label="FN missed oil")
    plt.xticks(x, xlabels, rotation=12, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Share among TP + FP + FN")
    plt.title("Oil-Class Error Composition")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(ncol=3, loc="upper center")
    plt.tight_layout()
    for d in out_dirs:
        plt.savefig(d / "oil_error_composition.png", dpi=180)
    plt.close()

    # Plot confusion_matrices.png (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5))
    fig.suptitle("Normalized Pixel-Level Confusion Matrices on Validation Set", fontsize=14, y=0.98)
    for ax, r in zip(axes.flat, selected):
        tp, fp, fn, tn = int(r["tp"]), int(r["fp"]), int(r["fn"]), int(r["tn"])
        bg_tot = tn + fp
        oil_tot = fn + tp
        cm_norm = np.array([
            [tn / bg_tot, fp / bg_tot],
            [fn / oil_tot, tp / oil_tot],
        ])
        counts = np.array([[tn, fp], [fn, tp]])
        ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred BG", "Pred Oil"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["True BG", "True Oil"])
        ax.set_title(f"{r['model']} - {r['variant']}\nthr={float(r['threshold']):.2f}, Dice={float(r['dice']):.4f}", fontsize=11)
        for i in range(2):
            for j in range(2):
                pct = cm_norm[i, j] * 100
                cnt = counts[i, j]
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{pct:.2f}%\n{cnt:,}", ha="center", va="center", color=color, fontsize=9.5)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    for d in out_dirs:
        plt.savefig(d / "confusion_matrices.png", dpi=180)
    plt.close()

    print("Successfully regenerated all 5 evaluation charts and cleaned CSVs!")


if __name__ == "__main__":
    main()
