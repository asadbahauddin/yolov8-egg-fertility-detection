from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import os

# ===========================
# 1. Setup
# ===========================
data_yaml = "data.yaml"   # pastikan file ini ada
models_to_train = ["yolov8n.pt", "yolov8s.pt"]  # bisa tambah "yolov8m.pt" kalau GPU kuat

# ===========================
# 2. Training & Evaluasi
# ===========================
for model_name in models_to_train:
    print(f"\n=== Training {model_name} ===")
    model = YOLO(model_name)

    # Training dengan augmentasi cukup kompleks
    results = model.train(
        data=data_yaml,
        epochs=120,
        imgsz=320,
        batch=16,
        degrees=15,
        translate=0.1,
        scale=0.2,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=0.7,
        mixup=0.1,
        patience=30
    )

    # Evaluasi di test set
    print(f"\n--- Evaluasi {model_name} ---")
    metrics = model.val(data=data_yaml, split="test", imgsz=320)
    print(metrics)

    # Export ke TFLite (siap Raspberry Pi)
    model.export(format="tflite", imgsz=320)


# ===========================
# 3. Visualisasi Hasil Training
# ===========================
# Ambil hasil training terakhir
save_dir = model.trainer.save_dir
print("📂 Hasil training tersimpan di:", save_dir)

# Baca CSV hasil training
csv_path = os.path.join(save_dir, "results.csv")
df = pd.read_csv(csv_path)

# Plot mAP50-95
plt.figure()
plt.plot(df["                  metrics/mAP50-95(B)"], label="mAP50-95")
plt.plot(df["             metrics/precision(B)"], label="Precision")
plt.plot(df["                metrics/recall(B)"], label="Recall")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Evaluasi Training YOLOv8")
plt.legend()
plt.grid()
plt.savefig(os.path.join(save_dir, "custom_metrics.png"))
plt.show()

print("Grafik evaluasi tersimpan di:", os.path.join(save_dir, "custom_metrics.png"))

# ===========================
# 4. Ringkasan Tabel untuk Laporan
# ===========================
final_epoch = df.iloc[-1]  # ambil baris terakhir (epoch terakhir)
summary = {
    "Epoch": len(df),
    "Precision": final_epoch["             metrics/precision(B)"],
    "Recall": final_epoch["                metrics/recall(B)"],
    "mAP50": final_epoch["                    metrics/mAP50(B)"],
    "mAP50-95": final_epoch["                  metrics/mAP50-95(B)"]
}

print("\n Ringkasan performa model:")
for k, v in summary.items():
    print(f"{k:10s}: {v:.4f}" if isinstance(v, float) else f"{k:10s}: {v}")

# Bisa juga simpan ke CSV untuk laporan
pd.DataFrame([summary]).to_csv(os.path.join(save_dir, "summary_report.csv"), index=False)
print("Ringkasan tersimpan di summary_report.csv")
