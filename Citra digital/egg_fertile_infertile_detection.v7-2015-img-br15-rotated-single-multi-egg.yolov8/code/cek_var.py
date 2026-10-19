from ultralytics import YOLO
import matplotlib.pyplot as plt

# 🔹 Load model hasil training
model = YOLO("runs/detect/train/weights/best.pt")  # bisa juga "best.onnx"

# 🔹 Ganti path gambar sesuai inputmu
img_path = r"C:\Asad\Kuliah\Penenlitian\pak latif\Citra digital\egg_fertile_infertile_detection.v7-2015-img-br15-rotated-single-multi-egg.yolov8\gambar random cek\infertile.infertile1214 Small.png.4gbb6llh.ingestion-6bf64f87d9-gzjsg.jpg"

# 🔹 Lakukan prediksi
results = model.predict(img_path, imgsz=320, conf=0.5, save=True)

# 🔹 Tampilkan hasil di Notebook
for r in results:
    im_array = r.plot()  # gambar dengan bbox
    plt.imshow(im_array[..., ::-1])  # BGR -> RGB
    plt.axis("off")
    plt.show()
