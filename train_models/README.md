# Hướng dẫn Train Model Face Recognition

Thư mục này chứa các pipeline train và test model nhận diện khuôn mặt cho hệ thống điểm danh thông minh.

## Cấu trúc thư mục

```
train_models/
├── examples_001/
│   ├── data/                  # Ảnh gốc để train (chia theo thư mục từng người)
│   │   ├── bao/               # Ảnh của Bao
│   │   ├── nguyen/            # Ảnh của Nguyên
│   │   └── thinh/             # Ảnh của Thịnh
│   ├── models/                # Model đã train (output)
│   ├── csv/                   # File CSV điểm danh
│   ├── embedding_extractor.py # Hàm trích xuất embedding từ DeepFace
│   ├── train_model.py         # Train SVM + OpenCV detector
│   ├── train_model_mtcnn.py   # Train centroid + MTCNN detector
│   ├── train_model_retinaface.py # Train centroid + RetinaFace detector
│   └── realtime.py            # Chạy điểm danh real-time với webcam
└── README.md
```

## Yêu cầu

```bash
pip install opencv-python numpy pandas scikit-learn deepface tensorflow
```

## 1. Chuẩn bị dữ liệu

Đặt ảnh khuôn mặt của mỗi người vào thư mục riêng trong `data/`:

```
data/
├── ten_nguoi_1/
│   ├── anh1.jpg
│   ├── anh2.jpg
│   └── ...
├── ten_nguoi_2/
│   └── ...
```

Mỗi thư mục con đại diện cho một người (tên thư mục = nhãn). Hỗ trợ `.jpg`, `.png`, `.jpeg`.

## 2. Các phương pháp train

### 2.1. SVM + OpenCV detector

**File:** `train_model.py`

- Dùng `detector_backend` mặc định của DeepFace (OpenCV)
- Trích xuất embedding bằng Facenet512
- Chuẩn hóa L2 (Normalizer)
- Train SVM (kernel linear, probability=True)
- Lưu tuple `(model, norm)` dạng pickle

```bash
python train_models/examples_001/train_model.py
```

Output: `models/Facenet_svm_001.pkl`, `models/Facenet512_svm_001.pkl`

### 2.2. Centroid + MTCNN detector

**File:** `train_model_mtcnn.py`

- Dùng `detector_backend="mtcnn"` — phát hiện mặt chính xác hơn OpenCV
- Trích xuất embedding Facenet512
- Tính centroid (embedding trung bình) cho mỗi người
- Không dùng SVM, dùng cosine similarity khi inference
- Lưu dict `{tên: centroid_vector}` dạng pickle

```bash
python train_models/examples_001/train_model_mtcnn.py
```

Output: `models/Facenet512_mtcnn_001.pkl`

### 2.3. Centroid + RetinaFace detector (Khuyến nghị)

**File:** `train_model_retinaface.py`

- Dùng `detector_backend="retinaface"` — phát hiện mặt chính xác nhất
- Trích xuất embedding Facenet512
- Tính centroid cho mỗi người
- Lưu dict `{tên: centroid_vector}` dạng pickle

```bash
python train_models/examples_001/train_model_retinaface.py
```

Output: `models/Facenet512_retinaface_001.pkl`

> **So sánh detectors:** `retinaface` > `mtcnn` > `opencv` (độ chính xác phát hiện mặt)

## 3. Chạy real-time attendance

**File:** `realtime.py`

Sau khi train xong, chạy điểm danh qua webcam:

```bash
python train_models/examples_001/realtime.py
```

- Nhận diện khuôn mặt qua webcam
- So sánh embedding với centroids bằng cosine similarity
- Ghi nhận điểm danh vào CSV (mỗi người 1 lần/ngày)
- Nhấn `Q` hoặc `ESC` để thoát

## 4. Các model đã train sẵn

| File | Detector | Classifier | Kích thước |
|------|----------|------------|------------|
| `Facenet_svm_001.pkl` | OpenCV | SVM | Nhỏ (FaceNet 128-dim) |
| `Facenet512_svm_001.pkl` | OpenCV | SVM | Trung bình (FaceNet512) |
| `Facenet512_mtcnn_001.pkl` | MTCNN | Cosine + Centroid | Trung bình |
| `Facenet512_retinaface_001.pkl` | RetinaFace | Cosine + Centroid | Trung bình |

## 5. Cấu hình

Các tham số trong `realtime.py`:

- `CONFIDENCE_THRESHOLD`: Ngưỡng confidence (%) để chấp nhận nhận diện (mặc định 60)
- `SKIP_FRAMES`: Số frame bỏ qua giữa các lần nhận diện (mặc định 3)
- `DETECTOR_BACKEND`: Backend phát hiện mặt khi realtime (mặc định "opencv" — nhanh hơn)
- `START_TIME` / `END_TIME`: Khung giờ điểm danh
