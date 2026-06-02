import os
import re
import csv
import pickle
import numpy as np
from datetime import datetime
from collections import defaultdict
from deepface import DeepFace

# =========================================================
# CẤU HÌNH
# =========================================================
FACES_DB_DIR = "train_models/examples/data"
MODEL_DIR = "train_models/examples/models"

MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "retinaface"

MODEL_PREFIX = f"{MODEL_NAME}_{DETECTOR_BACKEND}"
REGISTRY_FILE = os.path.join(MODEL_DIR, "model_registry.csv")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# =========================================================
# TẠO TÊN MODEL TỰ ĐỘNG KHÔNG BỊ GHI ĐÈ
# =========================================================
def get_next_model_version():
    """
    Tự động tìm version tiếp theo trong thư mục models.
    Ví dụ:
        Facenet512_retinaface_001.pkl
        Facenet512_retinaface_002.pkl
        => version tiếp theo là 003
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    pattern = re.compile(rf"^{MODEL_PREFIX}_(\d+)\.pkl$")
    versions = []

    for file_name in os.listdir(MODEL_DIR):
        match = pattern.match(file_name)
        if match:
            versions.append(int(match.group(1)))

    next_version = max(versions, default=0) + 1
    return f"{next_version:03d}"


def get_model_save_path(version):
    file_name = f"{MODEL_PREFIX}_{version}.pkl"
    return os.path.join(MODEL_DIR, file_name)


# =========================================================
# TRÍCH XUẤT EMBEDDING
# =========================================================
def get_embedding(img_path):
    """
    Trích xuất vector embedding từ ảnh khuôn mặt.
    """
    result = DeepFace.represent(
        img_path=img_path,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=False,
    )

    if not result:
        raise ValueError("Không trích xuất được embedding.")

    return np.array(result[0]["embedding"], dtype=np.float32)


# =========================================================
# LOAD DATASET ẢNH KHUÔN MẶT
# =========================================================
def load_data():
    """
    Đọc dữ liệu theo cấu trúc:

    data/
    ├── person_1/
    │   ├── img1.jpg
    │   └── img2.jpg
    ├── person_2/
    │   ├── img1.jpg
    │   └── img2.jpg

    Trả về:
        embeddings_by_person = {
            "person_1": [embedding1, embedding2],
            "person_2": [embedding1, embedding2]
        }
    """
    if not os.path.exists(FACES_DB_DIR):
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {FACES_DB_DIR}")

    embeddings_by_person = defaultdict(list)
    total_images = 0
    failed_images = 0

    print("Đang đọc dữ liệu khuôn mặt...")

    for person_name in sorted(os.listdir(FACES_DB_DIR)):
        person_dir = os.path.join(FACES_DB_DIR, person_name)

        if not os.path.isdir(person_dir):
            continue

        print(f"\nNgười: {person_name}")

        image_files = [
            file for file in os.listdir(person_dir)
            if file.lower().endswith(IMAGE_EXTENSIONS)
        ]

        if len(image_files) == 0:
            print("  Không có ảnh hợp lệ.")
            continue

        for file_name in sorted(image_files):
            img_path = os.path.join(person_dir, file_name)

            try:
                emb = get_embedding(img_path)
                embeddings_by_person[person_name].append(emb)
                total_images += 1
                print(f"  OK: {file_name}")

            except Exception as e:
                failed_images += 1
                print(f"  FAILED: {file_name} - {e}")

    print("\n========== THỐNG KÊ DATASET ==========")
    print(f"Số người hợp lệ: {len(embeddings_by_person)}")
    print(f"Số ảnh trích xuất thành công: {total_images}")
    print(f"Số ảnh lỗi: {failed_images}")

    return embeddings_by_person, total_images, failed_images


# =========================================================
# TÍNH CENTROID CHO TỪNG NGƯỜI
# =========================================================
def compute_centroids(embeddings_by_person):
    """
    Tính vector trung bình đại diện cho từng người.
    """
    centroids = {}

    print("\nĐang tính centroid...")

    for person_name, embeddings in embeddings_by_person.items():
        if len(embeddings) == 0:
            continue

        embeddings = np.array(embeddings, dtype=np.float32)
        centroid = np.mean(embeddings, axis=0)

        # Chuẩn hóa L2 để cosine similarity ổn định hơn
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        centroids[person_name] = centroid

        print(
            f"  {person_name}: {len(embeddings)} ảnh "
            f"-> centroid shape: {centroid.shape}"
        )

    if len(centroids) == 0:
        raise ValueError("Không tạo được centroid nào. Hãy kiểm tra dữ liệu ảnh.")

    return centroids


# =========================================================
# LƯU MODEL REGISTRY
# =========================================================
def update_model_registry(version, save_path, num_classes, total_images, failed_images):
    """
    Lưu lịch sử các lần train vào model_registry.csv.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    file_exists = os.path.exists(REGISTRY_FILE)

    row = {
        "version": version,
        "model_name": MODEL_NAME,
        "detector_backend": DETECTOR_BACKEND,
        "num_classes": num_classes,
        "total_images": total_images,
        "failed_images": failed_images,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_file": os.path.basename(save_path),
        "model_path": save_path,
    }

    with open(REGISTRY_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    print(f"Đã cập nhật registry: {REGISTRY_FILE}")


# =========================================================
# LƯU MODEL
# =========================================================
def save_model(centroids, embeddings_by_person, total_images, failed_images):
    """
    Lưu model với tên tự động tăng version.
    """
    version = get_next_model_version()
    save_path = get_model_save_path(version)

    model_data = {
        "version": version,
        "model_name": MODEL_NAME,
        "detector_backend": DETECTOR_BACKEND,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_classes": len(centroids),
        "total_images": total_images,
        "failed_images": failed_images,
        "persons": list(centroids.keys()),
        "images_per_person": {
            person: len(embs)
            for person, embs in embeddings_by_person.items()
        },
        "centroids": centroids,
    }

    with open(save_path, "wb") as f:
        pickle.dump(model_data, f)

    update_model_registry(
        version=version,
        save_path=save_path,
        num_classes=len(centroids),
        total_images=total_images,
        failed_images=failed_images,
    )

    return save_path, model_data


# =========================================================
# KIỂM TRA MODEL SAU KHI LƯU
# =========================================================
def verify_saved_model(save_path):
    """
    Kiểm tra nhanh file model sau khi lưu.
    """
    with open(save_path, "rb") as f:
        model_data = pickle.load(f)

    print("\n========== KIỂM TRA MODEL ==========")
    print(f"Version: {model_data['version']}")
    print(f"Model: {model_data['model_name']}")
    print(f"Detector: {model_data['detector_backend']}")
    print(f"Số lớp/người: {model_data['num_classes']}")
    print(f"Số ảnh train: {model_data['total_images']}")
    print(f"Thời gian tạo: {model_data['created_at']}")
    print("Danh sách người:")
    for person in model_data["persons"]:
        print(f"  - {person}")


# =========================================================
# TRAIN PIPELINE
# =========================================================
def train_and_save():
    print("========== TRAIN FACE RECOGNITION MODEL ==========")
    print(f"Faces DB: {FACES_DB_DIR}")
    print(f"Model dir: {MODEL_DIR}")
    print(f"Model name: {MODEL_NAME}")
    print(f"Detector backend: {DETECTOR_BACKEND}")

    embeddings_by_person, total_images, failed_images = load_data()

    if total_images == 0:
        raise ValueError("Không có ảnh nào được trích xuất embedding thành công.")

    centroids = compute_centroids(embeddings_by_person)

    save_path, model_data = save_model(
        centroids=centroids,
        embeddings_by_person=embeddings_by_person,
        total_images=total_images,
        failed_images=failed_images,
    )

    verify_saved_model(save_path)

    print("\n========== HOÀN TẤT ==========")
    print(f"Đã lưu model tại: {save_path}")
    print(f"File registry: {REGISTRY_FILE}")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    train_and_save()
