import cv2
import numpy as np
import pickle
import pandas as pd
from datetime import datetime, time
from deepface import DeepFace
import os

# ================== CẤU HÌNH ==================
MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "opencv"
SKIP_FRAMES = 3

MODEL_PATH = "train_models/examples/models/Facenet512_svm_001.pkl"

ATTENDANCE_DIR = "train_models/examples/csv"
ATTENDANCE_BASE = "attendance"

UNKNOWN_THRESHOLD = 70.0

START_TIME = time(14, 43, 0)
END_TIME   = time(14, 44, 0)

# ================== CẤU HÌNH KHUNG NHẬN DIỆN ==================
FRAME_PADDING_RATIO = 0.35
MIN_BOX_PADDING = 35
BOX_THICKNESS = 4
LABEL_FONT_SCALE = 0.75
LABEL_THICKNESS = 2

# ================== LOAD SVM MODEL ==================
with open(MODEL_PATH, "rb") as f:
    svm_model, norm = pickle.load(f)

print("Loaded SVM model:", MODEL_PATH)
print("Classes:", svm_model.classes_)

# ================== FILE CSV ==================
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

session_start = datetime.now()
session_tag = session_start.strftime("%Y%m%d_%H%M%S")
ATTENDANCE_FILE = f"{ATTENDANCE_DIR}/{ATTENDANCE_BASE}_{session_tag}.csv"

print(f"File ghi điểm danh: {ATTENDANCE_FILE}")

COLUMNS = ["Name", "Time", "Status", "Date", "Confidence"]
df = pd.DataFrame(columns=COLUMNS)

def save_attendance():
    df.to_csv(ATTENDANCE_FILE, index=False)

def get_marked_names_today():
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_df = df[df["Date"] == today_str]
    return set(today_df["Name"].values)

def expand_face_box(x, y, w, h, frame_shape,
                    padding_ratio=FRAME_PADDING_RATIO,
                    min_padding=MIN_BOX_PADDING):
    pad_x = max(int(w * padding_ratio), min_padding)
    pad_y = max(int(h * padding_ratio), min_padding)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame_shape[1] - 1, x + w + pad_x)
    y2 = min(frame_shape[0] - 1, y + h + pad_y)

    return x1, y1, x2 - x1, y2 - y1

# ================== HÀM NHẬN DIỆN BẰNG SVM ==================
def recognize_face_svm(embedding):
    emb = np.array(embedding, dtype=np.float32).reshape(1, -1)

    # Normalize giống lúc train
    emb_norm = norm.transform(emb)

    # Dự đoán xác suất
    probabilities = svm_model.predict_proba(emb_norm)[0]

    best_idx = np.argmax(probabilities)
    best_prob = probabilities[best_idx]

    best_name = svm_model.classes_[best_idx]
    confidence = best_prob * 100

    if confidence >= UNKNOWN_THRESHOLD:
        return best_name, confidence

    return "Unknown", confidence

# ================== KHỞI TẠO ==================
marked_names = get_marked_names_today()
current_date_holder = datetime.now().strftime("%Y-%m-%d")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)

frame_count = 0
last_results = []
blink_counter = 0

print("Nhấn Q hoặc ESC để thoát")
print(f"Khung giờ điểm danh: {START_TIME.strftime('%H:%M:%S')} - {END_TIME.strftime('%H:%M:%S')}")
print("Mỗi người chỉ được điểm danh MỘT LẦN duy nhất trong ngày.")

# ================== REALTIME LOOP ==================
while True:
    ret, frame = cap.read()

    if not ret:
        print("Không đọc được webcam.")
        break

    frame_count += 1

    now = datetime.now()
    current_time = now.time()
    time_str_now = now.strftime("%H:%M:%S")
    today_str = now.strftime("%Y-%m-%d")

    if today_str != current_date_holder:
        marked_names = get_marked_names_today()
        current_date_holder = today_str
        print(f"=== Đã chuyển sang ngày mới {today_str}. Reset danh sách điểm danh. ===")

    if current_time < START_TIME:
        system_status = "TOO EARLY"
        sys_color = (0, 255, 255)
    elif START_TIME <= current_time <= END_TIME:
        system_status = "IN TIME"
        sys_color = (0, 255, 0)
    else:
        system_status = "OUT OF TIME"
        sys_color = (0, 0, 255)

    # ================== NHẬN DIỆN ==================
    if frame_count % SKIP_FRAMES == 0:
        try:
            small_frame = cv2.resize(frame, (320, 240))

            results = DeepFace.represent(
                img_path=small_frame,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False
            )

            last_results = []

            for res in results:
                embedding = res["embedding"]
                area = res["facial_area"]

                scale_x = frame.shape[1] / small_frame.shape[1]
                scale_y = frame.shape[0] / small_frame.shape[0]

                x = int(area["x"] * scale_x)
                y = int(area["y"] * scale_y)
                w = int(area["w"] * scale_x)
                h = int(area["h"] * scale_y)

                x, y, w, h = expand_face_box(x, y, w, h, frame.shape)

                name, confidence = recognize_face_svm(embedding)

                last_results.append((x, y, w, h, name, confidence))

        except Exception as e:
            print("Recognition error:", e)

    show_out_of_time_warning = False

    # ================== XỬ LÝ KẾT QUẢ ==================
    for (x, y, w, h, name, confidence) in last_results:
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, BOX_THICKNESS)

        label = f"{name} ({confidence:.1f}%)"

        (label_w, label_h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            LABEL_FONT_SCALE,
            LABEL_THICKNESS
        )

        label_y1 = max(0, y - label_h - baseline - 10)
        label_y2 = max(label_h + baseline + 5, y)

        cv2.rectangle(
            frame,
            (x, label_y1),
            (x + label_w + 12, label_y2),
            color,
            -1
        )

        cv2.putText(
            frame,
            label,
            (x + 6, label_y2 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            LABEL_FONT_SCALE,
            (255, 255, 255),
            LABEL_THICKNESS
        )

        if name != "Unknown":
            if name not in marked_names:
                now = datetime.now()
                current_time = now.time()
                time_str = now.strftime("%H:%M:%S")
                date_str = now.strftime("%Y-%m-%d")

                if current_time < START_TIME:
                    new_status = "Too Early"
                elif START_TIME <= current_time <= END_TIME:
                    new_status = "On"
                else:
                    new_status = "Off"
                    show_out_of_time_warning = True

                df.loc[len(df)] = [
                    name,
                    time_str,
                    new_status,
                    date_str,
                    f"{confidence:.1f}%"
                ]

                marked_names.add(name)
                save_attendance()

                print(f"[{time_str}] {name} - {new_status} - {confidence:.1f}%")

                cv2.putText(
                    frame,
                    new_status,
                    (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

            else:
                cv2.putText(
                    frame,
                    "Da diem danh roi!",
                    (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 165, 255),
                    2
                )

                existing = df[(df["Name"] == name) & (df["Date"] == today_str)]

                if not existing.empty:
                    old_status = existing.iloc[0]["Status"]

                    cv2.putText(
                        frame,
                        f"Trang thai: {old_status}",
                        (x, y + h + 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (200, 200, 200),
                        1
                    )

    # ================== CẢNH BÁO NGOÀI GIỜ ==================
    if system_status == "OUT OF TIME" and show_out_of_time_warning:
        blink_counter += 1

        if (blink_counter // 15) % 2 == 0:
            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (0, 120),
                (frame.shape[1], 180),
                (0, 0, 255),
                -1
            )

            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            cv2.putText(
                frame,
                "!! HET GIO - DIEM DANH NGOAI GIO (Off) !!",
                (50, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
    else:
        blink_counter = 0

    # ================== HIỂN THỊ THÔNG TIN ==================
    cv2.putText(
        frame,
        f"Time: {time_str_now}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Gio quy dinh: {START_TIME.strftime('%H:%M:%S')} - {END_TIME.strftime('%H:%M:%S')}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"He thong: {system_status}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        sys_color,
        2
    )

    # ================== DANH SÁCH ĐÃ ĐIỂM DANH ==================
    y_offset = frame.shape[0] - 30

    cv2.putText(
        frame,
        f"Hom nay ({today_str}):",
        (10, y_offset - 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        2
    )

    for i, marked_name in enumerate(list(marked_names)[-8:]):
        cv2.putText(
            frame,
            marked_name,
            (10, y_offset - 30 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    cv2.imshow("Smart Attendance - FaceNet512 + SVM", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break

# ================== KẾT THÚC ==================
save_attendance()
cap.release()
cv2.destroyAllWindows()

print("Da luu diem danh. Chuong trinh ket thuc.")