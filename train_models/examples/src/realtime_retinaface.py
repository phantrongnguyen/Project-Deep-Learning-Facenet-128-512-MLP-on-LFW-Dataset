import cv2
import numpy as np
import pickle
import pandas as pd
import threading
import time as time_module
from datetime import datetime, time
from deepface import DeepFace
import os

# ================== CẤU HÌNH ==================
MODEL_NAME = "Facenet512"
CONFIDENCE_THRESHOLD = 65
SKIP_FRAMES = 15
DETECTOR_BACKEND = "opencv"

MODEL_PATH = "train_models/examples/models/Facenet512_retinaface_004.pkl"

ATTENDANCE_DIR = "train_models/examples/csv"
ATTENDANCE_BASE = "attendance_004"

FRAME_PADDING_RATIO = 0.35
MIN_BOX_PADDING = 35
BOX_THICKNESS = 3

START_TIME = time(14, 43, 0)
END_TIME   = time(14, 44, 0)

# ================== LOAD MODEL ==================
with open(MODEL_PATH, "rb") as f:
    raw = pickle.load(f)

if "centroids" in raw:
    centroids = raw["centroids"]
    print(f"Loaded model version: {raw.get('version', 'N/A')}")
    print(f"Persons: {raw.get('persons', list(centroids.keys()))}")
else:
    centroids = raw
    print(f"Loaded flat centroid dict with {len(centroids)} person(s)")

# ================== TẠO FILE CSV ==================
os.makedirs(ATTENDANCE_DIR, exist_ok=True)
session_start = datetime.now()
session_tag = session_start.strftime("%Y%m%d_%H%M%S")
ATTENDANCE_FILE = f"{ATTENDANCE_DIR}/{ATTENDANCE_BASE}_{session_tag}.csv"
print(f"File ghi điểm danh: {ATTENDANCE_FILE}")

COLUMNS = ["Name", "Time", "Status", "Date", "Confidence"]
df = pd.DataFrame(columns=COLUMNS)
recognition_lock = threading.Lock()

def save_attendance():
    df.to_csv(ATTENDANCE_FILE, index=False)

def get_marked_names_today():
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_df = df[df["Date"] == today_str]
    return set(today_df["Name"].values)

def expand_face_box(x, y, w, h, frame_shape, padding_ratio=FRAME_PADDING_RATIO, min_padding=MIN_BOX_PADDING):
    pad_x = max(int(w * padding_ratio), min_padding)
    pad_y = max(int(h * padding_ratio), min_padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame_shape[1] - 1, x + w + pad_x)
    y2 = min(frame_shape[0] - 1, y + h + pad_y)
    return x1, y1, x2 - x1, y2 - y1

marked_names = get_marked_names_today()
current_date_holder = datetime.now().strftime("%Y-%m-%d")

# ================== HÀM NHẬN DIỆN CENTROID ==================
def recognize_face(embedding):
    best_name = "Unknown"
    best_sim = 0.0
    emb_norm = np.linalg.norm(embedding)
    if emb_norm == 0:
        return "Unknown", 0.0
    for cname, centroid in centroids.items():
        sim = np.dot(embedding, centroid) / (emb_norm * np.linalg.norm(centroid))
        if sim > best_sim:
            best_sim = sim
            best_name = cname
    confidence = best_sim * 100
    if confidence >= CONFIDENCE_THRESHOLD:
        return best_name, confidence
    return "Unknown", confidence

# ================== THREAD XỬ LÝ NHẬN DIỆN ==================
pending_frame = None
pending_results = []
results_lock = threading.Lock()

def process_faces():
    global pending_frame, pending_results
    while True:
        if pending_frame is not None:
            frame_copy = pending_frame
            pending_frame = None
            try:
                small = cv2.resize(frame_copy, (420, 340))
                results = DeepFace.represent(
                    img_path=small,
                    model_name=MODEL_NAME,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=False
                )
                local_results = []
                for res in results:
                    emb = res["embedding"]
                    area = res["facial_area"]
                    sx = frame_copy.shape[1] / small.shape[1]
                    sy = frame_copy.shape[0] / small.shape[0]
                    x = int(area["x"] * sx)
                    y = int(area["y"] * sy)
                    w = int(area["w"] * sx)
                    h = int(area["h"] * sy)
                    x, y, w, h = expand_face_box(x, y, w, h, frame_copy.shape)
                    name, conf = recognize_face(emb)
                    local_results.append((x, y, w, h, name, conf))
                with results_lock:
                    pending_results.clear()
                    pending_results.extend(local_results)
            except Exception:
                pass
        time_module.sleep(0.005)

threading.Thread(target=process_faces, daemon=True).start()

# ================== KHỞI TẠO CAMERA ==================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 940)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 680)

frame_count = 0
last_results = []
blink_counter = 0

print("Nhấn Q hoặc ESC để thoát.")
print(f"Khung giờ điểm danh: {START_TIME.strftime('%H:%M:%S')} - {END_TIME.strftime('%H:%M:%S')}")
print("Mỗi người chỉ được điểm danh một lần trong ngày.")

# ================== VÒNG LẶP REALTIME ==================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Không đọc được camera.")
        break

    frame = cv2.flip(frame, 1)

    frame_count += 1
    now = datetime.now()
    current_time = now.time()
    time_str_now = now.strftime("%H:%M:%S")
    today_str = now.strftime("%Y-%m-%d")

    if today_str != current_date_holder:
        marked_names = get_marked_names_today()
        current_date_holder = today_str
        print(f"=== Sang ngày mới {today_str}. Reset danh sách điểm danh. ===")

    if current_time < START_TIME:
        system_status = "TOO EARLY"
        sys_color = (0, 255, 255)
    elif START_TIME <= current_time <= END_TIME:
        system_status = "IN TIME"
        sys_color = (0, 255, 0)
    else:
        system_status = "OUT OF TIME"
        sys_color = (0, 0, 255)

    # ================== NHẬN DIỆN KHUÔN MẶT (thread) ==================
    if frame_count % SKIP_FRAMES == 0:
        pending_frame = frame.copy()

    with results_lock:
        if pending_results:
            last_results = list(pending_results)

    show_out_of_time_warning = False

    # ================== XỬ LÝ KẾT QUẢ ==================
    for (x, y, w, h, name, confidence) in last_results:
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, BOX_THICKNESS)
        label = f"{name} ({confidence:.1f}%)"

        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        label_y1 = max(0, y - label_h - baseline - 10)
        label_y2 = max(label_h + baseline + 5, y)
        cv2.rectangle(frame, (x, label_y1), (x + label_w + 12, label_y2), color, -1)
        cv2.putText(frame, label, (x + 6, label_y2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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

                with recognition_lock:
                    if name not in marked_names:
                        df.loc[len(df)] = [name, time_str, new_status, date_str, f"{confidence:.1f}%"]
                        marked_names.add(name)
                        save_attendance()
                        print(f"[{time_str}] {name} - {new_status} - {confidence:.1f}%")

                cv2.putText(frame, new_status, (x, y + h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                cv2.putText(frame, "Da diem danh roi!", (x, y + h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                existing = df[(df["Name"] == name) & (df["Date"] == today_str)]
                if not existing.empty:
                    old_status = existing.iloc[0]["Status"]
                    cv2.putText(frame, f"Trang thai: {old_status}", (x, y + h + 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # ================== CẢNH BÁO NGOÀI GIỜ ==================
    if system_status == "OUT OF TIME" and show_out_of_time_warning:
        blink_counter += 1
        if (blink_counter // 15) % 2 == 0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 120), (frame.shape[1], 180), (0, 0, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            cv2.putText(frame, "!! HET GIO - DIEM DANH NGOAI GIO (Off) !!",
                        (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        blink_counter = 0

    # ================== HIỂN THỊ THÔNG TIN HỆ THỐNG ==================
    cv2.putText(frame, f"Time: {time_str_now}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Gio quy dinh: {START_TIME.strftime('%H:%M:%S')} - {END_TIME.strftime('%H:%M:%S')}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"He thong: {system_status}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, sys_color, 2)

    # ================== DANH SÁCH ĐÃ ĐIỂM DANH ==================
    y_offset = frame.shape[0] - 30
    cv2.putText(frame, f"Hom nay ({today_str}):", (10, y_offset - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    for i, marked_name in enumerate(list(marked_names)[-8:]):
        cv2.putText(frame, marked_name, (10, y_offset - 30 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Smart Attendance - FaceNet512 + RetinaFace", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

# ================== KẾT THÚC ==================
save_attendance()
cap.release()
cv2.destroyAllWindows()
print("Da luu diem danh. Chuong trinh ket thuc.")
