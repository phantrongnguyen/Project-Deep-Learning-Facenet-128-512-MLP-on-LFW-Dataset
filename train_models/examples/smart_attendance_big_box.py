import cv2
import numpy as np
import pickle
import pandas as pd
from datetime import datetime, time
from deepface import DeepFace
import os

# ===== CẤU HÌNH =====
MODEL_NAME = "Facenet512"
CONFIDENCE_THRESHOLD = 60
SKIP_FRAMES = 3
DETECTOR_BACKEND = "opencv"

# ===== CẤU HÌNH KHUNG NHẬN DIỆN =====
# Tăng FRAME_PADDING_RATIO nếu muốn khung bao quanh mặt to hơn nữa.
FRAME_PADDING_RATIO = 0.35
MIN_BOX_PADDING = 35
BOX_THICKNESS = 4
LABEL_FONT_SCALE = 0.75
LABEL_THICKNESS = 2

MODEL_PATH = "train_models/examples/models/Facenet512_mtcnn_001.pkl"
ATTENDANCE_DIR = "train_models/examples/csv"
ATTENDANCE_BASE = "attendance_001"

START_TIME = time(14, 43, 0)
END_TIME   = time(14, 44, 0)

# ===== Load model =====
with open(MODEL_PATH, "rb") as f:
    centroids = pickle.load(f)

# ===== Tạo file CSV mới cho phiên này =====
os.makedirs(ATTENDANCE_DIR, exist_ok=True)
session_start = datetime.now()
session_tag = session_start.strftime("%Y%m%d_%H%M%S")
ATTENDANCE_FILE = f"{ATTENDANCE_DIR}/{ATTENDANCE_BASE}_{session_tag}.csv"
print(f"File ghi điểm danh: {ATTENDANCE_FILE}")

COLUMNS = ["Name", "Time", "Status", "Date", "Confidence"]
df = pd.DataFrame(columns=COLUMNS)
df["Confidence"] = df["Confidence"].astype(object)

def save_attendance():
    df.to_csv(ATTENDANCE_FILE, index=False)

def expand_face_box(x, y, w, h, frame_shape, padding_ratio=FRAME_PADDING_RATIO, min_padding=MIN_BOX_PADDING):
    """Nới rộng khung nhận diện nhưng không vượt ra ngoài kích thước frame."""
    pad_x = max(int(w * padding_ratio), min_padding)
    pad_y = max(int(h * padding_ratio), min_padding)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame_shape[1] - 1, x + w + pad_x)
    y2 = min(frame_shape[0] - 1, y + h + pad_y)

    return x1, y1, x2 - x1, y2 - y1

# ===== Hàm lấy danh sách đã điểm danh hôm nay =====
def get_marked_names_today():
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_df = df[df["Date"] == today_str]
    return set(today_df["Name"].values)

marked_names = get_marked_names_today()
# Lưu ngày hiện tại để kiểm tra sang ngày mới
current_date_holder = datetime.now().strftime("%Y-%m-%d")

# ===== Khởi tạo webcam =====
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)

frame_count = 0
last_results = []
blink_counter = 0

print("Nhấn Q hoặc ESC để thoát")
print(f"Khung giờ điểm danh: {START_TIME.strftime('%H:%M:%S')} - {END_TIME.strftime('%H:%M:%S')}")
print("Mỗi người chỉ được điểm danh MỘT LẦN duy nhất trong ngày.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    now = datetime.now()
    current_time = now.time()
    time_str_now = now.strftime("%H:%M:%S")
    today_str = now.strftime("%Y-%m-%d")

    # Kiểm tra sang ngày mới -> reset danh sách đã điểm danh
    if today_str != current_date_holder:
        marked_names = get_marked_names_today()
        current_date_holder = today_str
        print(f"=== Đã chuyển sang ngày mới {today_str}. Reset danh sách điểm danh. ===")

    # Trạng thái hệ thống (dựa trên khung giờ)
    if current_time < START_TIME:
        system_status = "TOO EARLY"
        sys_color = (0, 255, 255)
    elif START_TIME <= current_time <= END_TIME:
        system_status = "IN TIME"
        sys_color = (0, 255, 0)
    else:
        system_status = "OUT OF TIME"
        sys_color = (0, 0, 255)

    # Nhận diện khuôn mặt (skip frame)
    if frame_count % SKIP_FRAMES == 0:
        try:
            small_frame = cv2.resize(frame, (320, 240))
            results = DeepFace.represent(
                img_path=small_frame,
                model_name=MODEL_NAME,
                enforce_detection=False,
                detector_backend=DETECTOR_BACKEND
            )
            last_results = []
            for res in results:
                emb = res["embedding"]
                area = res["facial_area"]
                scale_x = frame.shape[1] / small_frame.shape[1]
                scale_y = frame.shape[0] / small_frame.shape[0]
                x = int(area["x"] * scale_x)
                y = int(area["y"] * scale_y)
                w = int(area["w"] * scale_x)
                h = int(area["h"] * scale_y)

                # Nới rộng khung nhận diện để hiển thị rõ hơn trên webcam
                x, y, w, h = expand_face_box(x, y, w, h, frame.shape)

                best_name = "Unknown"
                best_sim = 0.0
                for cname, centroid in centroids.items():
                    sim = np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid))
                    if sim > best_sim:
                        best_sim = sim
                        best_name = cname
                confidence = best_sim * 100
                name = best_name if confidence >= CONFIDENCE_THRESHOLD else "Unknown"
                last_results.append((x, y, w, h, name, confidence))
        except:
            pass

    show_out_of_time_warning = False  # cờ hiển thị cảnh báo ngoài giờ

    # Xử lý từng khuôn mặt
    for (x, y, w, h, name, confidence) in last_results:
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        # Vẽ khung và tên: khung to hơn, dày hơn, chữ dễ nhìn hơn
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, BOX_THICKNESS)
        label = f"{name} ({confidence:.1f}%)"

        # Vẽ nền cho label để không bị chìm vào video
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_THICKNESS
        )
        label_y1 = max(0, y - label_h - baseline - 10)
        label_y2 = max(label_h + baseline + 5, y)
        cv2.rectangle(frame, (x, label_y1), (x + label_w + 12, label_y2), color, -1)
        cv2.putText(frame, label, (x + 6, label_y2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, (255, 255, 255), LABEL_THICKNESS)

        if name != "Unknown":
            # ----- KIỂM TRA ĐÃ ĐIỂM DANH HÔM NAY CHƯA -----
            if name not in marked_names:
                # Chưa điểm danh -> ghi nhận lần đầu
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
                    show_out_of_time_warning = True  # bật cảnh báo ngoài giờ

                # Thêm vào DataFrame và lưu
                df.loc[len(df)] = [name, time_str, new_status, date_str, f"{confidence:.1f}%"]
                marked_names.add(name)
                save_attendance()
                print(f"[{time_str}] {name} - {new_status} (lần đầu trong ngày)")

                # Hiển thị trạng thái bên dưới khung mặt
                cv2.putText(frame, new_status, (x, y + h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                # Đã điểm danh rồi -> hiển thị thông báo
                cv2.putText(frame, "Da diem danh roi!", (x, y + h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                # (Tuỳ chọn) Hiển thị trạng thái đã lưu
                existing = df[(df["Name"] == name) & (df["Date"] == today_str)]
                if not existing.empty:
                    old_status = existing.iloc[0]["Status"]
                    cv2.putText(frame, f"Trang thai: {old_status}", (x, y + h + 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # ===== Cảnh báo ngoài giờ (chỉ khi có người mới điểm danh ngoài giờ) =====
    if system_status == "OUT OF TIME" and show_out_of_time_warning:
        blink_counter += 1
        if (blink_counter // 15) % 2 == 0:  # nhấp nháy
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 120), (frame.shape[1], 180), (0, 0, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            cv2.putText(frame, "!! HET GIO - DIEM DANH NGOAI GIO (Off) !!",
                        (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        blink_counter = 0

    # ===== Hiển thị thông tin hệ thống =====
    cv2.putText(frame, f"Time: {time_str_now}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Gio quy dinh: {START_TIME.strftime('%H:%M:%S')} - {END_TIME.strftime('%H:%M:%S')}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"He thong: {system_status}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, sys_color, 2)

    # ===== Danh sách đã điểm danh trong ngày =====
    y_offset = frame.shape[0] - 30
    cv2.putText(frame, f"Hom nay ({today_str}):", (10, y_offset - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    for i, name in enumerate(list(marked_names)[-8:]):
        cv2.putText(frame, name, (10, y_offset - 30 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Smart Attendance - Chi mot lan", frame)
    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
        break

# ===== Kết thúc =====
save_attendance()
cap.release()
cv2.destroyAllWindows()
print("Da luu diem danh. Chuong trinh ket thuc.")