import cv2
import numpy as np
import pickle
import pandas as pd
import os
import json
from datetime import datetime, time
from PIL import Image, ImageTk
from deepface import DeepFace
from customtkinter import (
    CTkLabel, CTkScrollableFrame, CTkFrame, CTkButton,
    CTkImage, CTkEntry, CTkOptionMenu, CTkSwitch, CTkTextbox
)

class FaceRecognitionPage(CTkScrollableFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs, fg_color="transparent")

        # -------------------- ICONS --------------------
        self.start_img = CTkImage(dark_image=Image.open("app/assets/img/start.png"), size=(45, 45))
        self.stop_img = CTkImage(dark_image=Image.open("app/assets/img/stop.png"), size=(45, 45))

        # -------------------- DEFAULT CONFIG --------------------
        self.config_file = "app/config/face_recognition_config.json"
        self.load_config()

        # -------------------- BACKEND CONFIG --------------------
        self.MODEL_NAME = "Facenet512"
        self.CONFIDENCE_THRESHOLD = 60
        self.SKIP_FRAMES = 3
        self.DETECTOR_BACKEND = "opencv"
        self.MODEL_PATH = "train_models/examples/models/Facenet512_retinaface_001.pkl"

        # -------------------- DATA --------------------
        self.load_models_and_data()
        self.frame_count = 0
        self.last_results = []
        self.blink_counter = 0

        # -------------------- CAMERA STATE --------------------
        self.is_running = False
        self.cap = None

        # -------------------- UI LAYOUT --------------------
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)  # webcam
        self.grid_rowconfigure(1, weight=0)  # buttons
        self.grid_rowconfigure(2, weight=1)  # status + attendance
        self.grid_rowconfigure(3, weight=0)  # settings

        # 1. WEBCAM CARD
        self.webcam_card = CTkFrame(self, corner_radius=15, border_width=2, border_color="gray40")
        self.webcam_card.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        self.webcam_card.grid_columnconfigure(0, weight=1)

        self.webcam_label = CTkLabel(
            self.webcam_card, text="Webcam is Offline",
            width=640, height=480, fg_color="gray20", corner_radius=10
        )
        self.webcam_label.grid(row=0, column=0, padx=10, pady=10)

        # 2. BUTTON FRAME
        self.controls_frame = CTkFrame(self, fg_color="transparent")
        self.controls_frame.grid(row=1, column=0, pady=10)

        self.start_btn = CTkButton(
            self.controls_frame, text="Start Webcam", font=("Times New Roman", 16, "bold"),
            border_width=3, image=self.start_img, hover_color="gray10",
            corner_radius=30, command=self.start_webcam
        )
        self.start_btn.grid(row=0, column=0, padx=10)

        self.stop_btn = CTkButton(
            self.controls_frame, text="Stop Webcam", font=("Times New Roman", 16, "bold"),
            border_width=3, image=self.stop_img, fg_color="transparent",
            corner_radius=30, command=self.stop_webcam
        )
        self.stop_btn.grid(row=0, column=1, padx=10)

        # 3. BOTTOM AREA: System Status (left) + Attendance (right)
        self.bottom_frame = CTkFrame(self, fg_color="transparent")
        self.bottom_frame.grid(row=2, column=0, padx=10, pady=(5, 5), sticky="nsew")
        self.bottom_frame.grid_columnconfigure(0, weight=1)
        self.bottom_frame.grid_columnconfigure(1, weight=1)
        self.bottom_frame.grid_rowconfigure(0, weight=1)

        # ---- System Status Card ----
        self.status_card = CTkFrame(self.bottom_frame, corner_radius=15, border_width=2, border_color="gray40")
        self.status_card.grid(row=0, column=0, padx=(0, 5), sticky="nsew")
        self.status_card.grid_columnconfigure(0, weight=1)

        CTkLabel(self.status_card, text="System Status", font=("Times New Roman", 18, "bold")).grid(
            row=0, column=0, pady=(10, 5)
        )
        self.status_label = CTkLabel(self.status_card, text="Status: --", font=("Times New Roman", 14))
        self.status_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.time_label = CTkLabel(self.status_card, text="Current Time: --", font=("Times New Roman", 14))
        self.time_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.range_label = CTkLabel(self.status_card, text="", font=("Times New Roman", 14))
        self.range_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.marked_count_label = CTkLabel(self.status_card, text="Marked Today: 0", font=("Times New Roman", 14, "bold"))
        self.marked_count_label.grid(row=4, column=0, padx=10, pady=(5, 10), sticky="w")

        # ---- Attendance Card ----
        self.attendance_card = CTkFrame(self.bottom_frame, corner_radius=15, border_width=2, border_color="gray40")
        self.attendance_card.grid(row=0, column=1, padx=(5, 0), sticky="nsew")
        self.attendance_card.grid_columnconfigure(0, weight=1)
        self.attendance_card.grid_rowconfigure(1, weight=1)

        CTkLabel(self.attendance_card, text="Today's Attendance", font=("Times New Roman", 18, "bold")).grid(
            row=0, column=0, pady=(10, 5)
        )
        self.attendance_text = CTkTextbox(self.attendance_card, font=("Courier New", 12), wrap="none")
        self.attendance_text.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")

        # 4. SETTINGS FRAME
        self.settings_frame = CTkFrame(self, corner_radius=15, border_width=2, border_color="gray40")
        self.settings_frame.grid(row=3, column=0, padx=10, pady=(5, 10), sticky="ew")
        self.settings_frame.grid_columnconfigure(0, weight=0)
        self.settings_frame.grid_columnconfigure(1, weight=1)
        self.settings_frame.grid_columnconfigure(2, weight=0)
        self.settings_frame.grid_columnconfigure(3, weight=1)

        CTkLabel(self.settings_frame, text="Settings", font=("Times New Roman", 18, "bold")).grid(
            row=0, column=0, columnspan=4, pady=(10, 5)
        )

        # Start time
        CTkLabel(self.settings_frame, text="Start Time (HH:MM:SS):", font=("Times New Roman", 14)).grid(
            row=1, column=0, padx=10, pady=5, sticky="e"
        )
        self.start_time_entry = CTkEntry(self.settings_frame, width=100)
        self.start_time_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.start_time_entry.insert(0, self.START_TIME.strftime("%H:%M:%S"))

        # End time
        CTkLabel(self.settings_frame, text="End Time (HH:MM:SS):", font=("Times New Roman", 14)).grid(
            row=2, column=0, padx=10, pady=5, sticky="e"
        )
        self.end_time_entry = CTkEntry(self.settings_frame, width=100)
        self.end_time_entry.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        self.end_time_entry.insert(0, self.END_TIME.strftime("%H:%M:%S"))

        # Shift selection
        CTkLabel(self.settings_frame, text="Shift:", font=("Times New Roman", 14)).grid(
            row=1, column=2, padx=10, pady=5, sticky="e"
        )
        self.shift_option = CTkOptionMenu(
            self.settings_frame,
            values=["Morning (08:00-12:00)", "Afternoon (13:00-17:00)", "Evening (18:00-22:00)", "Custom"],
            command=self.on_shift_selected,
            width=180
        )
        self.shift_option.grid(row=1, column=3, padx=10, pady=5, sticky="w")
        self.shift_option.set(self.current_shift_name())

        # CSV file path
        CTkLabel(self.settings_frame, text="CSV File:", font=("Times New Roman", 14)).grid(
            row=2, column=2, padx=10, pady=5, sticky="e"
        )
        self.csv_entry = CTkEntry(self.settings_frame, width=250)
        self.csv_entry.grid(row=2, column=3, padx=10, pady=5, sticky="w")
        self.csv_entry.insert(0, self.ATTENDANCE_FILE)

        # Buttons
        self.apply_btn = CTkButton(
            self.settings_frame, text="Apply Settings", font=("Times New Roman", 14),
            command=self.apply_settings, width=120
        )
        self.apply_btn.grid(row=3, column=0, columnspan=2, pady=10)

        self.save_config_btn = CTkButton(
            self.settings_frame, text="Save as Default", font=("Times New Roman", 14),
            command=self.save_current_config, width=120
        )
        self.save_config_btn.grid(row=3, column=2, columnspan=2, pady=10)

        # Update range label
        self.update_range_label()

        # Start periodic UI updates
        self.update_ui_info()

        # Cleanup
        self.bind("<Destroy>", self.on_close)

    # -------------------- CONFIG MANAGEMENT --------------------
    def load_config(self):
        default_config = {
            "START_TIME": "00:18:00",
            "END_TIME": "00:19:00",
            "ATTENDANCE_FILE": "attendance/attendance_001.csv"
        }
        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as f:
                cfg = json.load(f)
        else:
            cfg = default_config
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, "w") as f:
                json.dump(cfg, f, indent=4)

        self.START_TIME = datetime.strptime(cfg["START_TIME"], "%H:%M:%S").time()
        self.END_TIME = datetime.strptime(cfg["END_TIME"], "%H:%M:%S").time()
        self.ATTENDANCE_FILE = cfg["ATTENDANCE_FILE"]

    def save_current_config(self):
        cfg = {
            "START_TIME": self.START_TIME.strftime("%H:%M:%S"),
            "END_TIME": self.END_TIME.strftime("%H:%M:%S"),
            "ATTENDANCE_FILE": self.ATTENDANCE_FILE
        }
        with open(self.config_file, "w") as f:
            json.dump(cfg, f, indent=4)

    def current_shift_name(self):
        shifts = {
            ("08:00:00", "12:00:00"): "Morning (08:00-12:00)",
            ("13:00:00", "17:00:00"): "Afternoon (13:00-17:00)",
            ("18:00:00", "22:00:00"): "Evening (18:00-22:00)"
        }
        current = (self.START_TIME.strftime("%H:%M:%S"), self.END_TIME.strftime("%H:%M:%S"))
        return shifts.get(current, "Custom")

    def on_shift_selected(self, choice):
        if choice == "Morning (08:00-12:00)":
            self.start_time_entry.delete(0, "end")
            self.start_time_entry.insert(0, "08:00:00")
            self.end_time_entry.delete(0, "end")
            self.end_time_entry.insert(0, "12:00:00")
        elif choice == "Afternoon (13:00-17:00)":
            self.start_time_entry.delete(0, "end")
            self.start_time_entry.insert(0, "13:00:00")
            self.end_time_entry.delete(0, "end")
            self.end_time_entry.insert(0, "17:00:00")
        elif choice == "Evening (18:00-22:00)":
            self.start_time_entry.delete(0, "end")
            self.start_time_entry.insert(0, "18:00:00")
            self.end_time_entry.delete(0, "end")
            self.end_time_entry.insert(0, "22:00:00")
        # Custom: giữ nguyên giá trị hiện tại

    def apply_settings(self):
        try:
            new_start = datetime.strptime(self.start_time_entry.get(), "%H:%M:%S").time()
            new_end = datetime.strptime(self.end_time_entry.get(), "%H:%M:%S").time()
            new_csv = self.csv_entry.get().strip()

            self.START_TIME = new_start
            self.END_TIME = new_end
            self.ATTENDANCE_FILE = new_csv

            # Reload attendance data from new file
            self.load_models_and_data()
            self.update_range_label()

            # Update shift dropdown display
            self.shift_option.set(self.current_shift_name())

        except ValueError:
            # Invalid time format
            pass

    def update_range_label(self):
        self.range_label.configure(
            text=f"Valid Hours: {self.START_TIME.strftime('%H:%M:%S')} - {self.END_TIME.strftime('%H:%M:%S')}"
        )

    # -------------------- BACKEND METHODS (cập nhật để dùng self.ATTENDANCE_FILE) --------------------
    def load_models_and_data(self):
        try:
            with open(self.MODEL_PATH, "rb") as f:
                self.centroids = pickle.load(f)
        except FileNotFoundError:
            print(f"Warning: Model not found at {self.MODEL_PATH}")

        if not os.path.exists(self.ATTENDANCE_FILE):
            self.df = pd.DataFrame(columns=["Name", "Time", "Status", "Date"])
        else:
            self.df = pd.read_csv(self.ATTENDANCE_FILE)

        if "Date" not in self.df.columns:
            self.df["Date"] = ""

        self.marked_names = self.get_marked_names_today()
        self.current_date_holder = datetime.now().strftime("%Y-%m-%d")

    def get_marked_names_today(self):
        today_str = datetime.now().strftime("%Y-%m-%d")
        today_df = self.df[self.df["Date"] == today_str]
        return set(today_df["Name"].values)

    def save_attendance(self):
        os.makedirs(os.path.dirname(self.ATTENDANCE_FILE), exist_ok=True)
        self.df.to_csv(self.ATTENDANCE_FILE, index=False)

    def start_webcam(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.is_running = True
            self.webcam_label.configure(text="")
            self.update_webcam()

    def stop_webcam(self):
        self.is_running = False
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        if self.webcam_label.winfo_exists():
            self.webcam_label.configure(image="", text="Webcam is Offline")
            self.webcam_label.image = None

    def update_webcam(self):
        if not self.is_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.after(10, self.update_webcam)
            return

        frame = cv2.flip(frame, 1)
        self.frame_count += 1
        now = datetime.now()
        current_time = now.time()
        today_str = now.strftime("%Y-%m-%d")
        time_str_now = now.strftime("%H:%M:%S")

        if today_str != self.current_date_holder:
            self.marked_names = self.get_marked_names_today()
            self.current_date_holder = today_str

        # System status
        if current_time < self.START_TIME:
            system_status, sys_color = "TOO EARLY", (0, 255, 255)
        elif self.START_TIME <= current_time <= self.END_TIME:
            system_status, sys_color = "IN TIME", (0, 255, 0)
        else:
            system_status, sys_color = "OUT OF TIME", (0, 0, 255)

        # Recognition logic (giữ nguyên)
        if self.frame_count % self.SKIP_FRAMES == 0:
            try:
                small_frame = cv2.resize(frame, (320, 240))
                results = DeepFace.represent(
                    img_path=small_frame,
                    model_name=self.MODEL_NAME,
                    enforce_detection=False,
                    detector_backend=self.DETECTOR_BACKEND
                )
                self.last_results = []
                for res in results:
                    emb = res["embedding"]
                    area = res["facial_area"]
                    scale_x, scale_y = frame.shape[1] / 320, frame.shape[0] / 240
                    x = int(area["x"] * scale_x)
                    y = int(area["y"] * scale_y)
                    w = int(area["w"] * scale_x)
                    h = int(area["h"] * scale_y)

                    best_name = "Unknown"
                    best_sim = 0.0
                    for cname, centroid in self.centroids.items():
                        sim = np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid))
                        if sim > best_sim:
                            best_sim = sim
                            best_name = cname
                    conf = best_sim * 100
                    name = best_name if conf >= self.CONFIDENCE_THRESHOLD else "Unknown"
                    self.last_results.append((x, y, w, h, name, conf))
            except Exception:
                pass

        # Draw on frame (giữ nguyên logic)
        show_out_of_time_warning = False
        for (x, y, w, h, name, conf) in self.last_results:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{name} ({conf:.1f}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if name != "Unknown":
                if name not in self.marked_names:
                    if current_time < self.START_TIME:
                        status = "Too Early"
                    elif self.START_TIME <= current_time <= self.END_TIME:
                        status = "On"
                    else:
                        status = "Off"
                        show_out_of_time_warning = True

                    self.df.loc[len(self.df)] = [name, time_str_now, status, today_str]
                    self.marked_names.add(name)
                    self.save_attendance()
                    cv2.putText(frame, status, (x, y + h + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    cv2.putText(frame, "Already marked", (x, y + h + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                    existing = self.df[(self.df["Name"] == name) & (self.df["Date"] == today_str)]
                    if not existing.empty:
                        old_status = existing.iloc[0]["Status"]
                        cv2.putText(frame, f"Status: {old_status}", (x, y + h + 45),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Blinking warning
        if system_status == "OUT OF TIME" and show_out_of_time_warning:
            self.blink_counter += 1
            if (self.blink_counter // 15) % 2 == 0:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 120), (frame.shape[1], 180), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                cv2.putText(frame, "!! LATE - OUT OF TIME MARKING !!",
                            (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            self.blink_counter = 0

        # Display info on frame
        cv2.putText(frame, f"Time: {time_str_now}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Valid: {self.START_TIME.strftime('%H:%M:%S')} - {self.END_TIME.strftime('%H:%M:%S')}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"System: {system_status}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, sys_color, 2)

        # Convert for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.webcam_label.configure(image=img_tk)
        self.webcam_label.image = img_tk

        self.after(10, self.update_webcam)

    # -------------------- UI UPDATE --------------------
    def update_ui_info(self):
        if not self.winfo_exists():
            return

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        today_str = now.strftime("%Y-%m-%d")

        current_t = now.time()
        if current_t < self.START_TIME:
            status = "TOO EARLY"
            color = "yellow"
        elif self.START_TIME <= current_t <= self.END_TIME:
            status = "IN TIME"
            color = "green"
        else:
            status = "OUT OF TIME"
            color = "red"

        self.status_label.configure(text=f"Status: {status}", text_color=color)
        self.time_label.configure(text=f"Current Time: {current_time}")
        self.marked_count_label.configure(text=f"Marked Today: {len(self.marked_names)}")

        # Update attendance table
        today_df = self.df[self.df["Date"] == today_str]
        if today_df.empty:
            self.attendance_text.delete("1.0", "end")
            self.attendance_text.insert("1.0", "No attendance records yet.")
        else:
            lines = ["{:<20} {:<10} {:<10}".format("Name", "Time", "Status")]
            lines.append("-" * 42)
            for _, row in today_df.iterrows():
                lines.append("{:<20} {:<10} {:<10}".format(row["Name"], row["Time"], row["Status"]))
            self.attendance_text.delete("1.0", "end")
            self.attendance_text.insert("1.0", "\n".join(lines))

        self.after(1000, self.update_ui_info)

    def on_close(self, event=None):
        self.stop_webcam()
