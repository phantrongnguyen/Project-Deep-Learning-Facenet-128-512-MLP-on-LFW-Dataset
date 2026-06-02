import customtkinter as ctk
from customtkinter import CTkLabel, CTkButton, CTkScrollableFrame, CTkImage, CTkProgressBar
import cv2
from PIL import Image, ImageTk
import threading
import time

from main_frame import MainFrame
from face_recognition import FaceRecognitionPage
from register_face import FaceRegisterPage
from attendance import AttendancePage
from face_verification import FaceVerificationPage

ctk.set_appearance_mode("system")
ctk.set_default_color_theme("dark-blue")


class App(ctk.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.title("Huit Face")
        self.geometry("1400x750")
        self.iconbitmap('app/assets/icons/huit.ico')

        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (1400 // 2)
        y = (self.winfo_screenheight() // 2) - (750 // 2)
        self.geometry(f"1400x750+{x}+{y}")
        
        # grid config 
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.load_image()
        self.setup_sidebar()
        self.setup_mainframe()
    
    def load_image(self):
        self.huit_logo = CTkImage(dark_image=Image.open("app/assets/img/huit.png"),
                                  light_image=Image.open("app/assets/img/huit.png"), size=(70,70))
        
        self.face_img = CTkImage(dark_image=Image.open("app/assets/img/eye-scan.png"),
                                 light_image=Image.open("app/assets/img/face-id.png"), size=(45,45))
        
        self.register_face_img = CTkImage(dark_image=Image.open("app/assets/img/face_registered.png"),
                                      light_image=Image.open("app/assets/img/face_registered_light.png"), size=(45,45))
        
        self.attendance_img = CTkImage(dark_image=Image.open("app/assets/img/antendance_dark.png"),
                                       light_image=Image.open("app/assets/img/antendance_light.png"), size=(45, 45))
        
        self.face_verification_img = CTkImage(dark_image=Image.open("app/assets/img/face_verification_dark.png"),
                                              light_image=Image.open("app/assets/img/face_verification_light.png"), size=(45,45))
    
    def setup_sidebar(self):
        self.sidebar = CTkScrollableFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(7, weight=1)
        self.sidebar.grid_columnconfigure(0, weight=1)
        
        # title logo app 
        self.logo_app = CTkLabel(self.sidebar, text=" HUIT FACE", font=("Times New Roman", 20, "bold"), image=self.huit_logo, compound="left")
        self.logo_app.grid(row=0, column=0, padx=0, pady=(20,20), sticky="w")
        
        # progress first 
        self.progress_first = CTkProgressBar(self.sidebar, mode="determinate", height=3)
        self.progress_first.grid(row=1, column=0, pady=(0,10), sticky="ew")
        self.progress_first.set(1)
        
        # face verification btn 
        self.face_ver_btn = CTkButton(self.sidebar, text="Face verification", font=("Times New Roman", 16, "bold"), fg_color="transparent", 
                                    hover_color=["#F3F3F3", "GRAY30"], corner_radius=20, height=60, image=self.face_verification_img,
                                    compound="left", anchor="w", command=self.show_face_verification)
        self.face_ver_btn.grid(row=2, column=0, pady=(0, 10), sticky="ew")
        
        # face recognition button 
        self.face_btn = CTkButton(self.sidebar, text="Face recognition", font=("Times New Roman", 16, "bold"), fg_color="transparent", 
                                    hover_color=["#F3F3F3", "GRAY30"], corner_radius=20, height=60, image=self.face_img,
                                    compound="left", anchor="w", command=self.show_face_recognition)
        self.face_btn.grid(row=3, column=0, pady=(0, 10), sticky="ew")
        
        # register face button
        self.register_face_btn = CTkButton(self.sidebar, text="Register face", font=("Times New Roman", 16, "bold"), fg_color="transparent",
                                           hover_color=["#F3F3F3", "GRAY30"], corner_radius=20, height=60, image=self.register_face_img,
                                           compound="left", anchor="w", command=self.show_face_register)
        self.register_face_btn.grid(row=4, column=0, pady=(0,10), sticky="ew")
        
        # attendance btn
        self.attendance_btn = CTkButton(self.sidebar, text="Attendance", font=("Times New Roman", 16, "bold"), fg_color="transparent",
                                           hover_color=["#F3F3F3", "GRAY30"], corner_radius=20, height=60, image=self.attendance_img,
                                           compound="left", anchor="w", command=self.show_attendance)
        self.attendance_btn.grid(row=5, column=0, pady=(0,10), sticky="ew")
                                                                            
    
    def setup_mainframe(self):
        self.main_frame = MainFrame(self)
        self.main_frame.grid(row=0, column=1, sticky="nsew")
        self.show_face_recognition()
    
    def show_face_recognition(self):
        self.main_frame.show_frame(FaceRecognitionPage)
    
    def show_face_register(self):
        self.main_frame.show_frame(FaceRegisterPage)
    
    def show_attendance(self):
        self.main_frame.show_frame(AttendancePage)
    
    def show_face_verification(self):
        self.main_frame.show_frame(FaceVerificationPage)
        