<p align="center">
  <img src="app/assets/img/huit.png" alt="Huit Face" width="200"/>
</p>

<h1 align="center">Face Recognition using FaceNet + MLP</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.0-orange?logo=tensorflow" alt="TensorFlow">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/FaceNet-128D%2F512D-yellow" alt="FaceNet">
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen" alt="Status">
</p>

<p align="center">
  <b>An end-to-end face recognition system with a desktop GUI, real-time webcam recognition, face verification, and automatic attendance tracking.</b>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Pipeline](#pipeline)
- [Experiments & Results](#experiments--results)
- [Technologies](#technologies)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Limitations & Future Work](#limitations--future-work)
- [License](#license)

---

## Overview

This project implements a complete facial recognition pipeline using **FaceNet** for feature extraction and a **Multi-Layer Perceptron (MLP)** classifier for identity prediction. The system is packaged as a desktop application (CustomTkinter) with real-time webcam recognition, face verification, face enrollment, and automated attendance tracking.

The pipeline was systematically evaluated on **Labeled Faces in the Wild (LFW)** — comparing FaceNet128 (128D) vs FaceNet512 (512D) embeddings across training sets of 3, 10, 20, and 30 images per identity.

**Best result:** FaceNet512 + MLP with 30 images/identity achieves **93.5% test accuracy** and **94.5% real-world confidence**.

---

## Features

- **Face Verification** – Compare two images and determine if they belong to the same person using cosine similarity, Euclidean distance, and MLP prediction.
- **Face Recognition** – Real-time webcam recognition with bounding boxes and confidence scores.
- **Face Registration** – Enroll new faces via webcam with auto-capture mode and duplicate detection.
- **Attendance System** – Automatic CSV logging with time-based access control, shift management (Morning/Afternoon/Evening/Custom), and duplicate marking prevention.
- **Multi-backend Detection** – OpenCV, RetinaFace, and MTCNN face detectors.
- **Desktop GUI** – CustomTkinter interface with dark mode support.

---

## Pipeline

```
Input Image
    │
    ▼
Face Detection (OpenCV / RetinaFace / MTCNN)
    │
    ▼
Face Alignment
    │
    ▼
Face Embedding (FaceNet128 / FaceNet512 via DeepFace)
    │
    ▼
MLP / SVM Classifier
    │
    ▼
Identity Prediction
```

---

## Experiments & Results

### Experimental Setup

| Parameter          | Value                        |
|--------------------|------------------------------|
| Dataset            | LFW (13,000+ images)         |
| Embedding Models   | FaceNet128 (128D), FaceNet512 (512D) |
| Classifier         | MLP (ReLU, BatchNorm, Dropout, Softmax) |
| Train/Test Split   | 70/30                        |
| Identities         | 30–900 (varies by experiment)|
| Images/Identity    | 3, 10, 20, 30                |

### Accuracy & Loss Comparison

| Images/Identity | Model       | Test Accuracy | Test Loss | EER     |
|----------------|-------------|---------------|-----------|---------|
| 3              | FaceNet128  | 80.8%         | 1.1911    | 9.59%   |
| 3              | FaceNet512  | 84.2%         | 1.0637    | 12.45%  |
| 10             | FaceNet128  | 87.7%         | 0.6844    | 5.66%   |
| 10             | FaceNet512  | 91.4%         | 0.5178    | 8.11%   |
| 20             | FaceNet128  | 89.4%         | 0.5496    | 6.25%   |
| 20             | FaceNet512  | 91.2%         | 0.4607    | 7.55%   |
| **30**         | FaceNet128  | 91.8%         | 0.4017    | 5.13%   |
| **30**         | FaceNet512  | **93.5%**     | **0.3374**| 9.68%   |

> Result figures (confusion matrices, EER curves, training history) are available under `logs/` and `results/images/`.

### Analysis

**Effect of training set size.** Accuracy gains are steepest between 3 and 10 images (+6.9–7.2%), then diminish beyond 20 images. This suggests ~10–20 well-labeled samples per class are sufficient for practical deployment.

**FaceNet128 vs FaceNet512.** FaceNet512 consistently outperforms FaceNet128 in classification accuracy (+1.8–3.7%), but FaceNet128 achieves lower EER (5.13% vs 9.68%), making it preferable for verification tasks where false accepts are costly.

**Generalization.** The train-test accuracy gap narrows from ~14% (3 images) to ~6% (30 images), confirming that regularization (Dropout, BatchNorm, Early Stopping) and more data effectively control overfitting.

**Real-time performance.** Average inference time is ~0.2–0.3 ms per image. In live webcam tests, the deployed FaceNet512 + SVM model achieves 85–95% confidence on registered faces.

---

## Technologies

| Category            | Technology                    |
|---------------------|-------------------------------|
| Language            | Python 3.9                    |
| Deep Learning       | TensorFlow / Keras            |
| Face Recognition    | DeepFace                      |
| Feature Extraction  | FaceNet128 / FaceNet512       |
| Classification      | MLP (Scikit-learn), SVM       |
| GUI                 | CustomTkinter                 |
| Data Processing     | NumPy, Pandas                 |
| Visualization       | Matplotlib, Seaborn           |
| Image Processing    | OpenCV                        |
| Dataset             | LFW (Labeled Faces in the Wild) |

---

## Dataset

**Labeled Faces in the Wild (LFW)** – 13,000+ face images of 5,700+ celebrities in unconstrained environments.

| Property          | Value                          |
|-------------------|--------------------------------|
| Total Images      | 13,000+                        |
| Total Identities  | 5,700+                         |
| Environment       | Unconstrained (web-collected)  |
| Variations        | Lighting, pose, expression, background, image quality |

---

## Installation

```bash
# Clone
git clone https://github.com/<your-username>/facenet_mlp.git
cd facenet_mlp

# Install
pip install -r requirements.txt

# Run
python -m app.ui
```

**requirements.txt:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `customtkinter`, `deepface`, `tensorflow`, `opencv-python`, `joblib`, `tqdm`, `scikit-learn`

---

## Usage

### Desktop Application

```bash
python -m app.ui
```

Four main pages:

| Page               | Description                                                    |
|--------------------|----------------------------------------------------------------|
| Face Verification  | Upload two images → returns match decision with confidence     |
| Face Recognition   | Real-time webcam → bounding boxes, names, confidence scores    |
| Register Face      | Webcam enrollment with auto-capture and duplicate checking     |
| Attendance         | View attendance logs (auto-populated during recognition)       |

### Training & Evaluation (Notebooks)

```bash
jupyter notebook notebooks/
```

| Notebook                                   | Purpose                                  |
|--------------------------------------------|------------------------------------------|
| `01_data_filtering.ipynb`                  | Dataset filtering and preparation        |
| `02_face_recognition_pipeline_v21.ipynb`   | Main training pipeline (FaceNet + MLP)   |
| `03_pipeline.ipynb`                        | Consolidated/updated pipeline            |
| `model_002.ipynb`                          | Model experiment #2                      |
| `test_facenet512.ipynb`                    | FaceNet512 testing                       |
| `understanding_model.ipynb`                | Model behavior analysis                  |

---

## Project Structure

```
facenet_mlp/
│
├── app/                           # Desktop application
│   ├── ui/                        # GUI pages (CustomTkinter)
│   │   ├── app.py                 # Main window (title "Huit Face", 1400×750)
│   │   ├── main_frame.py          # Page router
│   │   ├── face_recognition.py    # Real-time recognition + attendance
│   │   ├── face_verification.py   # Face verification with MLP
│   │   ├── register_face.py       # Face enrollment via webcam
│   │   └── attendance.py          # Attendance page stub
│   ├── config/                    # App configuration (JSON)
│   ├── models/                    # Deployed models (MLP, SVM, label map, normalizer)
│   ├── data/                      # Registered face embeddings (.pkl)
│   ├── assets/                    # Icons and images (UI assets)
│   ├── attendance/                # Attendance CSV logs
│   ├── core/                      # Core logic (WIP)
│   └── utils/                     # Utilities (WIP)
│
├── notebooks/                     # Jupyter notebooks
├── data/                          # LFW subsets
│   ├── raw_3_images/              # 900 identities × 3 images
│   ├── raw_50_images/             # 12 identities × 50 images
│   └── raw_name/                  # 11 personal face directories
│
├── models/                        # Trained model weights
│   ├── model_10_images/           # Models trained on 10 images/identity
│   ├── model_20_images/           # Models trained on 20 images/identity
│   ├── model_30_images/           # Models trained on 30 images/identity
│   └── model_50_images/           # Production MLP model (50 images/identity)
│
├── logs/                          # Training logs with final reports
│   ├── log_3_128/                 # FaceNet128 × 3 images/identity
│   ├── log_3_512/                 # FaceNet512 × 3 images/identity
│   ├── log_10_128/                # FaceNet128 × 10 images/identity
│   ├── log_10_512/                # FaceNet512 × 10 images/identity
│   ├── log_20_128/                # FaceNet128 × 20 images/identity
│   ├── log_20_512/                # FaceNet512 × 20 images/identity
│   ├── log_30_128/                # FaceNet128 × 30 images/identity
│   └── log_30_512/                # FaceNet512 × 30 images/identity
│
├── results/                       # Final outputs
│   ├── images/                    # Result figures + demo screenshot
│   └── pp_deep_learning.pptx      # Presentation slides
│
├── documents/                     # Project documentation & diagrams
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Limitations & Future Work

### Limitations
- Accuracy drops under low lighting and extreme pose angles
- Not optimized for low-performance devices
- Single-camera (webcam) only; no IP camera support yet

### Future Work
- IP camera / IoT integration
- Web deployment (Flask / FastAPI)
- ArcFace for improved accuracy
- Database storage (MySQL, MongoDB)
- Cloud-based recognition APIs

---

## License

This project is for educational and research purposes.
