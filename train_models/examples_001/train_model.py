import os
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer

from embedding_extractor import get_embedding

FACES_DB_DIR = "train_models/examples_001/data"
MODEL_DIR = "train_models/examples_001/models"


def load_data(model_name):
    X, y = [], []

    for person_name in os.listdir(FACES_DB_DIR):
        person_dir = os.path.join(FACES_DB_DIR, person_name)

        if not os.path.isdir(person_dir):
            continue

        for file in os.listdir(person_dir):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                path = os.path.join(person_dir, file)

                try:
                    emb = get_embedding(path, model_name)
                    X.append(emb)
                    y.append(person_name)
                except:
                    continue

    return np.array(X), np.array(y)


def train_and_save(model_name):
    print(f"Training model: {model_name}")

    X, y = load_data(model_name)

    # ===== Normalize (QUAN TRỌNG) =====
    norm = Normalizer()
    X = norm.fit_transform(X)

    # ===== Train SVM =====
    model = SVC(kernel='linear', probability=True)
    model.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ===== LƯU CẢ model + norm =====
    save_path = f"{MODEL_DIR}/{model_name}_svm_001.pkl"
    with open(save_path, "wb") as f:
        pickle.dump((model, norm), f)   #FIX

    print("Saved:", save_path)


if __name__ == "__main__":
    train_and_save("Facenet")
    train_and_save("Facenet512")