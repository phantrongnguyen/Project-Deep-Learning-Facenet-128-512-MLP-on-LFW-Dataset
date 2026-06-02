import os
import numpy as np
import pickle
from collections import defaultdict
from deepface import DeepFace

FACES_DB_DIR = "train_models/examples/data"
MODEL_DIR = "train_models/examples/models"
DETECTOR_BACKEND = "mtcnn"
MODEL_NAME = "Facenet512"


def get_embedding(img_path):
    result = DeepFace.represent(
        img_path=img_path,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=False,
    )
    return result[0]["embedding"]


def load_data():
    embeddings_by_person = defaultdict(list)

    for person_name in os.listdir(FACES_DB_DIR):
        person_dir = os.path.join(FACES_DB_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        for file in os.listdir(person_dir):
            if not file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            path = os.path.join(person_dir, file)
            try:
                emb = get_embedding(path)
                embeddings_by_person[person_name].append(emb)
                print(f"  Extracted: {person_name}/{file}")
            except Exception as e:
                print(f"  Failed: {person_name}/{file} - {e}")

    return embeddings_by_person


def train_and_save():
    print(f"Detector backend: {DETECTOR_BACKEND}")
    print(f"Model: {MODEL_NAME}")
    print("Loading data...")

    embeddings_by_person = load_data()

    # Compute average embedding (centroid) for each person
    centroids = {}
    for person_name, embs in embeddings_by_person.items():
        centroids[person_name] = np.mean(embs, axis=0)
        print(f"  {person_name}: {len(embs)} images -> centroid shape {centroids[person_name].shape}")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, f"Facenet512_mtcnn_001.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(centroids, f)

    print("Saved:", save_path)


if __name__ == "__main__":
    train_and_save()
