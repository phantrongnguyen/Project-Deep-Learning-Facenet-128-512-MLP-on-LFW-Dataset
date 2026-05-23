from deepface import DeepFace

def get_embedding(img_path, model_name):
    result = DeepFace.represent(
        img_path=img_path,
        model_name=model_name,
        enforce_detection=False
    )
    return result[0]["embedding"]