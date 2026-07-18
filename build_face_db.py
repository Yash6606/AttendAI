import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis

print("[INFO] Building ArcFace embeddings...")

FACE_DIR = "dataset"

if not os.path.exists(FACE_DIR):
    os.makedirs(FACE_DIR, exist_ok=True)
    raise FileNotFoundError("[ERROR] 'dataset' folder was missing and has been created automatically. Please register a student first.")

# Load ArcFace
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU

FACE_DB = {}

for roll in os.listdir(FACE_DIR):
    person_path = os.path.join(FACE_DIR, roll)
    if not os.path.isdir(person_path):
        continue

    embeddings = []

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        faces = face_app.get(img)

        # Use only images with exactly ONE face
        if len(faces) != 1:
            continue

        embeddings.append(faces[0].embedding)

    if embeddings:
        # 🔑 SAVE WITH STRING KEY (IMPORTANT)
        FACE_DB[str(roll)] = np.mean(embeddings, axis=0)
        print(f"[OK] Roll {roll} embeddings saved")

if not FACE_DB:
    raise RuntimeError("[ERROR] No valid face embeddings found")

# Save embeddings
np.savez("face_db.npz", **FACE_DB)
print("[SUCCESS] face_db.npz created successfully")

