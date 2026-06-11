import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
from insightface.app import FaceAnalysis

# ---------------- INIT MODEL ----------------
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))  # use -1 if CPU only

# ---------------- CONFIG ----------------
FACES_FOLDER = "Faces"
ATTENDANCE_FILE = "attendance.csv"
THRESHOLD = 0.45

# ---------------- LOAD KNOWN FACES ----------------
known_faces = []

for file in os.listdir(FACES_FOLDER):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):

        path = os.path.join(FACES_FOLDER, file)
        img = cv2.imread(path)

        if img is None:
            continue

        faces = app.get(img)
        if len(faces) == 0:
            print(f"Skipping {file} (no face detected)")
            continue

        if len(faces) == 0:
            print(f"No face found in {file}")
            continue

        name = os.path.splitext(file)[0]
        embedding = faces[0].embedding

        known_faces.append({
            "name": name,
            "embedding": embedding
        })

print(f"Loaded {len(known_faces)} faces ✔")

# ---------------- COSINE SIMILARITY ----------------
def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

# ---------------- ATTENDANCE FUNCTION ----------------
def mark_attendance(name):
    if name == "Unknown":
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=["Name", "Time"])
        df.to_csv(ATTENDANCE_FILE, index=False)

    df = pd.read_csv(ATTENDANCE_FILE)

    if name in df["Name"].values:
        return

    new_row = pd.DataFrame([[name, now]], columns=["Name", "Time"])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(ATTENDANCE_FILE, index=False)

    print(f"[✓] Attendance marked: {name}")

# ---------------- WEBCAM ----------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:

        best_name = "Unknown"
        best_score = -1

        for person in known_faces:
            score = cosine_similarity(face.embedding, person["embedding"])

            if score > best_score:
                best_score = score
                best_name = person["name"]

        if best_score > THRESHOLD:
            name = best_name
        else:
            name = "Unknown"

        mark_attendance(name)

        x1, y1, x2, y2 = map(int, face.bbox)

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{name} {best_score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow("Face Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()