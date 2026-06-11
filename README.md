# 👤 Face Attendance System (AI-Based)

A simple real-time face recognition system that automatically marks attendance using your webcam.  
Built with Python and InsightFace, this project uses deep learning face embeddings instead of traditional training models.

---

## 🚀 What this project does

- Detects faces in real-time using webcam
- Recognizes known people instantly
- Automatically records attendance with timestamp
- Uses a simple folder of images as the database (no training required)

---

## 🛠 Tech Used

- Python  
- OpenCV  
- InsightFace (deep learning face recognition)  
- NumPy  
- Pandas  

---

## 📁 How the project is organized

Faces/            → Add photos of people here  
recognize.py      → Main program (run this file)  
attendance.csv    → Auto-generated attendance log  

---

## ▶️ How to run it

### 1. Install required libraries
pip install opencv-python numpy pandas insightface onnxruntime

### 2. Add face images
Put clear face images inside the `Faces/` folder.

Example:
Faces/shivani.jpg → Shivani  
Faces/john.jpg → John  

### 3. Start the system
python recognize.py

---

## 🎯 How to use

- Show your face to the webcam  
- If recognized → your name appears  
- Attendance is automatically saved  
- Press **Q** to exit  

---

## 📌 Important note

This project uses face embeddings (InsightFace) instead of training a model, which makes it:
- faster  
- simpler  
- more accurate for small datasets  

---

## 💡 Why this project is useful

It demonstrates:
- real-time computer vision  
- AI-based face recognition  
- automation of attendance systems  
- practical use of deep learning embeddings  
