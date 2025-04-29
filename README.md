# ♻️ Deep Learning-Based Waste Classification System

A real-time waste classification system leveraging deep learning for efficient and accurate categorization of waste materials. This project uses a YOLO-based object detection model integrated with a Flask backend, a Streamlit UI, and OpenCV for video processing. The goal is to promote better waste management through automation.

---

## 🚀 Project Objectives

- Develop a deep learning-based system to classify waste (e.g., plastic, glass, metal, paper).
- Enable real-time video processing and object detection.
- Provide an intuitive and interactive web interface for both image and webcam input.
- Facilitate easy integration and frontend-backend communication using Flask and Streamlit.

---

## 🛠️ Technology Stack

- **Backend:** Flask
- **Frontend/UI:** HTML, CSS
- **Object Detection:** YOLO (You Only Look Once)
- **Video Processing:** OpenCV
- **Integration:** flask_jsglue

---


---

## ⚙️ Configuration

- All system parameters are defined in `settings.py`:
  - Model paths
  - Detection thresholds
  - Database locations (if used)
- Flask routes handle:
  - Image uploads
  - Real-time video streaming

---

## 🧠 AI Model Integration

- Load the YOLO model via `helper.py`.
- Process images or video frames for classification.
- Detected objects are mapped to waste categories:
  - Plastic
  - Metal
  - Glass
  - Paper
  - Others

---

## 🎥 Real-Time Video Processing

- Capture frames using OpenCV.
- Feed frames into the YOLO model.
- Display bounding boxes with classification labels.
- Stream output to the frontend for real-time visualization.

---

## 🎨 User Interface

- Built with **Streamlit**.
- Two primary modes:
  1. **Upload Image** – Classify and visualize uploaded waste images.
  2. **Webcam Mode** – Real-time classification through webcam feed.
- Dashboard displays:
  - Detected waste types
  - Confidence scores
  - Visual overlays

---

## ⚡ Optimization & UX

- Streamlit caching for improved performance.
- Asynchronous Flask handling for real-time tasks.
- Minimal, responsive design for better user experience.

---

## ✅ Testing & Deployment

- `test.py` handles unit testing and debugging.
- Documentation includes:
  - Setup instructions
  - API endpoints (if any)
  - Common troubleshooting steps
- Deploy prototype and iterate based on feedback.

---

## 📦 Setup Instructions

```bash
# Clone the repo
git clone https://github.com/krishnakumar51/waste-classification.git
cd waste-classification

# Install dependencies
pip install -r requirements.txt

# Run Flask backend
python application.py

