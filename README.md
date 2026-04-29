# 🚔 Smart Traffic Speed Monitoring System

## 📌 Description

The system uses computer vision and deep learning to detect vehicles in real time, re-identify them between entry and exit cameras, and determine whether they exceed the speed limit.

---

## ⚙️ Features

* 🚗 Real-time vehicle detection using YOLOv8
* 🎯 Multi-object tracking using SORT algorithm
* 🔍 Vehicle Re-Identification (Re-ID) across cameras
* 📏 Speed calculation based on travel time and distance
* 🚨 Automatic violation detection
* 📩 Instant Telegram alerts with vehicle image
* 🧠 Smart filtering (removes duplicate/false detections)
* 🎥 Saves processed output videos

---

## 🛠️ Technologies Used

* Python
* OpenCV
* PyTorch
* YOLOv8 (Ultralytics)
* NumPy
* SQLite Database
* Telegram Bot API

---

## 📂 Project Structure

* `main.py` → Main system script
* `sort.py` → Tracking algorithm
* `db_utils.py` → Database functions
* `reid_utils.py` → Vehicle embedding & Re-ID
* `owners.db` → Vehicle owners database
* `violations/` → Saved violation images
* `Final_Output/` → Output processed videos

---

## ▶️ How to Run

### 1. Install dependencies:

```
pip install opencv-python torch ultralytics numpy requests
```

### 2. Make sure required files exist:

* sort.py
* db_utils.py
* reid_utils.py

### 3. Run the system:

```
python main.py
```

### 4. Enter:

* Entry video path
* Exit video path
* Distance between cameras
* Speed limit
* Time delay (if any)

---

## 📊 How It Works

1. Vehicles are detected using YOLOv8
2. Each vehicle is tracked using SORT
3. A unique embedding is extracted for each vehicle
4. Vehicles are matched between entry and exit cameras

### Speed Calculation:

```
Speed = Distance / Time
```

5. If speed exceeds the limit:

   * Vehicle image is captured
   * Alert is sent via Telegram

---

## 🚨 Example Alert

* Owner Name
* Plate Number
* Detected Speed
* Location
* Timestamp
* Vehicle Image

---

## ⚠️ Notes

* Requires a working Telegram Bot Token
* GPU recommended for better performance
* Ensure database is initialized before running

---

## 👨‍💻 Author

Developed by: **Taha bashier**
Field: Communication Engineering / AI Systems

---

## ⭐ Future Improvements

* License plate recognition (OCR)
* Real-time camera streaming
* Web dashboard for monitoring
* Cloud deployment

<img width="1408" height="768" alt="Gemini_Generated_Image_qjo9ilqjo9ilqjo9" src="https://github.com/user-attachments/assets/1d018945-2fda-45b6-810e-6fccd4fde6ab" />

