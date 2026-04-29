The system uses computer vision and deep learning to identify vehicles in real time, re-identify them between entry and exit cameras, and determine whether they exceed the speed limit.

⚙️ Features
🚗 Real-time vehicle detection using YOLOv8
🎯 Multi-object tracking using SORT algorithm
🔍 Vehicle Re-Identification (Re-ID) across cameras
📏 Speed calculation based on travel time and distance
🚨 Automatic violation detection
📩 Instant Telegram alerts with vehicle image
🧠 Smart filtering (removes duplicate/false detections)
🎥 Saves processed output videos
🛠️ Technologies Used
Python
OpenCV
PyTorch
YOLOv8 (Ultralytics)
NumPy
SQLite Database
Telegram Bot API
📂 Project Structure
main.py → Main system script
sort.py → Tracking algorithm
db_utils.py → Database functions
reid_utils.py → Vehicle embedding & Re-ID
owners.db → Vehicle owners database
violations/ → Saved violation images
Final_Output/ → Output processed videos
▶️ How to Run
Install dependencies:
pip install opencv-python torch ultralytics numpy requests
Make sure required files exist:
sort.py
db_utils.py
reid_utils.py
Run the system:
python main.py
Enter:
Entry video path
Exit video path
Distance between cameras
Speed limit
Time delay (if any)
📊 How It Works
Vehicles are detected using YOLOv8
Each vehicle is tracked using SORT
A unique embedding is extracted for each vehicle
Vehicles are matched between entry and exit cameras

Speed is calculated using:

Speed = Distance / Time

If speed exceeds the limit:
Vehicle image is captured
Alert is sent via Telegram
🚨 Example Alert
Owner Name
Plate Number
Detected Speed
Location
Timestamp
Vehicle Image
⚠️ Notes
Requires a working Telegram Bot Token
GPU recommended for better performance
Ensure database is initialized before running
👨‍💻 Author

Developed by: Taha bashier
Field: Communication Engineering / AI Systems

⭐ Future Improvements
License plate recognition (OCR)
Real-time camera streaming
Web dashboard for monitoring
Cloud deployment
