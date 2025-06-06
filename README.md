# Self-Driving-Car-Simulation
This project simulates a self-driving car using computer vision and real-time object detection. It uses YOLOv8 (via Ultralytics), OpenCV, and pyttsx3 to analyze video input, detect obstacles like vehicles and pedestrians, make decisions (like slowing down or stopping), and even provide voice feedback.


🔍 Features
🧠 YOLOv8 Object Detection: Identifies cars, buses, trucks, people, stop signs, and traffic lights.

🎥 Video Frame Processing: Analyzes each frame in real time from a provided video.

🚦 Dynamic Decision Making: Stops, slows down, or steers based on object proximity.

🔊 Text-to-Speech (TTS): Announces driving decisions and collision warnings.

📊 Visual HUD Overlay: Displays speed, steering, decisions, and alerts on video frames.

💬 Threaded Voice Feedback: Runs TTS without freezing the main loop.

🛠️ Technologies Used
Ultralytics YOLOv8

OpenCV (cv2)

pyttsx3 (TTS)

Python 3.x

NumPy

🎞️ Demo
A pre-recorded driving video is processed frame-by-frame, and the system makes live decisions like a basic autonomous vehicle would.

🚀 How to Run
Clone the repository

Install dependencies

bash
Copy
Edit
pip install ultralytics opencv-python pyttsx3 numpy
Download or record a driving video

Update video_path in the script with your video location

Run the script

bash
Copy
Edit
python your_script.py
⚠️ Notes
The system does not control a real car — it’s for simulation/demo purposes only.

Make sure your machine supports video playback and TTS output.

📄 License
This project is open-source under the MIT License.
