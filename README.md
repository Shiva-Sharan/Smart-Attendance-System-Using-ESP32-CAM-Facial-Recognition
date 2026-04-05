# Smart Attendance System Using ESP32-CAM & Facial Recognition

A robust, IoT-based Smart Attendance System that utilizes an ESP32-CAM module for real-time image capture and advanced deep learning models for accurate facial recognition. The system automates attendance tracking, stores records in a database, and provides a web interface for easy management and monitoring.

## 🌟 Key Features

* **Real-time Image Capture:** Uses the ESP32-CAM module to securely stream and capture facial data over a local hotspot.
* **High-Accuracy Face Detection & Recognition:** Powered by **YOLO** for robust face detection and **ArcFace** for precise facial feature extraction and matching.
* **Automated Logging:** Recognizes registered faces and automatically updates the SQLite database (`attendance.db`).
* **Hardware Integration:** Communicates with an LCD screen to display real-time attendance status and feedback.
* **Interactive Web Interface:** A dedicated web dashboard to view attendance records, manage users, and monitor the system.

---

## 🛠️ Technology Stack & Hardware

### Hardware
* **ESP32-CAM Module** (with FTDI programmer for initial upload)
* **LCD Display** (for real-time status output)
* Power supply and connecting jumper wires

### Software & AI Models
* **Backend:** Python
* **Web Interface:** Python (Flask/FastAPI)
* **Face Detection:** YOLO
* **Face Recognition:** ArcFace
* **Database:** SQLite
* **Microcontroller IDE:** Arduino IDE (for compiling `.ino` files)

---

## 📂 Project Structure

```text
├── .vscode/               # VS Code specific settings
├── Faces/                 # Directory for storing registered user face datasets
├── models/                # Directory containing YOLO and ArcFace model weights
├── static/                # Static assets for the web interface (CSS, JS)
├── templates/             # HTML templates for the web interface
├── attendance.db          # SQLite database storing attendance logs
├── cam_code.ino           # ESP32-CAM camera streaming and capture logic
├── hotspot.ino            # ESP32 network/hotspot configuration
├── Lcd_code.ino           # Code for driving the LCD display module
├── python_code.py         # Main Python backend script (AI models & processing)
├── web.py                 # Web server script for the dashboard
├── requirements.txt       # Python dependencies
└── .gitignore             # Ignored files (virtual environments, cache, models)
```
Installation & Setup
1. Hardware Setup (ESP32-CAM & LCD)
Install the Arduino IDE.

Add the ESP32 board manager URL to your Arduino IDE preferences and install the ESP32 boards package.

Open cam_code.ino, hotspot.ino, and Lcd_code.ino in the Arduino IDE.

Configure your WiFi/Hotspot credentials within the .ino files as needed.

Connect your ESP32-CAM to your computer using an FTDI programmer and upload the code.

Connect the LCD to the corresponding pins defined in Lcd_code.ino.

2. Software Setup (Python Backend)
It is recommended to use a virtual environment to avoid dependency conflicts.

Clone the repository:
git clone https://github.com/Shiva-Sharan/Smart-Attendance-System-Using-ESP32-CAM-Facial-Recognition.git
cd Smart-Attendance-System-Using-ESP32-CAM-Facial-Recognition

Create and activate a virtual environment:

Windows:
python -m venv .venv
.venv\Scripts\activate

Install the required packages:
pip install -r requirements.txt

Add Models:

Ensure your pre-trained YOLO and ArcFace model files (.onnx or .pkl) are placed inside the models/ directory.

🚀 Usage
1. Registering Faces
Add images of the individuals you want to recognize into the Faces/ directory. Ensure the images are clear and named according to the person's identity (e.g., john_doe.jpg). The system will process these to generate ArcFace embeddings.

2. Running the System
You need to start both the hardware module and the Python backend.

Power on the ESP32-CAM so it begins broadcasting/connecting to the network.

Run the main processing script to start facial recognition:
python python_code.py

To view the dashboard and logs, open a new terminal (with the virtual environment activated) and run the web server:
python web.py

3. Verification
Once running, step in front of the ESP32-CAM. The YOLO model will detect the face, ArcFace will process the identity, the LCD will display a success message, and the record will instantly appear on your web.py dashboard.

🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute.

📝 License
This project is open-source and available under the MIT License.
```

Would you like me to walk you through the browser upload process on GitHub once you have
