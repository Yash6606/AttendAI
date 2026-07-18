# AI Face Recognition Attendance System

A modern, contactless, real-time face recognition attendance system built with Flask, OpenCV, and InsightFace (using the ArcFace `buffalo_l` model). 

---

## Key Features

1. **Real-time Video Identification**:
   * Uses your webcam to detect multiple faces instantly.
   * Matches real-time face embeddings against the registered database using cosine similarity.
   * Displays dynamic bounding boxes (Green for recognized, Red for unknown) with status tags.
2. **Dynamic Daily Logs (Attendance Sheet)**:
   * Dynamically generates attendance logs named `attendance_YYYY-MM-DD.csv` for the active day.
   * Initializes logs with all students registered in the master list as `Absent`.
   * Automatically updates status to `Present` along with their entry timestamp upon face recognition.
   * Auto-syncs newly registered students into pre-existing daily sheets.
3. **Interactive Student Registration**:
   * Client-side guided session capturing **18 photos** at different face angles (straight, left, right, tilt up, tilt down, smile) with a 2-second interval.
   * Displays captured snaps instantly in a grid preview.
   * Option to automatically retrain the face database upon registration finalization.
4. **Attendance History Explorer**:
   * Month-by-month grouped calendar logs sidebar.
   * Live statistics overview showing Present, Absent, and Total registered count.
   * Full search, filters (All, Present, Absent), and deletion options for historical files.
   * Direct download buttons to export daily CSV sheets.

---

## File Structure

```text
AttendAI/
├── dataset/                  # Registration images folder (Gitignored, created automatically)
│   ├── 1/                    # Roll 1 photos (locally saved)
│   └── ...
├── static/                   # CSS and JS frontend assets
├── templates/                # HTML pages (Home, Registration, History, About)
├── app.py                    # Main Flask application and server APIs
├── build_face_db.py          # Script extracting face embeddings (creates dataset/ automatically if missing)
├── face_db.npz               # Compiled ArcFace face embeddings database (Gitignored)
├── students.csv              # Master list of registered students (roll_no, name)
├── requirements.txt          # Python dependency list
└── README.md                 # Project documentation
```

---

## Setup Instructions

### 1. Prerequisiets
Ensure you have **Python 3.10 to 3.12** installed on your system.

### 2. Navigate to Project Directory
Open your terminal (PowerShell or Command Prompt) and type:
```powershell
cd d:\projects\done\AttendAI
```

### 3. Activate the Virtual Environment
Activate the pre-existing virtual environment:
* **PowerShell**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
* **Command Prompt (CMD)**:
  ```cmd
  venv\Scripts\activate.bat
  ```

### 4. Install Dependencies
Install dependencies inside the virtual environment:
```powershell
pip install -r requirements.txt
```

---

## Running the Application

### 1. Compile embeddings (Optional)
If your `face_db.npz` model file is missing or you want to rebuild it from existing dataset folders, run:
```powershell
python build_face_db.py
```

### 2. Start the Server
Start the Flask development server:
```powershell
python app.py
```

### 3. Open Web UI
Open your web browser and navigate to:
**`http://127.0.0.1:5000/`**

---

## Technologies Used

* **Frontend**: HTML5, Vanilla CSS, Bootstrap 5, JavaScript (Webcam Navigator API)
* **Backend**: Flask (Python)
* **Computer Vision**: OpenCV, InsightFace (ArcFace embeddings), NumPy, Pandas
