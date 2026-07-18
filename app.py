from flask import Flask, render_template, Response, send_file, request, jsonify
import cv2
import base64
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime
import time
import os
import sys
from insightface.app import FaceAnalysis
from numpy.linalg import norm

app = Flask(__name__)

# =========================
# STREAM CONTROL FLAG
# =========================
streaming = False

# =========================
# AUTO BUILD FACE DATABASE
# =========================
if not os.path.exists("face_db.npz"):
    os.system(f"{sys.executable} build_face_db.py")

# =========================
# LOAD FACE DATABASE
# =========================
data = np.load("face_db.npz")
FACE_DB = {int(k): data[k] for k in data.files}

# =========================
# LOAD MODEL
# =========================
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=-1, det_size=(640, 640))

# =========================
# LOAD STUDENTS
# =========================
students_df = pd.read_csv("students.csv")
students_df["roll_no"] = students_df["roll_no"].astype(int)

active_date = datetime.now().strftime("%Y-%m-%d")

def get_attendance_filepath():
    global active_date
    return f"attendance_{active_date}.csv"

def load_or_create_daily_attendance():
    filepath = get_attendance_filepath()
    if not os.path.exists(filepath):
        df = students_df.copy()
        df["status"] = "Absent"
        df["time"] = ""
        df.to_csv(filepath, index=False)
        return df
    else:
        df = pd.read_csv(filepath)
        df["roll_no"] = df["roll_no"].astype(int)
        
        # Sync with students_df in case new students were registered mid-day
        missing_rolls = students_df[~students_df["roll_no"].isin(df["roll_no"])]
        if not missing_rolls.empty:
            new_rows = missing_rolls.copy()
            new_rows["status"] = "Absent"
            new_rows["time"] = ""
            df = pd.concat([df, new_rows], ignore_index=True)
            df.to_csv(filepath, index=False)
            
        return df


# =========================
# UTILS
# =========================
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# =========================
# VIDEO STREAM
# =========================
def gen_frames():
    global streaming
    cam = None
    tracked_faces = []
    frame_count = 0

    while True:
        if not streaming:
            if cam:
                cam.release()
                cam = None
            time.sleep(0.1)
            continue

        if cam is None:
            cam = cv2.VideoCapture(0)

        success, frame = cam.read()
        if not success:
            break

        frame_count += 1
        current_time = time.time()

        if frame_count % 5 == 0:
            tracked_faces = []
            faces = face_app.get(frame)

            for face in faces:
                emb = face.embedding
                best_score, best_roll = 0, None

                for roll, db_emb in FACE_DB.items():
                    score = cosine_similarity(emb, db_emb)
                    if score > best_score:
                        best_score, best_roll = score, roll

                label = "Unknown"
                if best_score > 0.5:
                    name = students_df.loc[
                        students_df["roll_no"] == best_roll, "name"
                    ].values[0]

                    attendance_df = load_or_create_daily_attendance()
                    student_idx = attendance_df.index[attendance_df["roll_no"] == best_roll]
                    if not student_idx.empty:
                        idx = student_idx[0]
                        if attendance_df.loc[idx, "status"] != "Present":
                            attendance_df.loc[idx, "status"] = "Present"
                            attendance_df.loc[idx, "time"] = datetime.now().strftime("%H:%M:%S")
                            attendance_df.to_csv(get_attendance_filepath(), index=False)

                    label = f"{name} (Marked)"

                tracked_faces.append((face.bbox.astype(int), label, current_time))

        for bbox, label, seen in tracked_faces:
            if current_time - seen < 1:
                x1, y1, x2, y2 = bbox
                color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
                )

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

# =========================
# ROUTES
# =========================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/start_stream")
def start_stream():
    global streaming
    streaming = True
    return "started"

@app.route("/stop_stream")
def stop_stream():
    global streaming
    streaming = False
    return "stopped"

# 🔽 DOWNLOAD ATTENDANCE CSV
@app.route("/download_attendance")
def download_attendance():
    target_date = request.args.get("date")
    if target_date:
        try:
            datetime.strptime(target_date, "%Y-%m-%d")
            file_path = f"attendance_{target_date}.csv"
        except ValueError:
            return "Invalid date format", 400
    else:
        load_or_create_daily_attendance()
        file_path = get_attendance_filepath()

    if os.path.exists(file_path):
        return send_file(
            file_path,
            as_attachment=True,
            download_name=file_path
        )
    return "Attendance file not found", 404

# 🔽 REGISTRATION PAGES AND APIs
@app.route("/register")
def register():
    return render_template("register.html")

@app.route("/api/register/finalize", methods=["POST"])
def api_register_finalize():
    global FACE_DB, students_df
    try:
        req_data = request.json
        roll_no = int(req_data["roll_no"])
        name = req_data["name"].strip()
        images = req_data["images"]
        retrain = bool(req_data.get("retrain", True))

        # 1. Save all images
        folder_path = os.path.join("dataset", str(roll_no))
        os.makedirs(folder_path, exist_ok=True)

        for idx, img_b64 in enumerate(images):
            header, encoded = img_b64.split(",", 1)
            data = base64.b64decode(encoded)
            file_path = os.path.join(folder_path, f"image_{idx + 1}.jpg")
            with open(file_path, "wb") as f:
                f.write(data)

        # 2. Update students.csv
        students_path = "students.csv"
        df = pd.read_csv(students_path)
        df["roll_no"] = df["roll_no"].astype(int)

        if roll_no not in df["roll_no"].values:
            df.loc[len(df)] = [roll_no, name]
            df.to_csv(students_path, index=False)
            
            # Reload students_df
            students_df = pd.read_csv("students.csv")
            students_df["roll_no"] = students_df["roll_no"].astype(int)
            
            # Immediately sync new student into the active daily log file
            load_or_create_daily_attendance()

        # 3. Retrain if requested
        retrained = False
        if retrain:
            result = subprocess.run(
                [sys.executable, "build_face_db.py"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                return jsonify({"error": f"Retraining failed: {result.stderr}"}), 500

            # Reload the face database
            if os.path.exists("face_db.npz"):
                data = np.load("face_db.npz")
                FACE_DB = {int(k): data[k] for k in data.files}
            retrained = True

        return jsonify({"status": "success", "retrained": retrained})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# 🔽 ATTENDANCE HISTORY PAGES AND APIs
@app.route("/history")
def history():
    return render_template("history.html")

@app.route("/api/history")
def api_history():
    try:
        files = os.listdir(".")
        attendance_files = [f for f in files if f.startswith("attendance_") and f.endswith(".csv")]
        
        months = {}
        for filename in attendance_files:
            try:
                date_str = filename.replace("attendance_", "").replace(".csv", "")
                datetime.strptime(date_str, "%Y-%m-%d")
                month_key = date_str[:7]
                if month_key not in months:
                    months[month_key] = []
                months[month_key].append(date_str)
            except ValueError:
                continue
                
        return jsonify(months)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/history/detail/<date>")
def api_history_detail(date):
    try:
        datetime.strptime(date, "%Y-%m-%d")
        filepath = f"attendance_{date}.csv"
        
        if not os.path.exists(filepath):
            return jsonify([])
            
        df = pd.read_csv(filepath)
        df = df.fillna("")
        df["roll_no"] = df["roll_no"].astype(int)
        records = df.to_dict(orient="records")
        return jsonify(records)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/active_session", methods=["GET", "POST"])
def api_active_session():
    global active_date, students_df
    if request.method == "POST":
        try:
            req_data = request.json
            new_date = req_data["date"].strip()
            reset = bool(req_data.get("reset", False))

            # Validate date format YYYY-MM-DD
            datetime.strptime(new_date, "%Y-%m-%d")
            active_date = new_date

            filepath = get_attendance_filepath()
            if reset or not os.path.exists(filepath):
                # Clear and create fresh file
                df = students_df.copy()
                df["status"] = "Absent"
                df["time"] = ""
                df.to_csv(filepath, index=False)
            else:
                # Sync pre-existing file in case of new master list additions
                df = pd.read_csv(filepath)
                df["roll_no"] = df["roll_no"].astype(int)
                missing_rolls = students_df[~students_df["roll_no"].isin(df["roll_no"])]
                if not missing_rolls.empty:
                    new_rows = missing_rolls.copy()
                    new_rows["status"] = "Absent"
                    new_rows["time"] = ""
                    df = pd.concat([df, new_rows], ignore_index=True)
                    df.to_csv(filepath, index=False)

            return jsonify({"status": "success", "active_date": active_date})
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    else:
        # GET returns current active date
        return jsonify({"active_date": active_date})

@app.route("/api/history/delete/<date>", methods=["DELETE"])
def api_history_delete(date):
    try:
        datetime.strptime(date, "%Y-%m-%d")
        filepath = f"attendance_{date}.csv"
        
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({"status": "success"})
        else:
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
