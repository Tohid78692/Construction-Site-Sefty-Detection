from flask import Flask, render_template, request, Response, jsonify
import cv2
from ultralytics import YOLO
import os
import pyttsx3
import threading
import time

app = Flask(__name__)

# =========================
# FOLDERS
# =========================
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# =========================
# MODEL
# =========================
model = YOLO("runs/detect/train2/weights/best.pt")

# =========================
# STATS
# =========================
stats_data = {"Person": 0, "Violations": 0}

# =========================
# VOICE ALERT
# =========================
engine = pyttsx3.init()

def speak():
    engine.say("Warning! Safety violation detected")
    engine.runAndWait()

last_alert = 0
ALERT_INTERVAL = 5

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

def generate_frames():
    global last_alert, stats_data

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, conf=0.3)
        boxes = results[0].boxes

        counts = {}
        violation = False

        for box in boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            counts[label] = counts.get(label, 0) + 1

            color = (0, 255, 0)
            if label.startswith("NO-") or label == "Fall-Detected":
                color = (0, 0, 255)
                violation = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        stats_data["Person"] = counts.get("Person", 0)
        stats_data["Violations"] = sum(counts[k] for k in counts if k.startswith("NO-"))

        if violation and time.time() - last_alert > ALERT_INTERVAL:
            threading.Thread(target=speak, daemon=True).start()
            last_alert = time.time()

        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats")
def stats():
    return jsonify(stats_data)


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file key"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"})

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    results = model(path)

    result_path = os.path.join(RESULT_FOLDER, "result_" + file.filename)
    results[0].save(filename=result_path)

    counts = {}
    for box in results[0].boxes:
        label = model.names[int(box.cls[0])]
        counts[label] = counts.get(label, 0) + 1

    return jsonify({
        "image": "/static/results/" + os.path.basename(result_path),
        "detections": counts
    })


if __name__ == "__main__":
    app.run(debug=True)