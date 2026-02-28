# app.py
import os
import io
import sys
import base64
import time
import threading
import subprocess
import shutil
import traceback
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, abort
import cv2
import numpy as np
from PIL import Image
import pickle
import pandas as pd
from deepface import DeepFace  # Using DeepFace
from numpy.linalg import norm # For normalization

# ---------- Config ----------
IMG_SIZE = 160 # We still resize to 160x160 for consistency
EMBED_DIM = 128 # FaceNet model from DeepFace uses 128 dimensions
TRAIN_SCRIPT = "train_model.py"
EXTRACT_SCRIPT = "extract_embeddings.py"
EMBED_PATH = "embeddings/embeddings.npz"
DATA_TRAIN = "data/train"
FACES_DIR = "data/faces"
ATTENDANCE_CSV = "attendance.csv"
HAAR_XML = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
RECOGNITION_THRESHOLD = 0.6 # This is a good cosine similarity threshold
# ----------------------------

# Initialize DeepFace model once globally
# This will build the model on first run, which may take a minute.
print("Building FaceNet model via DeepFace...")
try:
    DeepFace.build_model("Facenet")
    print("Model built successfully.")
except Exception as e:
    print(f"Failed to build model: {e}")

os.makedirs(DATA_TRAIN, exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs("embeddings", exist_ok=True)
os.makedirs("model", exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")

face_cascade = cv2.CascadeClassifier(HAAR_XML)
if face_cascade.empty():
    print("Error: Failed to load Haar Cascade Classifier from", HAAR_XML)
    face_cascade = None

# training background control
train_lock = threading.Lock()
training_thread = None
train_status = {"running": False, "last_result": None, "message": ""}

# in-memory embeddings
known_embeddings = None
known_labels = None

def reload_embeddings():
    """Load embeddings and their matching labels from .npz"""
    global known_embeddings, known_labels
    known_embeddings = np.zeros((0, EMBED_DIM), dtype=np.float32)
    known_labels = []

    if not os.path.exists(EMBED_PATH):
        print("No embeddings file found:", EMBED_PATH)
        return

    try:
        embeddings_data = np.load(EMBED_PATH)
        
        if "embeddings" not in embeddings_data or "labels" not in embeddings_data:
            print(f" Warning: {EMBED_PATH} is missing 'embeddings' or 'labels' key.")
            return

        known_embeddings = embeddings_data["embeddings"]
        known_labels = embeddings_data["labels"].tolist()

        if len(known_embeddings) != len(known_labels):
            print(f" FATAL: Mismatch! {len(known_embeddings)} embeddings and {len(known_labels)} labels.")
            known_embeddings = np.zeros((0, EMBED_DIM), dtype=np.float32)
            known_labels = []
            return
            
        # Normalize embeddings on load
        known_embeddings = known_embeddings / norm(known_embeddings, axis=1, keepdims=True)

        unique_users = sorted(list(set(known_labels)))
        print(f" Reloaded {len(known_embeddings)} embeddings for {len(unique_users)} users: {', '.join(unique_users)}")
        
    except Exception as e:
        print(" Failed to load embeddings or labels:", e)
        traceback.print_exc()
        known_embeddings = np.zeros((0, EMBED_DIM), dtype=np.float32)
        known_labels = []

reload_embeddings()


# ---------- Utilities ----------
def decode_base64_image(data_url):
    if "," in data_url:
        _, data = data_url.split(",", 1)
    else:
        data = data_url
    b = base64.b64decode(data)
    img = Image.open(io.BytesIO(b)).convert("RGB")
    # Convert RGB (PIL) to BGR (OpenCV) for DeepFace
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def detect_faces_bgr(image_bgr):
    global face_cascade
    if face_cascade is None: return []
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
    print(f"Detected {len(faces)} faces in image.")
    return faces

# Updated: DeepFace embedding computation
def compute_embedding_from_crop(face_bgr):
    try:
        # DeepFace expects BGR image, not normalized
        embedding_obj = DeepFace.represent(
            img_path=face_bgr,
            model_name="Facenet",
            enforce_detection=False,
            detector_backend="skip"
        )
        
        # Extract the vector
        emb = np.array(embedding_obj[0]["embedding"])
        
        # Normalize the embedding
        emb = emb / norm(emb)
        return emb
        
    except Exception as e:
        print(f" Failed to compute embedding: {e}")
        return None

def match_embedding(emb):
    global known_embeddings, known_labels
    if known_embeddings is None or len(known_labels) == 0:
        return None, 0.0
    
    emb = emb.reshape(1, -1)
    try:
        # Cosine similarity (dot product of normalized vectors)
        sims = np.dot(known_embeddings, emb.T).squeeze()
    except Exception as e:
        print("Embedding dimension mismatch in match_embedding:", e)
        return None, 0.0
        
    best_idx = int(np.argmax(sims))
    
    if best_idx >= len(known_labels):
        print(f" Error: best_idx {best_idx} out of range for labels len {len(known_labels)}")
        return None, 0.0
        
    return known_labels[best_idx], float(sims[best_idx])


# ---------- Training ----------
def run_training_blocking():
    """Run extract_embeddings.py synchronously."""
    python_exec = sys.executable
    
    cmd_extract = [python_exec, EXTRACT_SCRIPT, "--dataset", DATA_TRAIN, "--out", EMBED_PATH]
    
    try:
        # Use utf-8 encoding for safety
        p2 = subprocess.run(cmd_extract, capture_output=True, text=True, encoding='utf-8')
    except Exception as e:
        return False, f"Failed to start embedding extraction: {e}"
        
    if p2.returncode != 0:
        return False, f"Extraction failed: {p2.stderr}\n{p2.stdout}"
        
    reload_embeddings()
    return True, f"Embedding extraction completed.\n{p2.stdout}"


def retrain_background():
    global train_status
    with train_lock:
        train_status["running"] = True
        train_status["message"] = "Training started"
    
    ok, msg = run_training_blocking()
    
    with train_lock:
        train_status["running"] = False
        train_status["last_result"] = ok
        train_status["message"] = str(msg)


# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register")
def register_page():
    return render_template("register.html")

@app.route("/attendance")
def attendance_page():
    return render_template("attendance.html")

@app.route("/users")
def users_page():
    users = []
    if os.path.exists(DATA_TRAIN):
        for name in sorted(os.listdir(DATA_TRAIN)):
            p = os.path.join(DATA_TRAIN, name)
            if os.path.isdir(p):
                count = len([f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))])
                users.append({"name": name, "count": count})
    return render_template("users.html", users=users)

@app.route("/manage_users")
def manage_users_page():
    users = []
    if os.path.exists(DATA_TRAIN):
        for name in sorted(os.listdir(DATA_TRAIN)):
            p = os.path.join(DATA_TRAIN, name)
            if os.path.isdir(p):
                count = len([f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))])
                users.append({"name": name, "count": count})
    return render_template("manage_users.html", users=users)

@app.route("/attendance_log")
def attendance_log():
    if os.path.exists(ATTENDANCE_CSV):
        try:
            df = pd.read_csv(ATTENDANCE_CSV)
            records = df.to_dict(orient="records")
        except pd.errors.EmptyDataError:
            records = []
    else:
        records = []
    return render_template("logs.html", records=records)


# ---------- APIs ----------
@app.route("/api/register", methods=["POST"])
def api_register():
    try:
        payload = request.get_json(force=True)
        name = payload.get("name", "").strip()
        images = payload.get("images", [])
        
        if name == "" or len(images) == 0:
            return jsonify({"status": "error", "message": "provide name and images"}), 400
            
        dest_dir = os.path.join(DATA_TRAIN, name)
        os.makedirs(dest_dir, exist_ok=True)
        raw_dir = os.path.join(FACES_DIR, name)
        os.makedirs(raw_dir, exist_ok=True)
        
        saved = 0
        for idx, data_url in enumerate(images):
            try:
                # img_np is now BGR
                img_np = decode_base64_image(data_url)
            except Exception as e:
                print("Failed decode image:", e)
                continue
                
            boxes = detect_faces_bgr(img_np)
            
            if len(boxes) == 0:
                crop = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE))
            else:
                x, y, w, h = boxes[0]
                crop = img_np[y:y+h, x:x+w]
                if crop.size == 0:
                    crop = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE))
                else:
                    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            
            fname = os.path.join(dest_dir, f"{int(time.time()*1000)}_{idx}.jpg")
            # Save the BGR image directly with cv2
            cv2.imwrite(fname, crop)
            cv2.imwrite(os.path.join(raw_dir, f"{int(time.time()*1000)}_{idx}.jpg"), crop)
            saved += 1
            
        return jsonify({"status": "ok", "name": name, "saved_images": saved})
        
    except Exception as e:
        tb = traceback.format_exc()
        print("Exception in /api/register:", tb)
        return jsonify({"status": "error", "message": "internal server error", "detail": str(e)}), 500

@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    global training_thread, train_status
    if train_status.get("running", False):
        return jsonify({"status": "error", "message": "training already running"}), 409
        
    training_thread = threading.Thread(target=retrain_background, daemon=True)
    training_thread.start()
    return jsonify({"status": "ok", "message": "training started in background"})

@app.route("/api/train_status")
def api_train_status():
    return jsonify(train_status)

@app.route("/api/identify", methods=["POST"])
def api_identify():
    try:
        payload = request.get_json(force=True)
        data_url = payload.get("image", "")
        if not data_url:
            return jsonify({"status": "error", "message": "no image"}), 400
        
        try:
            # img_np is BGR
            img_np = decode_base64_image(data_url)
        except Exception as e:
            return jsonify({"status": "error", "message": "invalid image"}), 400
            
        boxes = detect_faces_bgr(img_np)
        if len(boxes) == 0:
            return jsonify({"status": "error", "message": "no face detected"}), 400
            
        x, y, w, h = boxes[0]
        crop = img_np[y:y+h, x:x+w]
        if crop.size == 0:
            crop = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE))
        else:
            crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            
        emb = compute_embedding_from_crop(crop)
        if emb is None:
            return jsonify({"status": "error", "message": "could not compute embedding"})
            
        label, score = match_embedding(emb)
        
        if label is None or score < RECOGNITION_THRESHOLD:
            return jsonify({"status": "unknown", "score": float(score)})
            
        if not os.path.exists(ATTENDANCE_CSV):
            df = pd.DataFrame(columns=["Name", "Time", "Status"])
            df.to_csv(ATTENDANCE_CSV, index=False)
            
        try:
            df = pd.read_csv(ATTENDANCE_CSV)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=["Name", "Time", "Status"])

        today = datetime.now().strftime("%Y-%m-%d")
        
        if not ((df["Name"] == label) & (df["Time"].str.contains(today))).any():
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_record = pd.DataFrame([{"Name": label, "Time": now, "Status": "Present"}])
            df = pd.concat([df, new_record], ignore_index=True)
            df.to_csv(ATTENDANCE_CSV, index=False)
            
        return jsonify({"status": "ok", "name": label, "score": float(score)})
        
    except Exception as e:
        tb = traceback.format_exc()
        print("Exception in /api/identify:", tb)
        return jsonify({"status": "error", "message": "internal server error", "detail": str(e)}), 500

@app.route("/api/delete_user", methods=["POST"])
def api_delete_user():
    try:
        payload = request.get_json(force=True)
        name = payload.get("name", "").strip()
        if name == "":
            return jsonify({"status": "error", "message": "name required"}), 400
            
        p_train = os.path.join(DATA_TRAIN, name)
        p_faces = os.path.join(FACES_DIR, name)
        
        if not os.path.exists(p_train):
            return jsonify({"status": "error", "message": "user not found"}), 404
            
        try:
            shutil.rmtree(p_train)
            if os.path.exists(p_faces):
                shutil.rmtree(p_faces)
        except Exception as e:
            return jsonify({"status": "error", "message": f"failed to delete: {e}"}), 500
            
        return jsonify({"status": "ok", "message": f"user {name} deleted (manual retrain required)"})
        
    except Exception as e:
        tb = traceback.format_exc()
        print("Exception in /api/delete_user:", tb)
        return jsonify({"status": "error", "message": "internal server error", "detail": str(e)}), 500

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory("static", filename)


# ---------- Error Handlers ----------
@app.errorhandler(404)
def handle_404(err):
    path = request.path or ""
    if path.startswith("/api/"):
        return jsonify({"status": "error", "message": "not found", "path": path}), 404
    return render_template("404.html") if os.path.exists(os.path.join(app.template_folder or "", "404.html")) else ("Not Found", 404)

@app.errorhandler(500)
def handle_500(err):
    path = request.path or ""
    print("Internal server error:", getattr(err, "original_exception", err))
    if path.startswith("/api/"):
        return jsonify({"status": "error", "message": "internal server error"}), 500
    return ("Internal Server Error", 500)

if __name__ == "__main__":
    reload_embeddings()
    app.run(host="0.0.0.0", port=5000, debug=True)