from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import base64
import os
import uuid
import numpy as np
import cv2
import requests
import firebase_admin
from firebase_admin import credentials, firestore
import gc
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024

# ===================== CONFIG =====================
MODEL = 'SFace'
BACKEND = 'opencv'
TMP_DIR = 'temp_files'
MAX_IMAGE_SIZE = 512
os.makedirs(TMP_DIR, exist_ok=True)

FIRESTORE_COLLECTION = "users"

print("=" * 80)
print("SERVER STARTING - FINAL STABLE VERSION v7.0")
print("=" * 80)

# ===================== FIREBASE INIT =====================
db = None
firebase_status = "NOT_INITIALIZED"

def init_firebase():
    global db, firebase_status
    service_account_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")

    if service_account_json:
        try:
            cred_dict = json.loads(service_account_json)
            if "private_key" in cred_dict:
                cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            firebase_status = "SUCCESS"
            print("✅ Firebase Initialized via Environment Variable")
            return
        except Exception as e:
            print(f"❌ Firebase Env Error: {e}")

    # Local development fallback
    if os.path.exists("serviceAccountKey.json"):
        try:
            cred = credentials.Certificate("serviceAccountKey.json")
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            firebase_status = "SUCCESS (Local)"
            print("✅ Firebase Initialized via Local File")
            return
        except Exception as e:
            print(f"❌ Local File Error: {e}")

    firebase_status = "DISABLED"
    print("⚠️ Firebase Disabled - No credentials found")

init_firebase()

# ===================== HELPERS =====================
def tmp_path(prefix='t'):
    return os.path.join(TMP_DIR, f'{prefix}_{uuid.uuid4().hex[:8]}.jpg')

def cleanup(*paths):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except:
            pass
    gc.collect()

def save_b64(base64_str: str, path: str):
    try:
        img_data = base64.b64decode(base64_str)
        with open(path, 'wb') as f:
            f.write(img_data)
        return True
    except:
        return False

def download_image(url: str):
    try:
        if not url or not url.startswith("http"):
            return None
        r = requests.get(url, timeout=15, headers={"User-Agent": "Attendance-Face/1.0"})
        r.raise_for_status()
        arr = np.frombuffer(r.content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        path = tmp_path("reg")
        cv2.imwrite(path, img)
        return path
    except Exception as e:
        print(f"[Download Error] {str(e)[:80]}")
        return None

def laplacian_variance(path):
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        return float(cv2.Laplacian(img, cv2.CV_64F).var())
    except:
        return 0.0

# ===================== ROUTES =====================

@app.route('/face/register', methods=['POST'])
def register_face():
    """Sirf Firestore mein photoURL save karta hai"""
    try:
        data = request.json or {}
        uid = (data.get('userId') or '').strip()
        photo_url = data.get('photoURL', '')

        if not uid or not photo_url:
            return jsonify({"success": False, "message": "userId and photoURL required"}), 400

        if db:
            db.collection(FIRESTORE_COLLECTION).document(uid).update({
                "photoURL": photo_url,
                "faceRegistered": True,
                "updatedAt": datetime.now()
            })

        return jsonify({"success": True, "message": "Face URL saved", "photoURL": photo_url})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/face/auto-verify', methods=['POST'])
def auto_verify():
    temp_paths = []
    try:
        data = request.json or {}
        uid = (data.get('userId') or '').strip()
        frames = data.get('framesBase64') or []
        photo_url = data.get('photoURL') or ''

        if not uid or len(frames) < 3:
            return jsonify({"success": False, "matched": False, "live": False, "msg": "Invalid request"}), 400

        # Download registered image (Cloudinary se)
        reg_path = download_image(photo_url)
        if not reg_path:
            return jsonify({"success": False, "matched": False, "live": False, "msg": "Registered face not found"}), 400
        temp_paths.append(reg_path)

        # Save frames
        frame_paths = []
        for i, b64 in enumerate(frames[:3]):
            p = tmp_path(f'f{i}')
            if save_b64(b64, p):
                frame_paths.append(p)
                temp_paths.append(p)

        if len(frame_paths) < 3:
            cleanup(*temp_paths)
            return jsonify({"success": False, "matched": False, "live": False, "msg": "Frame saving failed"}), 400

        # Face Detection
        for i, p in enumerate(frame_paths):
            faces = DeepFace.extract_faces(img_path=p, detector_backend=BACKEND, enforce_detection=False)
            valid = [f for f in faces if float(f.get('confidence', 0)) > 0.5]
            if len(valid) != 1:
                cleanup(*temp_paths)
                return jsonify({"success": False, "matched": False, "live": False, "msg": f"Frame {i+1}: Exactly 1 face required"}), 400

        # Liveness Check (Texture)
        lap_scores = [laplacian_variance(p) for p in frame_paths]
        avg_lap = sum(lap_scores) / len(lap_scores)
        if avg_lap < 40:
            cleanup(*temp_paths)
            return jsonify({"success": True, "matched": False, "live": False, "msg": "Flat image detected"}), 200

        # Face Matching
        best_frame = frame_paths[1]
        result = DeepFace.verify(
            img1_path=reg_path, img2_path=best_frame,
            model_name=MODEL, detector_backend=BACKEND, enforce_detection=False
        )

        matched = bool(result.get('verified', False))
        distance = float(result.get('distance', 0))
        threshold = float(result.get('threshold', 1))
        confidence = round(max(0, min(100, (1 - distance / threshold) * 100)), 1) if threshold > 0 else 0

        cleanup(*temp_paths)

        if not matched:
            return jsonify({
                "success": True, "matched": False, "live": True,
                "confidence": confidence, "msg": f"Face not matched ({confidence}%)"
            })

        return jsonify({
            "success": True, "matched": True, "live": True,
            "confidence": confidence, "msg": f"Verified successfully ({confidence}%)"
        })

    except Exception as e:
        print(f"[AUTO_VERIFY ERROR]: {e}")
        cleanup(*temp_paths)
        return jsonify({"success": False, "matched": False, "live": False, "msg": str(e)}), 500


@app.route('/')
def index():
    return jsonify({
        "status": "running",
        "version": "7.0-final",
        "firebase": firebase_status
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)