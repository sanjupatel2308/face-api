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
from io import BytesIO

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

print("="*80)
print("SERVER STARTING - v6.0 (Cleaned)")
print("="*80)

# ===================== FIREBASE =====================
db = None
firebase_status = "NOT_INITIALIZED"

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
        print("✅ Firebase Initialized Successfully")
    except Exception as e:
        firebase_status = f"ERROR: {e}"
        print(f"❌ Firebase Error: {e}")
else:
    print("❌ FIREBASE_SERVICE_ACCOUNT_JSON not found")

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

def resize_image(path):
    try:
        img = cv2.imread(path)
        if img is None:
            return path
        h, w = img.shape[:2]
        if max(h, w) <= MAX_IMAGE_SIZE:
            return path
        scale = MAX_IMAGE_SIZE / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        cv2.imwrite(path, cv2.resize(img, new_size))
        return path
    except:
        return path

def laplacian_variance(path):
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        return float(cv2.Laplacian(img, cv2.CV_64F).var())
    except:
        return 0.0

def get_registered_photo(uid: str):
    """Firebase se photoURL fetch karta hai"""
    if db is None:
        return None, "Firebase not initialized"
    try:
        doc = db.collection(FIRESTORE_COLLECTION).document(uid).get()
        if not doc.exists:
            return None, "User not found"

        data = doc.to_dict() or {}
        for field in ["photoURL", "photoUrl", "faceURL", "faceImageUrl"]:
            url = data.get(field)
            if url and isinstance(url, str) and url.startswith("http"):
                return url, data.get("name", "User")
        return None, "photoURL not found in user document"
    except Exception as e:
        return None, str(e)

def download_image(url: str):
    """Cloudinary/Firebase Storage se image download karta hai"""
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "AttendanceApp/1.0"})
        r.raise_for_status()
        arr = np.frombuffer(r.content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        path = tmp_path("reg")
        cv2.imwrite(path, img)
        return resize_image(path)
    except Exception as e:
        print(f"Download Error: {e}")
        return None

# ===================== MAIN ROUTE =====================
@app.route('/face/auto-verify', methods=['POST'])
def auto_verify():
    temp_paths = []
    try:
        data = request.json or {}
        uid = (data.get('userId') or '').strip()
        frames = data.get('framesBase64') or []
        photo_url = data.get('photoURL') or ''

        if not uid:
            return jsonify({"success": False, "matched": False, "live": False, "msg": "userId required"}), 400
        if len(frames) < 3:
            return jsonify({"success": False, "matched": False, "live": False, "msg": "Need 3 frames"}), 400

        # Step 1: Registered photo download karo
        reg_path = None
        if photo_url:
            reg_path = download_image(photo_url)
        else:
            # Fallback: Firebase se photoURL lo
            url, _ = get_registered_photo(uid)
            if url:
                reg_path = download_image(url)

        if not reg_path:
            return jsonify({"success": False, "matched": False, "live": False, "msg": "Registered face not found"}), 400

        temp_paths.append(reg_path)

        # Step 2: Frames save karo
        frame_paths = []
        for i, b64 in enumerate(frames[:3]):
            p = tmp_path(f'frame_{i}')
            if save_b64(b64, p):
                frame_paths.append(p)
                temp_paths.append(p)

        if len(frame_paths) < 3:
            cleanup(*temp_paths)
            return jsonify({"success": False, "matched": False, "live": False, "msg": "Frame saving failed"}), 400

        # Step 3: Face detection + Liveness checks
        face_boxes = []
        for i, p in enumerate(frame_paths):
            try:
                faces = DeepFace.extract_faces(img_path=p, detector_backend=BACKEND, enforce_detection=False)
                valid = [f for f in faces if float(f.get('confidence', 0)) > 0.5]
                if len(valid) != 1:
                    cleanup(*temp_paths)
                    return jsonify({"success": False, "matched": False, "live": False, "msg": f"Frame {i+1}: Exactly 1 face required"}), 400
                fa = valid[0]['facial_area']
                face_boxes.append(fa)
            except Exception as e:
                cleanup(*temp_paths)
                return jsonify({"success": False, "matched": False, "live": False, "msg": f"Face detection error: {str(e)}"}), 500

        # Movement check (anti-static)
        movements = []
        for i in range(len(face_boxes) - 1):
            dx = abs(face_boxes[i]['x'] - face_boxes[i+1]['x'])
            dy = abs(face_boxes[i]['y'] - face_boxes[i+1]['y'])
            movements.append(dx + dy)

        avg_movement = sum(movements) / len(movements) if movements else 0
        if avg_movement < 2:
            cleanup(*temp_paths)
            return jsonify({"success": True, "matched": False, "live": False, "msg": "Static image detected"}), 200

        # Texture check
        lap_scores = [laplacian_variance(p) for p in frame_paths]
        avg_lap = sum(lap_scores) / len(lap_scores)
        if avg_lap < 40:
            cleanup(*temp_paths)
            return jsonify({"success": True, "matched": False, "live": False, "msg": "Flat image detected"}), 200

        # Step 4: Face Match (middle frame use karo)
        best_frame = frame_paths[1]
        result = DeepFace.verify(
            img1_path=reg_path,
            img2_path=best_frame,
            model_name=MODEL,
            detector_backend=BACKEND,
            enforce_detection=False
        )

        matched = bool(result.get('verified', False))
        distance = float(result.get('distance', 0))
        threshold = float(result.get('threshold', 1))
        confidence = round(max(0, min(100, (1 - distance / threshold) * 100)), 1) if threshold > 0 else 0

        cleanup(*temp_paths)

        if not matched:
            return jsonify({
                "success": True,
                "matched": False,
                "live": True,
                "confidence": confidence,
                "msg": f"Face not matched ({confidence}%)"
            })

        return jsonify({
            "success": True,
            "matched": True,
            "live": True,
            "confidence": confidence,
            "msg": f"Verified successfully ({confidence}%)"
        })

    except Exception as e:
        print(f"[AUTO_VERIFY ERROR]: {e}")
        cleanup(*temp_paths)
        return jsonify({"success": False, "matched": False, "live": False, "msg": str(e)}), 500

# ===================== HEALTH =====================
@app.route('/')
def index():
    return jsonify({
        "status": "running",
        "version": "6.0",
        "firebase": firebase_status
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)