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
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024

# ===================== LIGHTWEIGHT CONFIG =====================
MODEL = 'SFace'                    # Lightweight & Fast
BACKEND = 'opencv'
TMP_DIR = 'temp_files'
MAX_IMAGE_SIZE = 512
LAP_THRESHOLD = 35.0
MIN_FRAMES = 3                     # ← Changed to 3 (frontend ke hisaab se)

os.makedirs(TMP_DIR, exist_ok=True)

FIRESTORE_COLLECTION = "users"

# ===================== FIREBASE =====================
if not firebase_admin._apps:
    service_account_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
    if service_account_json:
        cred_dict = json.loads(service_account_json)
        if "private_key" in cred_dict:
            cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")
        cred = credentials.Certificate(cred_dict)
        print("✅ Firebase Connected using ENV Variable")
    else:
        print("⚠️ FIREBASE_SERVICE_ACCOUNT_JSON not found!")
        cred = None

    if cred:
        firebase_admin.initialize_app(cred)
        db = firestore.client()
    else:
        db = None

# ===================== HELPERS =====================
def tmp_path(prefix='t'):
    return os.path.join(TMP_DIR, f'{prefix}_{uuid.uuid4().hex[:10]}.jpg')


def cleanup(*paths):
    for p in paths:
        try:
            if os.path.exists(p):
                os.remove(p)
        except:
            pass
    gc.collect()


def resize_image(path):
    img = cv2.imread(path)
    if img is None:
        return path
    h, w = img.shape[:2]
    if max(h, w) <= MAX_IMAGE_SIZE:
        return path
    scale = MAX_IMAGE_SIZE / max(h, w)
    new_size = (int(w * scale), int(h * scale))
    cv2.imwrite(path, cv2.resize(img, new_size, interpolation=cv2.INTER_AREA))
    return path


def laplacian_variance(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return float(cv2.Laplacian(img, cv2.CV_64F).var()) if img is not None else 0.0


def detect_faces(path):
    try:
        faces = DeepFace.extract_faces(
            img_path=path, detector_backend=BACKEND, enforce_detection=False
        )
        return [f for f in faces if float(f.get('confidence', 0)) > 0.5]
    except:
        return []


def verify_match(reg_path, live_path):
    try:
        result = DeepFace.verify(
            img1_path=reg_path,
            img2_path=live_path,
            model_name=MODEL,
            detector_backend=BACKEND,
            enforce_detection=False
        )
        matched = bool(result.get('verified', False))
        dist = float(result.get('distance', 0))
        thresh = float(result.get('threshold', 1))
        conf = max(0, min(100, (1 - dist / thresh) * 100)) if thresh > 0 else 0
        return matched, round(conf, 1)
    except:
        return False, 0.0


def get_registered_face_info(uid):
    if db is None:
        return None, "Firebase not initialized"
    try:
        doc = db.collection(FIRESTORE_COLLECTION).document(uid).get()
        if not doc.exists:
            return None, "User not found in Firebase"

        data = doc.to_dict() or {}
        for field in ["photoURL", "photoUrl", "faceURL", "faceImageUrl", "cloudinaryUrl"]:
            url = data.get(field)
            if url and isinstance(url, str) and url.startswith("http"):
                return url, data.get("name") or data.get("userName") or "User"
        return None, "photoURL field not found"
    except Exception as e:
        print(f"[Firebase Error] {e}")
        return None, str(e)


def download_cloudinary(url):
    r = requests.get(url, timeout=20, headers={"User-Agent": "Attendance-Face/1.0"})
    r.raise_for_status()
    arr = np.frombuffer(r.content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")
    path = tmp_path("reg")
    cv2.imwrite(path, img)
    return resize_image(path)


# ===================== MAIN ROUTE =====================
@app.route('/face/auto-verify', methods=['POST'])
def auto_verify():
    temp_paths = []
    try:
        d = request.json or {}
        uid = (d.get('userId') or '').strip()
        frames = d.get('framesBase64') or []

        print(f"[AUTO-VERIFY] UID: {uid} | Frames Received: {len(frames)}")

        if not uid:
            return jsonify({'success': False, 'msg': 'userId required'}), 400
        if len(frames) < MIN_FRAMES:
            return jsonify({'success': False, 'msg': f'Minimum {MIN_FRAMES} frames required'}), 400

        face_url, user_name = get_registered_face_info(uid)
        if not face_url:
            return jsonify({'success': False, 'msg': f'Face not registered: {user_name}'}), 400

        reg_path = download_cloudinary(face_url)
        temp_paths.append(reg_path)

        live_paths = []
        for i, b64 in enumerate(frames[:5]):
            p = tmp_path(f'live{i}')
            save_b64(b64, p)
            p = resize_image(p)
            temp_paths.append(p)
            live_paths.append(p)

        lap_scores = [laplacian_variance(p) for p in live_paths]
        avg_lap = sum(lap_scores) / len(lap_scores)
        print(f"[LIVENESS] Avg Laplacian: {avg_lap:.2f}")

        if avg_lap < LAP_THRESHOLD:
            return jsonify({
                'success': True,
                'matched': False,
                'live': False,
                'msg': 'Photo or screen detected. Real face use karein.'
            }), 200

        best_frame = live_paths[2] if len(live_paths) > 2 else live_paths[0]
        matched, confidence = verify_match(reg_path, best_frame)

        if matched and confidence >= 40:
            return jsonify({
                'success': True,
                'matched': True,
                'live': True,
                'confidence': confidence,
                'userName': user_name,
                'msg': f'✅ Verified Successfully ({confidence}%)'
            })
        else:
            return jsonify({
                'success': True,
                'matched': False,
                'live': True,
                'confidence': confidence,
                'msg': f'Face match nahi hua ({confidence}%). Better lighting aur face position try karein.'
            })

    except Exception as e:
        print(f"[CRITICAL ERROR] {e}")
        return jsonify({'success': False, 'msg': 'Server error occurred'}), 500
    finally:
        cleanup(*temp_paths)


@app.route('/')
def index():
    return jsonify({'status': 'running', 'version': '5.5-memory-optimized', 'min_frames': MIN_FRAMES})


@app.route('/health')
def health():
    return jsonify({'success': True, 'version': '5.5', 'status': 'healthy'})


@app.route('/face/debug/<uid>')
def debug_user(uid):
    face_url, info = get_registered_face_info(uid.strip())
    return jsonify({
        "success": True,
        "uid": uid,
        "face_url_found": bool(face_url),
        "info": info
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("=" * 80)
    print("🚀 MEMORY OPTIMIZED SERVER v5.5 (SFace + 512MB Friendly)")
    print(f"   Minimum Frames: {MIN_FRAMES}")
    print("=" * 80)
    app.run(host='0.0.0.0', port=port, debug=False)