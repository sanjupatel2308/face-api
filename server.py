from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import base64
import os
import uuid
from gunicorn.util import get_username
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
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024

# ===================== CONFIG =====================
MODEL = 'SFace'
BACKEND = 'opencv'
TMP_DIR = 'temp_files'
MAX_IMAGE_SIZE = 512
LAP_THRESHOLD = 35.0
MIN_FRAMES = 3

os.makedirs(TMP_DIR, exist_ok=True)

FIRESTORE_COLLECTION = "users"

print("="*80)
print("SERVER STARTING - v5.6 DEBUG MODE")
print("="*80)

# ===================== FIREBASE (ROBUST INITIALIZATION) =====================
db = None
firebase_status = "NOT_INITIALIZED"

service_account_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")

if service_account_json:
    print("✅ FIREBASE_SERVICE_ACCOUNT_JSON environment variable FOUND")
    try:
        cred_dict = json.loads(service_account_json)
        if "private_key" in cred_dict:
            cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")
        
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        firebase_status = "SUCCESSFULLY_INITIALIZED"
        print("✅ Firebase Successfully Initialized")
    except json.JSONDecodeError:
        print("❌ JSON Parse Error: FIREBASE_SERVICE_ACCOUNT_JSON is not valid JSON")
        firebase_status = "JSON_PARSE_ERROR"
    except Exception as e:
        print(f"❌ Firebase Initialization Failed: {e}")
        firebase_status = f"ERROR: {e}"
else:
    print("❌ FIREBASE_SERVICE_ACCOUNT_JSON environment variable NOT FOUND!")
    firebase_status = "ENV_VARIABLE_MISSING"

print(f"Firebase Status: {firebase_status}")
print("="*80)


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
        faces = DeepFace.extract_faces(img_path=path, detector_backend=BACKEND, enforce_detection=False)
        return [f for f in faces if float(f.get('confidence', 0)) > 0.5]
    except:
        return []


def verify_match(reg_path, live_path):
    try:
        result = DeepFace.verify(
            img1_path=reg_path, img2_path=live_path,
            model_name=MODEL, detector_backend=BACKEND, enforce_detection=False
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
        return None, f"Firebase not initialized ({firebase_status})"
    try:
        doc = db.collection(FIRESTORE_COLLECTION).document(uid).get()
        if not doc.exists:
            return None, "User document not found"

        data = doc.to_dict() or {}
        for field in ["photoURL", "photoUrl", "faceURL", "faceImageUrl", "cloudinaryUrl", "imageUrl"]:
            url = data.get(field)
            if url and isinstance(url, str) and url.startswith("http"):
                return url, data.get("name") or data.get("userName") or "User"
        return None, "photoURL field not found in document"
    except Exception as e:
        return None, f"Firebase error: {str(e)}"


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
    """
    Simple robust auto-verify:
    - 3 frames capture
    - Movement check (natural vs static)
    - Texture check (real vs flat photo)
    - Face match with registered
    """
    temp_paths = []
    try:
        d = request.json or {}
        uid = (d.get('userId') or '').strip()
        frames = d.get('framesBase64') or []

        if not uid:
            return jsonify({'success': False, 'matched': False, 'live': False, 'msg': 'userId required'}), 400
        if len(frames) < 3:
            return jsonify({'success': False, 'matched': False, 'live': False, 'msg': 'Need 3 frames'}), 400

        reg = os.path.join(FACES_DIR, uid, 'face.jpg')
        if not os.path.exists(reg):
            return jsonify({'success': False, 'matched': False, 'live': False, 'msg': 'Face not registered'}), 400

        # Save frames
        for i, b64 in enumerate(frames[:3]):
            p = tmp_path(f'auto_{i}')
            save_b64(b64, p)
            temp_paths.append(p)

        # Step 1: Check each frame has exactly 1 face
        face_boxes = []
        for i, p in enumerate(temp_paths):
            try:
                faces = DeepFace.extract_faces(img_path=p, detector_backend=BACKEND, enforce_detection=False)
                valid = [f for f in faces if float(f.get('confidence', 0)) > 0.5]
                if len(valid) != 1:
                    cleanup(*temp_paths)
                    return jsonify({'success': False, 'matched': False, 'live': False, 'msg': f'Frame {i+1}: {len(valid)} faces found. Exactly 1 face needed.'}), 400
                fa = valid[0].get('facial_area', {})
                face_boxes.append({'x': fa.get('x',0), 'y': fa.get('y',0), 'w': fa.get('w',0), 'h': fa.get('h',0)})
            except Exception as e:
                cleanup(*temp_paths)
                return jsonify({'success': False, 'matched': False, 'live': False, 'msg': f'Frame {i+1} face detect error: {str(e)}'}), 500

        # Step 2: Movement check (anti-static/anti-photo)
        # Real person: face moves slightly between frames
        # Photo: face stays exactly same
        movements = []
        for i in range(len(face_boxes)-1):
            dx = abs(face_boxes[i]['x'] - face_boxes[i+1]['x'])
            dy = abs(face_boxes[i]['y'] - face_boxes[i+1]['y'])
            dw = abs(face_boxes[i]['w'] - face_boxes[i+1]['w'])
            dh = abs(face_boxes[i]['h'] - face_boxes[i+1]['h'])
            movements.append(dx + dy + dw + dh)

        avg_movement = sum(movements) / len(movements) if movements else 0
        print(f"Auto-verify: avg_movement={avg_movement}")

        # If zero movement = static photo held still
        if avg_movement < 1.0:
            cleanup(*temp_paths)
            return jsonify({'success': True, 'matched': False, 'live': False, 'msg': 'Static image detected. Please hold phone naturally and do not use photos.'}), 200

        # Step 3: Texture check (Laplacian variance)
        lap_scores = [laplacian_variance(p) for p in temp_paths]
        avg_lap = sum(lap_scores) / len(lap_scores)
        print(f"Auto-verify: avg_lap={avg_lap}")

        if avg_lap < 45.0:  # Slightly lower threshold for mobile cameras
            cleanup(*temp_paths)
            return jsonify({'success': True, 'matched': False, 'live': False, 'msg': 'Flat image detected. Real person required.'}), 200

        # Step 4: Screen/Moire check
        for i, p in enumerate(temp_paths):
            is_moire, ratio = detect_moire(p)
            if is_moire:
                cleanup(*temp_paths)
                return jsonify({'success': True, 'matched': False, 'live': False, 'msg': 'Screen/photo spoof detected. Please do not use another device screen.'}), 200

        # Step 5: Face Match (use middle frame = frame 1)
        best_frame = temp_paths[1] if len(temp_paths) >= 2 else temp_paths[0]
        result = DeepFace.verify(
            img1_path=reg, img2_path=best_frame,
            model_name=MODEL, detector_backend=BACKEND,
            enforce_detection=False,
        )

        matched = bool(result.get('verified', False))
        distance = float(result.get('distance', 0))
        threshold = float(result.get('threshold', 1))
        confidence = max(0, min(100, (1 - distance / threshold) * 100)) if threshold > 0 else 0

        if not matched:
            cleanup(*temp_paths)
            return jsonify({
                'success': True, 'matched': False, 'live': True,
                'confidence': round(confidence, 2),
                'msg': f'Face NOT matched ❌ ({round(confidence,1)}%). Please contact admin to re-register your face.'
            }), 200

        # ALL PASSED
        cleanup(*temp_paths)
        return jsonify({
            'success': True,
            'matched': True,
            'live': True,
            'confidence': round(confidence, 2),
            'userName': get_username(uid),
            'msg': f'✅ Verified + Matched ({round(confidence,1)}%)',
            'time': datetime.now().isoformat()
        })

    except Exception as e:
        import traceback
        print(f"AUTO_VERIFY CRASH: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'matched': False, 'live': False, 'msg': f'Server error: {str(e)}'}), 500
    finally:
        cleanup(*temp_paths)


@app.route('/face/debug/<uid>')
def debug_user(uid):
    face_url, info = get_registered_face_info(uid.strip())
    return jsonify({
        "success": True,
        "uid": uid,
        "firebase_status": firebase_status,
        "face_url_found": bool(face_url),
        "info": info
    })


@app.route('/')
def index():
    return jsonify({
        'status': 'running',
        'version': '5.6-debug',
        'firebase': firebase_status,
        'min_frames': MIN_FRAMES
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("="*80)
    print("FINAL DEBUG VERSION v5.6")
    print(f"Firebase Status: {firebase_status}")
    print("="*80)
    app.run(host='0.0.0.0', port=port, debug=False)