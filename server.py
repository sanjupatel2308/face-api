from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import base64
import os
import uuid
import gc
import json
from datetime import datetime
from io import BytesIO

import requests
import numpy as np
import cv2
from PIL import Image

import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024

# ===================== CONFIG =====================
MODEL = 'SFace'
BACKEND = 'opencv'
TMP_DIR = 'temp_files'
FIRESTORE_COLLECTION = 'users'
MAX_IMAGE_SIZE = 512
MIN_FRAMES = 3

# Liveness thresholds
MIN_MOVEMENT = 1.0
MIN_LAPLACIAN = 35.0

os.makedirs(TMP_DIR, exist_ok=True)

print("=" * 80)
print("SERVER STARTING - FINAL FACE VERIFY SERVER")
print("=" * 80)

# ===================== FIREBASE INIT =====================
db = None
firebase_status = "NOT_INITIALIZED"

def init_firebase():
    global db, firebase_status

    if firebase_admin._apps:
        db = firestore.client()
        firebase_status = "ALREADY_INITIALIZED"
        print("✅ Firebase already initialized")
        return

    service_account_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")

    if service_account_json:
        try:
            cred_dict = json.loads(service_account_json)
            if "private_key" in cred_dict:
                cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")

            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            firebase_status = "SUCCESS_ENV"
            print("✅ Firebase initialized from ENV")
            return
        except Exception as e:
            print(f"❌ Firebase ENV init failed: {e}")

    if os.path.exists("serviceAccountKey.json"):
        try:
            cred = credentials.Certificate("serviceAccountKey.json")
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            firebase_status = "SUCCESS_LOCAL"
            print("✅ Firebase initialized from local file")
            return
        except Exception as e:
            print(f"❌ Firebase local init failed: {e}")

    firebase_status = "DISABLED"
    print("⚠️ Firebase disabled")

init_firebase()
print(f"Firebase Status: {firebase_status}")

# ===================== HELPERS =====================
def tmp_path(prefix='t'):
    return os.path.join(TMP_DIR, f'{prefix}_{uuid.uuid4().hex[:10]}.jpg')

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
        if not base64_str:
            return False

        # support data:image/...;base64,...
        if ',' in base64_str and 'base64' in base64_str[:60]:
            base64_str = base64_str.split(',', 1)[1]

        img_data = base64.b64decode(base64_str)
        with open(path, 'wb') as f:
            f.write(img_data)
        return True
    except Exception as e:
        print(f"[save_b64 ERROR] {e}")
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
        resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(path, resized)
        return path
    except Exception as e:
        print(f"[resize_image ERROR] {e}")
        return path

def laplacian_variance(path):
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        return float(cv2.Laplacian(img, cv2.CV_64F).var())
    except Exception:
        return 0.0

def get_registered_photo_url(uid: str):
    if db is None:
        return None, "Firestore not initialized"

    try:
        doc = db.collection(FIRESTORE_COLLECTION).document(uid).get()
        if not doc.exists:
            return None, "User document not found"

        data = doc.to_dict() or {}

        for key in ["photoURL", "photoUrl", "imageUrl", "cloudinaryUrl", "faceURL", "faceImageUrl"]:
            val = data.get(key)
            if isinstance(val, str) and val.strip().startswith("http"):
                return val.strip(), None

        return None, "No photo URL found in user document"
    except Exception as e:
        return None, f"Firestore error: {str(e)}"

def normalize_cloudinary_url(url: str):
    """
    Cloudinary URL ko JPG-friendly banata hai.
    """
    try:
        if not url or "res.cloudinary.com" not in url or "/upload/" not in url:
            return url

        # common replacements
        url = url.replace("/f_auto/", "/f_jpg/")
        url = url.replace(",f_auto,", ",f_jpg,")
        url = url.replace(",f_auto/", ",f_jpg/")

        before, after = url.split("/upload/", 1)

        # agar upload/ ke baad koi transformation nahi hai to add kar do
        if not (
            after.startswith("f_") or
            after.startswith("q_") or
            after.startswith("c_") or
            after.startswith("w_") or
            after.startswith("h_")
        ):
            url = before + "/upload/f_jpg,q_auto/" + after

        return url
    except Exception as e:
        print(f"[normalize_cloudinary_url ERROR] {e}")
        return url

def download_image(url: str):
    """
    URL se image download karke local JPG save karta hai.
    Returns: (path, error)
    """
    try:
        if not isinstance(url, str) or not url.strip().startswith("http"):
            return None, "Invalid or empty URL"

        raw_url = url.strip()
        final_url = normalize_cloudinary_url(raw_url)

        print(f"[DOWNLOAD] Raw URL: {raw_url[:200]}")
        print(f"[DOWNLOAD] Final URL: {final_url[:200]}")

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "image/jpeg,image/png,image/webp,image/*;q=0.8,*/*;q=0.5"
        }

        r = requests.get(final_url, timeout=20, headers=headers, allow_redirects=True)
        r.raise_for_status()

        content_type = r.headers.get("Content-Type", "")
        print(f"[DOWNLOAD] content-type={content_type}, size={len(r.content)}")

        # PIL first
        try:
            img = Image.open(BytesIO(r.content)).convert("RGB")
            path = tmp_path("reg")
            img.save(path, format="JPEG", quality=95)
            return resize_image(path), None
        except Exception as pil_err:
            print(f"[DOWNLOAD] PIL failed: {pil_err}")

        # OpenCV fallback
        try:
            arr = np.frombuffer(r.content, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return None, f"Decode failed. content-type={content_type}"
            path = tmp_path("reg")
            ok = cv2.imwrite(path, img)
            if not ok:
                return None, "Failed to write image"
            return resize_image(path), None
        except Exception as cv_err:
            return None, f"OpenCV failed: {cv_err}"

    except Exception as e:
        print(f"[Download Error] {e}")
        return None, str(e)

def extract_single_face_box(path: str):
    try:
        faces = DeepFace.extract_faces(
            img_path=path,
            detector_backend=BACKEND,
            enforce_detection=False
        )
        valid = [f for f in faces if float(f.get('confidence', 0)) > 0.5]

        if len(valid) != 1:
            return None, f"Expected 1 face, found {len(valid)}"

        fa = valid[0].get('facial_area', {})
        return {
            'x': fa.get('x', 0),
            'y': fa.get('y', 0),
            'w': fa.get('w', 0),
            'h': fa.get('h', 0),
        }, None
    except Exception as e:
        return None, str(e)

def verify_face_match(reg_path: str, live_path: str):
    try:
        result = DeepFace.verify(
            img1_path=reg_path,
            img2_path=live_path,
            model_name=MODEL,
            detector_backend=BACKEND,
            enforce_detection=False
        )

        matched = bool(result.get('verified', False))
        distance = float(result.get('distance', 0))
        threshold = float(result.get('threshold', 1))
        confidence = max(0, min(100, (1 - distance / threshold) * 100)) if threshold > 0 else 0
        confidence = round(confidence, 1)

        return matched, confidence, result
    except Exception as e:
        return False, 0.0, {"error": str(e)}

# ===================== ROUTES =====================

@app.route('/')
def index():
    return jsonify({
        "status": "running",
        "version": "final-face-server",
        "firebase": firebase_status,
        "time": datetime.utcnow().isoformat()
    })

@app.route('/health')
def health():
    return jsonify({
        "success": True,
        "status": "ok",
        "firebase": firebase_status,
        "time": datetime.utcnow().isoformat()
    })

@app.route('/face/register', methods=['POST'])
def register_face():
    """
    Sirf Cloudinary photoURL ko Firestore me save/update karega.
    Image upload yahan nahi hota.
    """
    temp_test_path = None
    try:
        data = request.json or {}
        uid = (data.get('userId') or '').strip()
        photo_url = (data.get('photoURL') or '').strip()

        if not uid or not photo_url:
            return jsonify({
                "success": False,
                "message": "userId and photoURL required"
            }), 400

        if db is None:
            return jsonify({
                "success": False,
                "message": "Firestore not initialized"
            }), 500

        # optional validation: photo download ho pa rahi hai ya nahi
        temp_test_path, dl_err = download_image(photo_url)
        if not temp_test_path:
            return jsonify({
                "success": False,
                "message": f"photoURL download failed: {dl_err}"
            }), 400

        db.collection(FIRESTORE_COLLECTION).document(uid).set({
            "photoURL": photo_url,
            "faceRegistered": True,
            "updatedAt": datetime.utcnow(),
        }, merge=True)

        return jsonify({
            "success": True,
            "message": "photoURL saved successfully",
            "photoURL": photo_url
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500
    finally:
        cleanup(temp_test_path)

@app.route('/face/debug-photo/<uid>', methods=['GET'])
def debug_photo(uid):
    test_path = None
    try:
        uid = (uid or '').strip()
        if not uid:
            return jsonify({
                "success": False,
                "message": "uid required"
            }), 400

        photo_url, err = get_registered_photo_url(uid)
        if not photo_url:
            return jsonify({
                "success": False,
                "firebase": firebase_status,
                "message": err
            }), 404

        test_path, dl_err = download_image(photo_url)
        ok = bool(test_path)

        return jsonify({
            "success": True,
            "firebase": firebase_status,
            "photoURL": photo_url,
            "normalizedURL": normalize_cloudinary_url(photo_url),
            "downloadOk": ok,
            "downloadError": dl_err
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500
    finally:
        cleanup(test_path)

@app.route('/face/auto-verify', methods=['POST'])
def auto_verify():
    temp_paths = []
    try:
        data = request.json or {}

        uid = (data.get('userId') or '').strip()
        frames = data.get('framesBase64') or []
        incoming_photo_url = (data.get('photoURL') or '').strip()

        print("=" * 60)
        print(f"[AUTO_VERIFY] uid={uid}")
        print(f"[AUTO_VERIFY] frames={len(frames)}")
        print(f"[AUTO_VERIFY] request_photo_exists={bool(incoming_photo_url)}")

        if not uid:
            return jsonify({
                "success": False,
                "matched": False,
                "live": False,
                "msg": "userId required"
            }), 400

        if len(frames) < MIN_FRAMES:
            return jsonify({
                "success": False,
                "matched": False,
                "live": False,
                "msg": f"Need at least {MIN_FRAMES} frames"
            }), 400

        # 1) request se photoURL lo
        photo_url = incoming_photo_url

        # 2) fallback Firestore se
        if not photo_url:
            photo_url, photo_err = get_registered_photo_url(uid)
            print(f"[AUTO_VERIFY] firestore photo_url={photo_url}")
            if not photo_url:
                return jsonify({
                    "success": False,
                    "matched": False,
                    "live": False,
                    "msg": f"Registered face not found: {photo_err}"
                }), 400

        # 3) registered image download
        reg_path, reg_err = download_image(photo_url)
        if not reg_path:
            return jsonify({
                "success": False,
                "matched": False,
                "live": False,
                "msg": f"Registered face download failed: {reg_err}",
                "photoURL": photo_url[:200]
            }), 400

        temp_paths.append(reg_path)

        # 4) frames save
        frame_paths = []
        for i, b64 in enumerate(frames[:MIN_FRAMES]):
            p = tmp_path(f'frame_{i}')
            if save_b64(b64, p):
                p = resize_image(p)
                frame_paths.append(p)
                temp_paths.append(p)

        if len(frame_paths) < MIN_FRAMES:
            cleanup(*temp_paths)
            return jsonify({
                "success": False,
                "matched": False,
                "live": False,
                "msg": "Frame saving failed"
            }), 400

        # 5) each frame should contain exactly one face
        face_boxes = []
        for i, p in enumerate(frame_paths):
            box, err = extract_single_face_box(p)
            if not box:
                cleanup(*temp_paths)
                return jsonify({
                    "success": False,
                    "matched": False,
                    "live": False,
                    "msg": f"Frame {i+1}: {err}"
                }), 400
            face_boxes.append(box)

        # 6) movement check
        movements = []
        for i in range(len(face_boxes) - 1):
            dx = abs(face_boxes[i]['x'] - face_boxes[i + 1]['x'])
            dy = abs(face_boxes[i]['y'] - face_boxes[i + 1]['y'])
            dw = abs(face_boxes[i]['w'] - face_boxes[i + 1]['w'])
            dh = abs(face_boxes[i]['h'] - face_boxes[i + 1]['h'])
            movements.append(dx + dy + dw + dh)

        avg_movement = sum(movements) / len(movements) if movements else 0
        print(f"[AUTO_VERIFY] avg_movement={avg_movement}")

        if avg_movement < MIN_MOVEMENT:
            cleanup(*temp_paths)
            return jsonify({
                "success": True,
                "matched": False,
                "live": False,
                "msg": "Static image detected. Please use live face."
            }), 200

        # 7) texture check
        lap_scores = [laplacian_variance(p) for p in frame_paths]
        avg_lap = sum(lap_scores) / len(lap_scores) if lap_scores else 0
        print(f"[AUTO_VERIFY] avg_lap={avg_lap}")

        if avg_lap < MIN_LAPLACIAN:
            cleanup(*temp_paths)
            return jsonify({
                "success": True,
                "matched": False,
                "live": False,
                "msg": "Flat image detected. Real person required."
            }), 200

        # 8) face match using middle frame
        best_frame = frame_paths[1] if len(frame_paths) >= 2 else frame_paths[0]
        matched, confidence, raw_result = verify_face_match(reg_path, best_frame)

        print(f"[AUTO_VERIFY] matched={matched}, confidence={confidence}")
        print(f"[AUTO_VERIFY] raw_result={raw_result}")

        cleanup(*temp_paths)

        if not matched:
            return jsonify({
                "success": True,
                "matched": False,
                "live": True,
                "confidence": confidence,
                "msg": f"Face not matched ({confidence}%)"
            }), 200

        return jsonify({
            "success": True,
            "matched": True,
            "live": True,
            "confidence": confidence,
            "msg": f"Verified successfully ({confidence}%)",
            "time": datetime.utcnow().isoformat()
        }), 200

    except Exception as e:
        import traceback
        print(f"[AUTO_VERIFY CRASH] {e}")
        traceback.print_exc()
        cleanup(*temp_paths)
        return jsonify({
            "success": False,
            "matched": False,
            "live": False,
            "msg": f"Server error: {str(e)}"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)