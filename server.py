import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import gc
import json
import uuid
import traceback
from io import BytesIO
from datetime import datetime

import requests
import numpy as np
import cv2
from PIL import Image

import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024

# ===================== CONFIG =====================
MODEL = "SFace"
TMP_DIR = "temp_files"
FIRESTORE_USER_COLLECTION = "users"
MAX_IMAGE_SIZE = 512
MIN_FRAMES = 3
MIN_MOVEMENT = 1.0
MIN_LAPLACIAN = 35.0

os.makedirs(TMP_DIR, exist_ok=True)

print("=" * 80)
print("SERVER STARTING - FACE VERIFY SERVER (RESTRUCTURED)")
print("=" * 80)

# ===================== FIREBASE INIT =====================
db = None
firebase_status = "NOT_INITIALIZED"

def init_firebase():
    global db, firebase_status

    try:
        if firebase_admin._apps:
            db = firestore.client()
            firebase_status = "ALREADY_INITIALIZED"
            print("✅ Firebase already initialized")
            return
    except:
        pass

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

# ===================== LAZY DEEPFACE =====================
_DEEPFACE = None
_TF = None

def get_deepface():
    global _DEEPFACE, _TF
    if _DEEPFACE is None:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        from deepface import DeepFace
        _TF = tf
        _DEEPFACE = DeepFace
        print("✅ DeepFace loaded lazily")
    return _DEEPFACE, _TF

def release_tf_memory():
    global _TF
    try:
        if _TF is not None:
            _TF.keras.backend.clear_session()
    except:
        pass
    gc.collect()

# ===================== OPENCV FACE DETECTOR =====================
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ===================== RESPONSE HELPERS =====================
def ok(data=None, status=200):
    payload = {"success": True}
    if data:
        payload.update(data)
    return jsonify(payload), status

def fail(message="Error", status=400, matched=False, live=False, extra=None):
    payload = {
        "success": False,
        "matched": matched,
        "live": live,
        "message": message,
        "msg": message,
    }
    if extra:
        payload.update(extra)
    return jsonify(payload), status

# ===================== HELPERS =====================
def tmp_path(prefix="t"):
    return os.path.join(TMP_DIR, f"{prefix}_{uuid.uuid4().hex[:10]}.jpg")

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

        if "," in base64_str and "base64" in base64_str[:60]:
            base64_str = base64_str.split(",", 1)[1]

        img_data = base64.b64decode(base64_str)
        with open(path, "wb") as f:
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
    except:
        return 0.0

def get_registered_photo_url(uid: str, org_id: str = ""):
    if db is None:
        return None, "Firestore not initialized"

    try:
        doc_ref = db.collection(FIRESTORE_USER_COLLECTION).document(uid)
        snap = doc_ref.get()

        if not snap.exists:
            return None, "User document not found"

        data = snap.to_dict() or {}

        # Optional org validation for future multi-tenant support
        if org_id and data.get("orgId") and data.get("orgId") != org_id:
            return None, "User does not belong to requested organization"

        for key in ["photoURL", "photoUrl", "imageUrl", "cloudinaryUrl", "faceURL", "faceImageUrl"]:
            val = data.get(key)
            if isinstance(val, str) and val.strip().startswith("http"):
                return val.strip(), None

        return None, "No photo URL found in user document"
    except Exception as e:
        return None, f"Firestore error: {str(e)}"

def normalize_cloudinary_url(url: str):
    try:
        if not url or "res.cloudinary.com" not in url or "/upload/" not in url:
            return url

        url = url.replace("/f_auto/", "/f_jpg/")
        url = url.replace(",f_auto,", ",f_jpg,")
        url = url.replace(",f_auto/", ",f_jpg/")

        before, after = url.split("/upload/", 1)

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
    try:
        if not isinstance(url, str) or not url.strip().startswith("http"):
            return None, "Invalid or empty URL"

        raw_url = url.strip()
        final_url = normalize_cloudinary_url(raw_url)

        print(f"[DOWNLOAD] Raw URL: {raw_url[:180]}")
        print(f"[DOWNLOAD] Final URL: {final_url[:180]}")

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "image/jpeg,image/png,image/webp,image/*;q=0.8,*/*;q=0.5"
        }

        r = requests.get(final_url, timeout=20, headers=headers, allow_redirects=True)
        r.raise_for_status()

        content_type = r.headers.get("Content-Type", "")
        print(f"[DOWNLOAD] content-type={content_type}, size={len(r.content)}")

        try:
            img = Image.open(BytesIO(r.content)).convert("RGB")
            path = tmp_path("reg")
            img.save(path, format="JPEG", quality=95)
            return resize_image(path), None
        except Exception as pil_err:
            print(f"[DOWNLOAD] PIL failed: {pil_err}")

        try:
            arr = np.frombuffer(r.content, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return None, f"Decode failed. content-type={content_type}"
            path = tmp_path("reg")
            ok_write = cv2.imwrite(path, img)
            if not ok_write:
                return None, "Failed to write image"
            return resize_image(path), None
        except Exception as cv_err:
            return None, f"OpenCV failed: {cv_err}"

    except Exception as e:
        print(f"[Download Error] {e}")
        return None, str(e)

def extract_single_face_box_cv(path: str):
    try:
        img = cv2.imread(path)
        if img is None:
            return None, "Image read failed"

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) != 1:
            return None, f"Expected 1 face, found {len(faces)}"

        x, y, w, h = faces[0]
        return {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}, None
    except Exception as e:
        return None, str(e)

def verify_face_match(reg_path: str, live_path: str):
    try:
        DeepFace, _ = get_deepface()
        result = DeepFace.verify(
            img1_path=reg_path,
            img2_path=live_path,
            model_name=MODEL,
            detector_backend="opencv",
            enforce_detection=False
        )

        matched = bool(result.get("verified", False))
        distance = float(result.get("distance", 0))
        threshold = float(result.get("threshold", 1))
        confidence = max(0, min(100, (1 - distance / threshold) * 100)) if threshold > 0 else 0
        confidence = round(confidence, 1)

        return matched, confidence, result
    except Exception as e:
        return False, 0.0, {"error": str(e)}

# ===================== ERROR HANDLERS =====================
@app.errorhandler(404)
def not_found(e):
    return fail("Route not found", 404)

@app.errorhandler(500)
def internal_error(e):
    return fail("Internal server error", 500)

# ===================== ROUTES =====================
@app.route("/")
def index():
    return ok({
        "status": "running",
        "version": "face-server-restructured-v1",
        "firebase": firebase_status,
        "time": datetime.utcnow().isoformat()
    })

@app.route("/health")
def health():
    return ok({
        "status": "ok",
        "firebase": firebase_status,
        "time": datetime.utcnow().isoformat()
    })

@app.route("/face/register", methods=["POST"])
def register_face():
    test_path = None
    try:
        data = request.json or {}
        uid = (data.get("userId") or "").strip()
        photo_url = (data.get("photoURL") or "").strip()

        # future multi-tenant metadata
        org_id = (data.get("orgId") or "").strip()
        org_name = (data.get("orgName") or "").strip()
        admin_id = (data.get("adminId") or "").strip()

        if not uid or not photo_url:
            return fail("userId and photoURL required", 400)

        if db is None:
            return fail("Firestore not initialized", 500)

        # optional validation
        test_path, dl_err = download_image(photo_url)
        if not test_path:
            return fail(f"photoURL download failed: {dl_err}", 400)

        payload = {
            "photoURL": photo_url,
            "faceRegistered": True,
            "updatedAt": datetime.utcnow(),
        }

        # only if provided
        if org_id:
            payload["orgId"] = org_id
        if org_name:
            payload["orgName"] = org_name
        if admin_id:
            payload["adminId"] = admin_id

        db.collection(FIRESTORE_USER_COLLECTION).document(uid).set(payload, merge=True)

        return ok({
            "message": "photoURL saved successfully",
            "photoURL": photo_url
        })
    except Exception as e:
        return fail(str(e), 500)
    finally:
        cleanup(test_path)

@app.route("/face/debug-photo/<uid>", methods=["GET"])
def debug_photo(uid):
    test_path = None
    try:
        uid = (uid or "").strip()
        org_id = (request.args.get("orgId") or "").strip()

        if not uid:
            return fail("uid required", 400)

        photo_url, err = get_registered_photo_url(uid, org_id)
        if not photo_url:
            return fail(err, 404, extra={"firebase": firebase_status})

        test_path, dl_err = download_image(photo_url)
        ok_download = bool(test_path)

        return ok({
            "firebase": firebase_status,
            "photoURL": photo_url,
            "normalizedURL": normalize_cloudinary_url(photo_url),
            "downloadOk": ok_download,
            "downloadError": dl_err
        })
    except Exception as e:
        return fail(str(e), 500)
    finally:
        cleanup(test_path)

@app.route("/face/auto-verify", methods=["POST"])
def auto_verify():
    temp_paths = []
    try:
        data = request.json or {}

        uid = (data.get("userId") or "").strip()
        frames = data.get("framesBase64") or []
        incoming_photo_url = (data.get("photoURL") or "").strip()
        org_id = (data.get("orgId") or "").strip()  # future-safe

        print("=" * 60)
        print(f"[AUTO_VERIFY] uid={uid}")
        print(f"[AUTO_VERIFY] orgId={org_id}")
        print(f"[AUTO_VERIFY] frames={len(frames)}")
        print(f"[AUTO_VERIFY] request_photo_exists={bool(incoming_photo_url)}")

        if not uid:
            return fail("userId required", 400)

        if len(frames) < MIN_FRAMES:
            return fail(f"Need at least {MIN_FRAMES} frames", 400)

        photo_url = incoming_photo_url

        # fallback from firestore
        if not photo_url:
            photo_url, photo_err = get_registered_photo_url(uid, org_id)
            print(f"[AUTO_VERIFY] firestore photo_url={photo_url}")
            if not photo_url:
                return fail(f"Registered face not found: {photo_err}", 400)

        reg_path, reg_err = download_image(photo_url)
        if not reg_path:
            return fail(
                f"Registered face download failed: {reg_err}",
                400,
                extra={"photoURL": photo_url[:200]}
            )

        temp_paths.append(reg_path)

        frame_paths = []
        for i, b64 in enumerate(frames[:MIN_FRAMES]):
            p = tmp_path(f"frame_{i}")
            if save_b64(b64, p):
                p = resize_image(p)
                frame_paths.append(p)
                temp_paths.append(p)

        if len(frame_paths) < MIN_FRAMES:
            return fail("Frame saving failed", 400)

        # lightweight face detection
        face_boxes = []
        for i, p in enumerate(frame_paths):
            box, err = extract_single_face_box_cv(p)
            if not box:
                return fail(f"Frame {i+1}: {err}", 400)
            face_boxes.append(box)

        # movement check
        movements = []
        for i in range(len(face_boxes) - 1):
            dx = abs(face_boxes[i]["x"] - face_boxes[i + 1]["x"])
            dy = abs(face_boxes[i]["y"] - face_boxes[i + 1]["y"])
            dw = abs(face_boxes[i]["w"] - face_boxes[i + 1]["w"])
            dh = abs(face_boxes[i]["h"] - face_boxes[i + 1]["h"])
            movements.append(dx + dy + dw + dh)

        avg_movement = sum(movements) / len(movements) if movements else 0
        print(f"[AUTO_VERIFY] avg_movement={avg_movement}")

        if avg_movement < MIN_MOVEMENT:
            return jsonify({
                "success": True,
                "matched": False,
                "live": False,
                "confidence": 0,
                "msg": "Static image detected. Please use live face."
            }), 200

        # texture check
        lap_scores = [laplacian_variance(p) for p in frame_paths]
        avg_lap = sum(lap_scores) / len(lap_scores) if lap_scores else 0
        print(f"[AUTO_VERIFY] avg_lap={avg_lap}")

        if avg_lap < MIN_LAPLACIAN:
            return jsonify({
                "success": True,
                "matched": False,
                "live": False,
                "confidence": 0,
                "msg": "Flat image detected. Real person required."
            }), 200

        # face match
        best_frame = frame_paths[1] if len(frame_paths) >= 2 else frame_paths[0]
        matched, confidence, raw_result = verify_face_match(reg_path, best_frame)

        print(f"[AUTO_VERIFY] matched={matched}, confidence={confidence}")
        print(f"[AUTO_VERIFY] raw_result={raw_result}")

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
        print(f"[AUTO_VERIFY CRASH] {e}")
        traceback.print_exc()
        return fail(f"Server error: {str(e)}", 500)
    finally:
        cleanup(*temp_paths)
        release_tf_memory()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)