import os
# ========= MEMORY OPTIMIZATION (Sabse pehle) =========
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'        # TensorFlow logs band
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'       # oneDNN band (RAM bachao)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'        # GPU mat dhundo
os.environ['OMP_NUM_THREADS'] = '2'              # CPU threads limit

from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import uuid
import gc
import threading
import requests
import numpy as np
import cv2
from datetime import datetime

# ========= TENSORFLOW MEMORY LIMIT =========
try:
    import tensorflow as tf
    # CPU memory growth limit
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception as e:
    print(f"TF config warning: {e}")

from deepface import DeepFace

# ========= MEDIAPIPE (LIVENESS) =========
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,   # 🆕 False = kam RAM
        min_detection_confidence=0.3
    )
    MEDIAPIPE_OK = True
    print("✅ MediaPipe loaded")
except Exception as e:
    MEDIAPIPE_OK = False
    face_mesh = None
    print(f"⚠️ MediaPipe disabled: {e}")

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024

TMP_DIR = 'temp_files'
# ========= LIGHTWEIGHT MODEL (Kam RAM) =========
MODEL = 'SFace'              # 🆕 VGG-Face (580MB) → SFace (90MB)
BACKEND = 'opencv'
EAR_THRESHOLD = 0.22

os.makedirs(TMP_DIR, exist_ok=True)

LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

# ========= THREAD LOCK (Multiple requests safe) =========
model_lock = threading.Lock()

# ========= MODEL PRE-LOAD (Startup pe ek baar) =========
print("⏳ Loading face model (pehli baar thoda time lagega)...")
try:
    # Dummy image se model load karwa do
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_path = os.path.join(TMP_DIR, 'warmup.jpg')
    cv2.imwrite(dummy_path, dummy)
    try:
        DeepFace.represent(img_path=dummy_path, model_name=MODEL,
                           detector_backend='skip', enforce_detection=False)
    except:
        pass
    if os.path.exists(dummy_path):
        os.remove(dummy_path)
    print(f"✅ Model '{MODEL}' pre-loaded successfully!")
except Exception as e:
    print(f"⚠️ Model preload warning: {e}")


# ==================== HELPERS ====================

def decode_b64(b64):
    if ',' in b64:
        b64 = b64.split(',')[1]
    return base64.b64decode(b64)


def save_b64(b64, path):
    with open(path, 'wb') as f:
        f.write(decode_b64(b64))


def tmp_path(prefix='t'):
    return os.path.join(TMP_DIR, f'{prefix}_{uuid.uuid4().hex[:12]}.jpg')


def cleanup(*paths):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except:
            pass


def download_image(url, save_path):
    """Cloudinary URL se image download"""
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return False, f'Download failed (status {resp.status_code})'
        with open(save_path, 'wb') as f:
            f.write(resp.content)
        return True, 'OK'
    except Exception as e:
        return False, str(e)


def get_registered_face(uid, photo_url):
    """Cloudinary photoURL se face nikalo"""
    if photo_url:
        reg_temp = tmp_path('reg')
        ok, msg = download_image(photo_url, reg_temp)
        if ok:
            return reg_temp, True, None
        else:
            return None, False, f'Photo download failed: {msg}'
    return None, False, 'photoURL required (no photo provided)'


def detect_faces(path):
    """Face detect - thread safe"""
    try:
        with model_lock:
            faces = DeepFace.extract_faces(
                img_path=path,
                detector_backend=BACKEND,
                enforce_detection=False
            )
        return [f for f in faces if float(f.get('confidence', 0)) > 0.85]  # 🆕 0.85 strict
    except Exception as e:
        print(f"detect_faces error: {e}")
        return []


def calc_ear(landmarks, indices, w, h):
    pts = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in indices]
    v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    horiz = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (v1 + v2) / (2.0 * horiz) if horiz > 0 else 0.0


def eye_state(path):
    if not MEDIAPIPE_OK:
        return 'unknown', 'mediapipe off'
    try:
        img = cv2.imread(path)
        if img is None:
            return 'error', 'cant read image'
        h, w = img.shape[:2]
        if h < 50 or w < 50:
            return 'error', f'image too small'
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if not res or not res.multi_face_landmarks:
            return 'error', 'no landmarks'
        lm = res.multi_face_landmarks[0]
        le = calc_ear(lm, LEFT_EYE_IDX, w, h)
        re = calc_ear(lm, RIGHT_EYE_IDX, w, h)
        avg = (le + re) / 2.0
        state = 'closed' if avg < EAR_THRESHOLD else 'open'
        return state, f'EAR={avg:.3f}'
    except Exception as e:
        return 'error', str(e)


def verify_match(reg_path, live_path):
    """Face match - thread safe"""
    with model_lock:
        result = DeepFace.verify(
            img1_path=reg_path,
            img2_path=live_path,
            model_name=MODEL,
            detector_backend=BACKEND,
            enforce_detection=False,
        )
    matched = bool(result.get('verified', False))
    dist = float(result.get('distance', 0))
    thresh = float(result.get('threshold', 1))
    conf = max(0, min(100, (1 - dist / thresh) * 100)) if thresh > 0 else 0
    return matched, round(conf, 2), round(dist, 4), round(thresh, 4)


def laplacian_variance(path):
    """Texture check - photo/screen detect"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def get_face_box(path):
    """Face ka location nikalo (movement check ke liye)"""
    try:
        with model_lock:
            faces = DeepFace.extract_faces(
                img_path=path,
                detector_backend=BACKEND,
                enforce_detection=False
            )
        valid = [f for f in faces if float(f.get('confidence', 0)) > 0.85]
        if len(valid) == 1:
            fa = valid[0].get('facial_area', {})
            return {
                'x': fa.get('x', 0), 'y': fa.get('y', 0),
                'w': fa.get('w', 0), 'h': fa.get('h', 0),
                'count': 1
            }
        return {'count': len(valid)}
    except Exception as e:
        print(f"get_face_box error: {e}")
        return {'count': 0}


# ==================== ROUTES ====================

@app.route('/')
def index():
    return jsonify({
        'status': 'running',
        'version': '6.0-optimized',
        'model': MODEL,
        'liveness': MEDIAPIPE_OK,
        'storage': 'Cloudinary photoURL',
        'features': ['low-RAM', 'multi-request', 'liveness', 'anti-spoof']
    })


@app.route('/health')
def health():
    return jsonify({
        'success': True,
        'liveness': MEDIAPIPE_OK,
        'model': MODEL,
        'time': datetime.now().isoformat()
    })


# ============ MAIN ENDPOINT - AUTO VERIFY ============
@app.route('/face/auto-verify', methods=['POST'])
def auto_verify():
    temp_paths = []
    reg_path = None
    reg_is_temp = False
    try:
        d = request.json or {}
        uid = (d.get('userId') or '').strip()
        frames = d.get('framesBase64') or []
        photo_url = (d.get('photoURL') or '').strip()

        # --- Validation ---
        if not uid:
            return jsonify({'success': False, 'matched': False, 'live': False,
                            'msg': 'userId required'}), 400
        if len(frames) < 3:
            return jsonify({'success': False, 'matched': False, 'live': False,
                            'msg': 'Need 3 frames'}), 400
        if not photo_url:
            return jsonify({'success': False, 'matched': False, 'live': False,
                            'msg': 'photoURL required'}), 400

        # --- Cloudinary se registered photo download ---
        reg_path, reg_is_temp, err = get_registered_face(uid, photo_url)
        if err:
            return jsonify({'success': False, 'matched': False, 'live': False,
                            'msg': err}), 400

        # --- Registered photo me face check ---
        if len(detect_faces(reg_path)) == 0:
            return jsonify({'success': False, 'matched': False, 'live': False,
                            'msg': 'No face in profile photo. Contact admin.'}), 400

        # --- Frames save karo ---
        for i, b64 in enumerate(frames[:3]):
            p = tmp_path(f'auto_{i}')
            save_b64(b64, p)
            temp_paths.append(p)

        # --- LAYER 1: Har frame me exactly 1 face check ---
        face_boxes = []
        for i, p in enumerate(temp_paths):
            box = get_face_box(p)
            if box.get('count', 0) != 1:
                cleanup(*temp_paths, reg_path)
                return jsonify({'success': False, 'matched': False, 'live': False,
                                'msg': f'Frame {i+1}: {box.get("count",0)} faces. Sirf 1 face chahiye.'}), 400
            face_boxes.append(box)

        # --- LAYER 2: Movement check (Liveness - static photo reject) ---
        movements = []
        for i in range(len(face_boxes) - 1):
            dx = abs(face_boxes[i]['x'] - face_boxes[i+1]['x'])
            dy = abs(face_boxes[i]['y'] - face_boxes[i+1]['y'])
            dw = abs(face_boxes[i]['w'] - face_boxes[i+1]['w'])
            dh = abs(face_boxes[i]['h'] - face_boxes[i+1]['h'])
            movements.append(dx + dy + dw + dh)
        avg_movement = sum(movements) / len(movements) if movements else 0
        print(f"[AutoVerify] movement={avg_movement:.2f}")

        if avg_movement < 1.0:
            cleanup(*temp_paths, reg_path)
            return jsonify({'success': True, 'matched': False, 'live': False,
                            'msg': 'Static image! Phone naturally pakdo, photo use mat karo.'}), 200

        # --- LAYER 3: Texture check (Flat photo/screen reject) ---
        lap_scores = [laplacian_variance(p) for p in temp_paths]
        avg_lap = sum(lap_scores) / len(lap_scores)
        print(f"[AutoVerify] texture={avg_lap:.2f}")

        if avg_lap < 25.0:
            cleanup(*temp_paths, reg_path)
            return jsonify({'success': True, 'matched': False, 'live': False,
                            'msg': 'Flat image detected. Real person chahiye.'}), 200

        # --- LAYER 4: Blink check (optional - agar mediapipe on) ---
        live_confirmed = True
        if MEDIAPIPE_OK:
            eye_states = [eye_state(p)[0] for p in temp_paths]
            print(f"[AutoVerify] eyes={eye_states}")
            # Agar saare frames me eye state error aaye to skip
            valid_states = [s for s in eye_states if s in ('open', 'closed')]
            # Blink optional rakha - sirf movement + texture pe rely karenge

        # --- LAYER 5: Face Match ---
        best_frame = temp_paths[1] if len(temp_paths) >= 2 else temp_paths[0]
        matched, conf, dist, thresh = verify_match(reg_path, best_frame)
        print(f"[AutoVerify] matched={matched} conf={conf}")

        cleanup(*temp_paths, reg_path)

        if not matched:
            return jsonify({'success': True, 'matched': False, 'live': True,
                            'confidence': conf,
                            'msg': f'Face NOT matched ❌ ({conf}%). Admin se re-register karwao.'}), 200

        return jsonify({
            'success': True,
            'matched': True,
            'live': True,
            'confidence': conf,
            'msg': f'✅ Verified + Matched ({conf}%)',
            'time': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"[AutoVerify CRASH] {str(e)}")
        cleanup(*temp_paths)
        if reg_is_temp and reg_path:
            cleanup(reg_path)
        return jsonify({'success': False, 'matched': False, 'live': False,
                        'msg': f'Server error: {str(e)}'}), 500
    finally:
        # 🆕 Memory cleanup - crash prevent
        gc.collect()


# ============ SIMPLE VERIFY (Optional) ============
@app.route('/face/verify', methods=['POST'])
def verify():
    t = None
    reg_path = None
    reg_is_temp = False
    try:
        d = request.json or {}
        uid = (d.get('userId') or '').strip()
        b64 = d.get('imageBase64') or d.get('image') or ''
        photo_url = (d.get('photoURL') or '').strip()

        if not uid or not b64:
            return jsonify({'success': False, 'matched': False, 'msg': 'userId + image required'}), 400

        reg_path, reg_is_temp, err = get_registered_face(uid, photo_url)
        if err:
            return jsonify({'success': False, 'matched': False, 'msg': err}), 400

        t = tmp_path('v')
        save_b64(b64, t)

        faces = detect_faces(t)
        if len(faces) == 0:
            return jsonify({'success': False, 'matched': False, 'msg': 'No face detected'}), 400
        if len(faces) > 1:
            return jsonify({'success': False, 'matched': False, 'msg': 'Multiple faces'}), 400

        matched, conf, dist, thresh = verify_match(reg_path, t)
        return jsonify({
            'success': True, 'matched': matched, 'confidence': conf,
            'msg': f'Matched {conf}%' if matched else f'Not matched {conf}%'
        })
    except Exception as e:
        print(f"verify error: {e}")
        return jsonify({'success': False, 'matched': False, 'msg': str(e)}), 500
    finally:
        cleanup(t)
        if reg_is_temp:
            cleanup(reg_path)
        gc.collect()


# ============ CHECK EYES (Optional) ============
@app.route('/face/check-eyes', methods=['POST'])
def check_eyes():
    t = None
    try:
        if not MEDIAPIPE_OK:
            return jsonify({'success': False, 'msg': 'mediapipe not available'}), 500
        d = request.json or {}
        b64 = d.get('imageBase64') or ''
        if not b64:
            return jsonify({'success': False, 'msg': 'image required'}), 400
        t = tmp_path('eye')
        save_b64(b64, t)
        state, detail = eye_state(t)
        return jsonify({'success': state != 'error', 'eyeState': state, 'detail': detail})
    except Exception as e:
        return jsonify({'success': False, 'msg': str(e)}), 500
    finally:
        cleanup(t)
        gc.collect()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("=" * 60)
    print(" FACE SERVER v6.0 - OPTIMIZED (Low RAM + Multi-Request) ")
    print(f" Model: {MODEL} (lightweight)")
    print(f" Port: {port}")
    print(f" Liveness: {'ON' if MEDIAPIPE_OK else 'OFF'}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)  # 🆕 threaded