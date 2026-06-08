# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from deepface import DeepFace
# import base64
# import os
# import json
# import shutil
# import uuid
# import numpy as np
# import cv2
# from datetime import datetime

# # ========= MEDIAPIPE (LIVENESS) =========
# try:
#     import mediapipe as mp
#     mp_face_mesh = mp.solutions.face_mesh
#     face_mesh = mp_face_mesh.FaceMesh(
#         static_image_mode=True,
#         max_num_faces=1,
#         refine_landmarks=True,
#         min_detection_confidence=0.5
#     )
#     MEDIAPIPE_OK = True
#     print("✅ MediaPipe loaded successfully")
# except Exception as e:
#     MEDIAPIPE_OK = False
#     mp_face_mesh = None
#     face_mesh = None
#     print(f"⚠️ MediaPipe disabled: {e}")

# app = Flask(__name__)
# CORS(app)
# app.config['MAX_CONTENT_LENGTH'] = 80 * 1024 * 1024

# FACES_DIR = 'registered_faces'
# TMP_DIR = 'temp_files'
# MODEL = 'VGG-Face'
# BACKEND = 'opencv'
# EAR_THRESHOLD = 0.22

# MIN_FRAMES = 5
# MAX_FRAMES = 7
# LAP_THRESHOLD = 60.0
# AREA_VAR_MIN = 0.03
# MOIRE_THRESHOLD = 14.0
# TEMP_STATIC_MAX = 2.5
# TEMP_JUMP_RATIO = 3.5

# os.makedirs(FACES_DIR, exist_ok=True)
# os.makedirs(TMP_DIR, exist_ok=True)

# LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
# RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]


# def decode_b64(b64):
#     if ',' in b64:
#         b64 = b64.split(',')[1]
#     return base64.b64decode(b64)


# def save_b64(b64, path):
#     with open(path, 'wb') as f:
#         f.write(decode_b64(b64))


# def tmp_path(prefix='t'):
#     return os.path.join(TMP_DIR, f'{prefix}_{uuid.uuid4().hex[:12]}.jpg')


# def cleanup(*paths):
#     for p in paths:
#         try:
#             if p and os.path.exists(p):
#                 os.remove(p)
#         except:
#             pass


# def detect_faces(path):
#     try:
#         faces = DeepFace.extract_faces(
#             img_path=path, detector_backend=BACKEND, enforce_detection=False
#         )
#         return [f for f in faces if float(f.get('confidence', 0)) > 0.5]
#     except Exception as e:
#         print(f"detect_faces error: {e}")
#         return []


# def calc_ear(landmarks, indices, w, h):
#     pts = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in indices]
#     v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
#     v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
#     horiz = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
#     return (v1 + v2) / (2.0 * horiz) if horiz > 0 else 0.0


# def eye_state(path):
#     if not MEDIAPIPE_OK:
#         return 'error', 'mediapipe not available'
#     try:
#         img = cv2.imread(path)
#         if img is None:
#             return 'error', 'cannot read image'
#         h, w = img.shape[:2]
#         if h < 50 or w < 50:
#             return 'error', f'image too small: {w}x{h}'
        
#         rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         res = face_mesh.process(rgb)
        
#         if not res or not res.multi_face_landmarks:
#             return 'error', 'no face landmarks found'
        
#         lm = res.multi_face_landmarks[0]
#         le = calc_ear(lm, LEFT_EYE_IDX, w, h)
#         re = calc_ear(lm, RIGHT_EYE_IDX, w, h)
#         avg = (le + re) / 2.0
#         state = 'closed' if avg < EAR_THRESHOLD else 'open'
#         return state, f'EAR={avg:.3f}'
#     except Exception as e:
#         print(f"EYE_STATE ERROR: {str(e)}")
#         return 'error', str(e)


# def verify_match(reg_path, live_path):
#     try:
#         result = DeepFace.verify(
#             img1_path=reg_path, img2_path=live_path,
#             model_name=MODEL, detector_backend=BACKEND,
#             enforce_detection=True,
#         )
#         matched = bool(result.get('verified', False))
#         dist = float(result.get('distance', 0))
#         thresh = float(result.get('threshold', 1))
#         conf = max(0, min(100, (1 - dist / thresh) * 100)) if thresh > 0 else 0
#         return matched, round(conf, 2), round(dist, 4), round(thresh, 4)
#     except Exception as e:
#         print(f"verify_match error: {e}")
#         return False, 0.0, 0.0, 0.0


# def get_username(uid):
#     meta = os.path.join(FACES_DIR, uid, 'meta.json')
#     if os.path.exists(meta):
#         try:
#             with open(meta, 'r', encoding='utf-8') as f:
#                 return json.load(f).get('userName', '')
#         except:
#             return ''
#     return ''


# def get_face_area(path):
#     try:
#         faces = DeepFace.extract_faces(img_path=path, detector_backend=BACKEND, enforce_detection=False)
#         if faces:
#             fa = faces[0].get('facial_area', {})
#             return float(fa.get('w', 0) * fa.get('h', 0))
#     except:
#         pass
#     return 0.0


# def laplacian_variance(path):
#     img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         return 0.0
#     return float(cv2.Laplacian(img, cv2.CV_64F).var())


# def temporal_consistency(paths):
#     diffs = []
#     for i in range(len(paths) - 1):
#         img1 = cv2.imread(paths[i], cv2.IMREAD_GRAYSCALE)
#         img2 = cv2.imread(paths[i+1], cv2.IMREAD_GRAYSCALE)
#         if img1 is None or img2 is None: continue
#         if img1.shape != img2.shape:
#             img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
#         diff = cv2.absdiff(img1, img2)
#         mean_diff = float(np.mean(diff))
#         diffs.append(mean_diff)
    
#     if len(diffs) < 2:
#         return False, {'reason': 'insufficient_frames'}
    
#     median_diff = float(np.median(diffs))
#     max_diff = float(np.max(diffs))
    
#     if max_diff < TEMP_STATIC_MAX:
#         return False, {'reason': 'completely_static', 'msg': 'No natural motion detected.'}
    
#     if max_diff > median_diff * TEMP_JUMP_RATIO and median_diff > 1.0:
#         return False, {'reason': 'sudden_jump', 'msg': 'Abnormal frame switch detected.'}
    
#     return True, {'max_diff': max_diff, 'median_diff': median_diff}


# def analyze_blink_sequence(states):
#     clean = [s for s in states if s in ('open', 'closed')]
#     if len(clean) < 3:
#         return False, 'Need at least 3 valid eye-state frames'
#     if 'closed' not in clean:
#         return False, 'No eye closure detected. Please blink naturally.'
    
#     natural_blink = False
#     for i in range(1, len(clean) - 1):
#         if clean[i] == 'closed' and clean[i-1] == 'open' and clean[i+1] == 'open':
#             natural_blink = True
#             break
#     if not natural_blink:
#         return False, 'Unnatural eye pattern. Blink normally (open-closed-open).'
    
#     transitions = sum(1 for i in range(len(clean)-1) if clean[i] != clean[i+1])
#     if transitions < 2:
#         return False, 'Eye movement insufficient'
#     if transitions > 5:
#         return False, 'Too many eye state changes (unnatural)'
    
#     return True, 'Natural blink verified'


# # ==================== ROUTES ====================

# @app.route('/')
# def index():
#     return jsonify({'status': 'running', 'version': '4.3-final', 'liveness': MEDIAPIPE_OK})


# @app.route('/health')
# def health():
#     return jsonify({'success': True, 'liveness': MEDIAPIPE_OK, 'status': 'healthy'})


# @app.route('/face/register', methods=['POST'])
# def register():
#     try:
#         d = request.json or {}
#         uid = (d.get('userId') or '').strip()
#         name = (d.get('name') or '').strip()
#         b64 = d.get('imageBase64') or ''

#         if not uid or not b64:
#             return jsonify({'success': False, 'msg': 'userId and imageBase64 required'}), 400

#         udir = os.path.join(FACES_DIR, uid)
#         os.makedirs(udir, exist_ok=True)
#         fpath = os.path.join(udir, 'face.jpg')
#         save_b64(b64, fpath)

#         faces = detect_faces(fpath)
#         if len(faces) == 0:
#             if os.path.exists(fpath): os.remove(fpath)
#             return jsonify({'success': False, 'msg': 'No face detected'}), 400
#         if len(faces) > 1:
#             if os.path.exists(fpath): os.remove(fpath)
#             return jsonify({'success': False, 'msg': 'Multiple faces not allowed'}), 400

#         with open(os.path.join(udir, 'meta.json'), 'w', encoding='utf-8') as f:
#             json.dump({'userId': uid, 'userName': name, 'registered_at': datetime.now().isoformat()}, f, indent=2)

#         return jsonify({'success': True, 'msg': 'Face registered successfully', 'userId': uid})
#     except Exception as e:
#         print(f"REGISTER ERROR: {e}")
#         return jsonify({'success': False, 'msg': str(e)}), 500


# @app.route('/face/auto-verify', methods=['POST'])
# def auto_verify():
#     temp_paths = []
#     try:
#         d = request.json or {}
#         uid = (d.get('userId') or '').strip()
#         frames = d.get('framesBase64') or []

#         if not uid:
#             return jsonify({'success': False, 'matched': False, 'live': False, 'msg': 'userId required'}), 400
#         if len(frames) < 3:
#             return jsonify({'success': False, 'matched': False, 'live': False, 'msg': 'Minimum 3 frames required'}), 400

#         reg_path = os.path.join(FACES_DIR, uid, 'face.jpg')
#         if not os.path.exists(reg_path):
#             return jsonify({'success': False, 'matched': False, 'live': False, 'msg': 'Face not registered'}), 400

#         for i, b64 in enumerate(frames[:3]):
#             p = tmp_path(f'auto_{i}')
#             save_b64(b64, p)
#             temp_paths.append(p)

#         # Check each frame has exactly 1 face
#         for i, p in enumerate(temp_paths):
#             if len(detect_faces(p)) != 1:
#                 cleanup(*temp_paths)
#                 return jsonify({'success': False, 'matched': False, 'live': False, 'msg': f'Frame {i+1}: Exactly 1 face required'}), 400

#         # Texture Check
#         lap_scores = [laplacian_variance(p) for p in temp_paths]
#         avg_lap = sum(lap_scores) / len(lap_scores)
#         if avg_lap < 30.0:
#             cleanup(*temp_paths)
#             return jsonify({'success': True, 'matched': False, 'live': False, 'msg': 'Flat image detected. Real person required.'}), 200

#         # Face Match
#         best_frame = temp_paths[1] if len(temp_paths) >= 2 else temp_paths[0]
#         matched, conf, dist, thresh = verify_match(reg_path, best_frame)

#         cleanup(*temp_paths)

#         if not matched:
#             return jsonify({
#                 'success': True,
#                 'matched': False,
#                 'live': True,
#                 'confidence': round(conf, 2),
#                 'msg': f'Face NOT matched ({round(conf,1)}%). Please re-register.'
#             }), 200

#         return jsonify({
#             'success': True,
#             'matched': True,
#             'live': True,
#             'confidence': round(conf, 2),
#             'userName': get_username(uid),
#             'msg': f'✅ Verified + Matched ({round(conf,1)}%)'
#         })

#     except Exception as e:
#         print(f"AUTO_VERIFY ERROR: {e}")
#         cleanup(*temp_paths)
#         return jsonify({'success': False, 'matched': False, 'live': False, 'msg': str(e)}), 500


# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 5000))
#     print("=" * 70)
#     print("   FACE RECOGNITION SERVER v4.3 (Final Stable)")
#     print(f"   MediaPipe: {'ENABLED' if MEDIAPIPE_OK else 'DISABLED'}")
#     print(f"   Port: {port}")
#     print("=" * 70)
#     app.run(host='0.0.0.0', port=port, debug=False)


from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import base64
import os
import json
import shutil
import uuid
import requests  # 🆕 Cloudinary image download ke liye
import numpy as np
import cv2
from datetime import datetime

# ========= MEDIAPIPE (LIVENESS) =========
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3
    )
    MEDIAPIPE_OK = True
    print("✅ MediaPipe loaded")
except Exception as e:
    MEDIAPIPE_OK = False
    mp_face_mesh = None
    face_mesh = None
    print(f"⚠️ MediaPipe disabled: {e}")

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024

FACES_DIR = 'registered_faces'
TMP_DIR = 'temp_files'
MODEL = 'VGG-Face'
BACKEND = 'opencv'
EAR_THRESHOLD = 0.22

# Anti-spoofing thresholds
MIN_FRAMES = 5
MAX_FRAMES = 7
LAP_THRESHOLD = 60.0
AREA_VAR_MIN = 0.03
MOIRE_THRESHOLD = 14.0
TEMP_STATIC_MAX = 2.5
TEMP_JUMP_RATIO = 3.5

os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]


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


# 🆕 Cloudinary URL se image download karta hai
def download_image(url, save_path):
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return False, f'Download failed (status {resp.status_code})'
        with open(save_path, 'wb') as f:
            f.write(resp.content)
        return True, 'OK'
    except Exception as e:
        return False, str(e)


# 🆕 Registered face nikalta hai - pehle Cloudinary photoURL try karta hai, fir local
def get_registered_face(uid, photo_url):
    """
    Returns: (path, is_temp, error_msg)
    is_temp = True hai to baad mein cleanup karna hai
    """
    # Pehle Cloudinary photoURL try karo
    if photo_url:
        reg_temp = tmp_path('reg')
        ok, msg = download_image(photo_url, reg_temp)
        if ok:
            return reg_temp, True, None
        else:
            return None, False, f'Photo download failed: {msg}'

    # Agar photoURL nahi hai to local file check karo (backward compatible)
    local_path = os.path.join(FACES_DIR, uid, 'face.jpg')
    if os.path.exists(local_path):
        return local_path, False, None

    return None, False, 'Face not registered (no photoURL provided and no local face found)'


def detect_faces(path):
    try:
        faces = DeepFace.extract_faces(
            img_path=path,
            detector_backend=BACKEND,
            enforce_detection=False
        )
        return [f for f in faces if float(f.get('confidence', 0)) > 0.5]
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
        return 'error', 'mediapipe not available'
    try:
        img = cv2.imread(path)
        if img is None:
            return 'error', 'cant read image'

        h, w = img.shape[:2]
        if h < 50 or w < 50:
            return 'error', f'image too small: {w}x{h}'

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        if not res or not res.multi_face_landmarks:
            return 'error', 'no face landmarks found'

        lm = res.multi_face_landmarks[0]
        le = calc_ear(lm, LEFT_EYE_IDX, w, h)
        re = calc_ear(lm, RIGHT_EYE_IDX, w, h)
        avg = (le + re) / 2.0
        state = 'closed' if avg < EAR_THRESHOLD else 'open'
        return state, f'EAR={avg:.3f}'
    except Exception as e:
        print(f"EYE_STATE ERROR: {str(e)}")
        return 'error', str(e)


def verify_match(reg_path, live_path):
    result = DeepFace.verify(
        img1_path=reg_path,
        img2_path=live_path,
        model_name=MODEL,
        detector_backend=BACKEND,
        enforce_detection=True,
    )
    matched = bool(result.get('verified', False))
    dist = float(result.get('distance', 0))
    thresh = float(result.get('threshold', 1))
    conf = max(0, min(100, (1 - dist / thresh) * 100)) if thresh > 0 else 0
    return matched, round(conf, 2), round(dist, 4), round(thresh, 4)


def get_username(uid):
    meta = os.path.join(FACES_DIR, uid, 'meta.json')
    if os.path.exists(meta):
        with open(meta, 'r', encoding='utf-8') as f:
            return json.load(f).get('userName', '')
    return ''


def get_face_area(path):
    try:
        faces = DeepFace.extract_faces(
            img_path=path,
            detector_backend=BACKEND,
            enforce_detection=False
        )
        if faces:
            fa = faces[0].get('facial_area', {})
            return float(fa.get('w', 0) * fa.get('h', 0))
    except Exception as e:
        print(f"get_face_area error: {e}")
    return 0.0


def laplacian_variance(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def detect_moire(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, 0.0

    small = cv2.resize(img, (256, 256))
    f = np.fft.fft2(small)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    rows, cols = small.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    r = 20
    mask[crow-r:crow+r, ccol-r:ccol+r] = 0

    peaks = magnitude * mask
    peak_ratio = float(np.max(peaks) / (np.mean(magnitude) + 1e-6))
    is_moire = peak_ratio > MOIRE_THRESHOLD
    return is_moire, peak_ratio


def temporal_consistency(paths):
    diffs = []
    for i in range(len(paths) - 1):
        img1 = cv2.imread(paths[i], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(paths[i + 1], cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            continue

        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        diff = cv2.absdiff(img1, img2)
        mean_diff = float(np.mean(diff))
        diffs.append(mean_diff)

    if len(diffs) < 2:
        return False, {'reason': 'insufficient_frames', 'diffs': diffs}

    median_diff = float(np.median(diffs))
    max_diff = float(np.max(diffs))

    if max_diff < TEMP_STATIC_MAX:
        return False, {
            'reason': 'completely_static',
            'max_diff': max_diff,
            'msg': 'No natural motion detected. Real person required.'
        }

    if max_diff > median_diff * TEMP_JUMP_RATIO and median_diff > 1.0:
        return False, {
            'reason': 'sudden_jump',
            'max_diff': max_diff,
            'median_diff': median_diff,
            'msg': 'Abnormal frame switch detected (possible photo/video spoof).'
        }

    return True, {'max_diff': max_diff, 'median_diff': median_diff}


def analyze_blink_sequence(states):
    clean = [s for s in states if s in ('open', 'closed')]
    if len(clean) < 3:
        return False, 'Need at least 3 valid eye-state frames'

    if 'closed' not in clean:
        return False, 'No eye closure detected. Please blink naturally.'

    natural_blink = False
    for i in range(1, len(clean) - 1):
        if clean[i] == 'closed' and clean[i - 1] == 'open' and clean[i + 1] == 'open':
            natural_blink = True
            break

    if not natural_blink:
        return False, 'Unnatural eye pattern. Blink your eyes normally (open-closed-open).'

    transitions = sum(1 for i in range(len(clean) - 1) if clean[i] != clean[i + 1])
    if transitions < 2:
        return False, 'Eye movement insufficient'
    if transitions > 5:
        return False, 'Too many eye state changes (unnatural)'

    return True, 'Natural blink verified'


# ==================== ROUTES ====================

@app.route('/')
def index():
    return jsonify({
        'status': 'running',
        'version': '5.0-cloudinary',
        'liveness': MEDIAPIPE_OK,
        'storage': 'Cloudinary photoURL (Firebase based)',
        'endpoints': [
            '/health',
            '/face/register',
            '/face/verify',
            '/face/liveness-verify',
            '/face/liveness-verify-strong',
            '/face/check-eyes',
            '/face/delete',
            '/face/list',
            '/face/auto-verify'
        ]
    })


@app.route('/health')
def health():
    return jsonify({
        'success': True,
        'liveness': MEDIAPIPE_OK,
        'time': datetime.now().isoformat()
    })


@app.route('/face/register', methods=['POST'])
def register():
    try:
        d = request.json or {}
        uid = (d.get('userId') or '').strip()
        name = (d.get('name') or d.get('userName') or '').strip()
        b64 = d.get('imageBase64') or d.get('image') or ''

        if not uid or not b64:
            return jsonify({'success': False, 'msg': 'userId + imageBase64 required'}), 400

        udir = os.path.join(FACES_DIR, uid)
        os.makedirs(udir, exist_ok=True)
        fpath = os.path.join(udir, 'face.jpg')
        save_b64(b64, fpath)

        faces = detect_faces(fpath)
        if len(faces) == 0:
            if os.path.exists(fpath):
                os.remove(fpath)
            return jsonify({'success': False, 'msg': 'No face detected'}), 400

        if len(faces) > 1:
            if os.path.exists(fpath):
                os.remove(fpath)
            return jsonify({'success': False, 'msg': 'Multiple faces — only 1 allowed'}), 400

        with open(os.path.join(udir, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'userId': uid,
                'userName': name,
                'at': datetime.now().isoformat()
            }, f, indent=2)

        return jsonify({
            'success': True,
            'msg': f'{name or uid} registered OK',
            'userId': uid
        })
    except Exception as e:
        print(f"register error: {e}")
        return jsonify({'success': False, 'msg': str(e)}), 500


@app.route('/face/verify', methods=['POST'])
def verify():
    t = None
    reg_path = None
    reg_is_temp = False
    try:
        d = request.json or {}
        uid = (d.get('userId') or '').strip()
        b64 = d.get('imageBase64') or d.get('image') or ''
        photo_url = (d.get('photoURL') or '').strip()  # 🆕

        if not uid or not b64:
            return jsonify({
                'success': False,
                'matched': False,
                'msg': 'userId + image required'
            }), 400

        # 🆕 Cloudinary photoURL se ya local se registered face nikalo
        reg_path, reg_is_temp, err = get_registered_face(uid, photo_url)
        if err:
            return jsonify({'success': False, 'matched': False, 'msg': err}), 404

        t = tmp_path('v')
        save_b64(b64, t)

        faces = detect_faces(t)
        if len(faces) == 0:
            return jsonify({'success': False, 'matched': False, 'msg': 'No face detected'}), 400
        if len(faces) > 1:
            return jsonify({'success': False, 'matched': False, 'msg': 'Multiple faces'}), 400

        matched, conf, dist, thresh = verify_match(reg_path, t)

        return jsonify({
            'success': True,
            'matched': matched,
            'confidence': conf,
            'distance': dist,
            'threshold': thresh,
            'userName': get_username(uid),
            'msg': f'Matched {conf}%' if matched else f'Not matched {conf}%'
        })
    except Exception as e:
        print(f"verify error: {e}")
        return jsonify({'success': False, 'matched': False, 'msg': str(e)}), 500
    finally:
        cleanup(t)
        if reg_is_temp:
            cleanup(reg_path)


@app.route('/face/check-eyes', methods=['POST'])
def check_eyes():
    t = None
    try:
        if not MEDIAPIPE_OK:
            return jsonify({'success': False, 'msg': 'mediapipe not installed'}), 500

        d = request.json or {}
        b64 = d.get('imageBase64') or ''
        if not b64:
            return jsonify({'success': False, 'msg': 'image required'}), 400

        t = tmp_path('eye')
        save_b64(b64, t)
        state, detail = eye_state(t)

        return jsonify({
            'success': state != 'error',
            'eyeState': state,
            'detail': detail
        })
    except Exception as e:
        print(f"check_eyes error: {e}")
        return jsonify({'success': False, 'msg': str(e)}), 500
    finally:
        cleanup(t)


@app.route('/face/liveness-verify', methods=['POST'])
def liveness_verify():
    tc = None
    to = None
    reg_path = None
    reg_is_temp = False
    try:
        d = request.json or {}
        uid = (d.get('userId') or '').strip()
        b64_closed = d.get('frameClosedBase64') or d.get('frame1') or ''
        b64_open = d.get('frameOpenBase64') or d.get('frame2') or ''
        photo_url = (d.get('photoURL') or '').strip()  # 🆕

        if not uid:
            return jsonify({
                'success': False,
                'matched': False,
                'live': False,
                'msg': 'userId required'
            }), 400

        if not b64_closed or not b64_open:
            return jsonify({
                'success': False,
                'matched': False,
                'live': False,
                'msg': 'Both frames required'
            }), 400

        # 🆕 Registered face nikalo
        reg_path, reg_is_temp, err = get_registered_face(uid, photo_url)
        if err:
            return jsonify({
                'success': False,
                'matched': False,
                'live': False,
                'msg': err
            }), 404

        tc = tmp_path('lc')
        to = tmp_path('lo')
        save_b64(b64_closed, tc)
        save_b64(b64_open, to)

        live_ok = False
        live_msg = ''

        if MEDIAPIPE_OK:
            s1, d1 = eye_state(tc)
            s2, d2 = eye_state(to)

            if s1 == 'error':
                live_msg = f'Frame1 error: {d1}'
            elif s2 == 'error':
                live_msg = f'Frame2 error: {d2}'
            elif s1 == 'closed' and s2 == 'open':
                live_ok = True
                live_msg = f'Blink OK (closed:{d1}, open:{d2})'
            elif s1 == 'open' and s2 == 'open':
                live_msg = 'Both frames eyes OPEN — blink not detected'
            elif s1 == 'closed' and s2 == 'closed':
                live_msg = 'Both frames eyes CLOSED — open eyes in frame 2'
            else:
                live_msg = f'Unexpected: frame1={s1}, frame2={s2}'
        else:
            live_ok = True
            live_msg = 'mediapipe missing — skipped'

        if not live_ok:
            return jsonify({
                'success': True,
                'matched': False,
                'live': False,
                'msg': live_msg,
                'step': 'liveness_failed'
            })

        faces = detect_faces(to)
        if len(faces) == 0:
            return jsonify({
                'success': True,
                'matched': False,
                'live': True,
                'msg': 'No face in frame 2'
            })

        matched, conf, dist, thresh = verify_match(reg_path, to)
        msg = f'Live+Matched {conf}%' if matched else f'Live but NOT matched {conf}%'

        return jsonify({
            'success': True,
            'matched': matched,
            'live': True,
            'confidence': conf,
            'distance': dist,
            'threshold': thresh,
            'userName': get_username(uid),
            'msg': msg,
            'liveMsg': live_msg,
            'time': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"liveness_verify error: {e}")
        return jsonify({'success': False, 'matched': False, 'live': False, 'msg': str(e)}), 500
    finally:
        cleanup(tc, to)
        if reg_is_temp:
            cleanup(reg_path)


@app.route('/face/liveness-verify-strong', methods=['POST'])
def liveness_verify_strong():
    temp_paths = []
    reg_path = None
    reg_is_temp = False
    try:
        d = request.json or {}
        uid = (d.get('userId') or '').strip()
        frames = d.get('framesBase64') or []
        photo_url = (d.get('photoURL') or '').strip()  # 🆕

        if not uid:
            return jsonify({
                'success': False,
                'matched': False,
                'live': False,
                'msg': 'userId required',
                'layer': 'input'
            }), 400

        if len(frames) < MIN_FRAMES:
            return jsonify({
                'success': False,
                'matched': False,
                'live': False,
                'msg': f'Send {MIN_FRAMES}-{MAX_FRAMES} frames for strong verification',
                'layer': 'input'
            }), 400

        # 🆕 Registered face nikalo
        reg_path, reg_is_temp, err = get_registered_face(uid, photo_url)
        if err:
            return jsonify({
                'success': False,
                'matched': False,
                'live': False,
                'msg': err,
                'layer': 'registration'
            }), 404

        for i, b64 in enumerate(frames[:MAX_FRAMES]):
            p = tmp_path(f'str_{i}')
            save_b64(b64, p)
            temp_paths.append(p)

        checks = {
            'frames_ok': False,
            'blink_natural': False,
            'micro_movement': False,
            'texture_real': False,
            'no_moire': False,
            'temporal_smooth': False,
            'face_match': False
        }

        # LAYER 1: Single face per frame
        face_data = []
        for p in temp_paths:
            fcs = detect_faces(p)
            if len(fcs) != 1:
                return jsonify({
                    'success': False,
                    'matched': False,
                    'live': False,
                    'msg': 'Multiple or no face in frame. Exactly 1 face required.',
                    'layer': 'face_count',
                    'checks': checks
                }), 400
            face_data.append({
                'confidence': float(fcs[0].get('confidence', 0)),
                'area': get_face_area(p)
            })

        checks['frames_ok'] = True

        # LAYER 2: Natural Blink
        eye_states = []
        eye_details = []
        for p in temp_paths:
            if MEDIAPIPE_OK:
                st, det = eye_state(p)
            else:
                st, det = 'unknown', 'mediapipe_off'
            eye_states.append(st)
            eye_details.append(det)

        blink_ok, blink_msg = analyze_blink_sequence(eye_states)
        if not blink_ok:
            return jsonify({
                'success': True,
                'matched': False,
                'live': False,
                'msg': blink_msg,
                'layer': 'blink',
                'eye_sequence': eye_states,
                'eye_details': eye_details,
                'checks': checks
            })
        checks['blink_natural'] = True

        # LAYER 3: Micro-movement
        areas = [fd['area'] for fd in face_data if fd['area'] > 0]
        if len(areas) >= 2:
            area_variance = (max(areas) - min(areas)) / (max(areas) + 1e-6)
            if area_variance < AREA_VAR_MIN:
                return jsonify({
                    'success': True,
                    'matched': False,
                    'live': False,
                    'msg': 'Static image detected. Real person required (no micro-movement).',
                    'layer': 'micro_movement',
                    'area_variance': round(area_variance, 4),
                    'checks': checks
                })
        checks['micro_movement'] = True

        # LAYER 4: Texture / Depth
        lap_vars = [laplacian_variance(p) for p in temp_paths]
        avg_lap = float(np.mean(lap_vars))
        if avg_lap < LAP_THRESHOLD:
            return jsonify({
                'success': True,
                'matched': False,
                'live': False,
                'msg': 'Flat image detected (possible photo/screen). Real 3D face required.',
                'layer': 'texture',
                'laplacian_avg': round(avg_lap, 2),
                'checks': checks
            })
        checks['texture_real'] = True

        # LAYER 5: Screen / Moire
        for i, p in enumerate(temp_paths):
            is_moire, ratio = detect_moire(p)
            if is_moire:
                return jsonify({
                    'success': True,
                    'matched': False,
                    'live': False,
                    'msg': 'Screen/photo spoof detected (moire pattern). Real person required.',
                    'layer': 'moire',
                    'frame_index': i,
                    'moire_ratio': round(ratio, 2),
                    'checks': checks
                })
        checks['no_moire'] = True

        # LAYER 6: Temporal consistency
        smooth, temp_info = temporal_consistency(temp_paths)
        if not smooth:
            return jsonify({
                'success': True,
                'matched': False,
                'live': False,
                'msg': temp_info.get('msg', 'Unnatural frame sequence.'),
                'layer': 'temporal',
                'details': temp_info,
                'checks': checks
            })
        checks['temporal_smooth'] = True

        # LAYER 7: Face Match
        best_idx = 0
        best_score = -1
        for i in range(len(temp_paths)):
            conf_score = face_data[i]['confidence']
            eye_ok = eye_states[i] == 'open'
            score = conf_score + (100 if eye_ok else 0)
            if score > best_score:
                best_score = score
                best_idx = i

        matched, conf, dist, thresh = verify_match(reg_path, temp_paths[best_idx])
        checks['face_match'] = matched

        if not matched:
            return jsonify({
                'success': True,
                'matched': False,
                'live': True,
                'msg': f'Live person verified, but face NOT matched ({conf}%). Register again if needed.',
                'layer': 'face_match',
                'confidence': conf,
                'checks': checks,
                'time': datetime.now().isoformat()
            })

        return jsonify({
            'success': True,
            'matched': True,
            'live': True,
            'confidence': conf,
            'distance': dist,
            'threshold': thresh,
            'userName': get_username(uid),
            'msg': f'✅ Strong Liveness Passed + Matched ({conf}%)',
            'checks': checks,
            'eye_sequence': eye_states,
            'eye_details': eye_details,
            'time': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"liveness_verify_strong error: {e}")
        return jsonify({
            'success': False,
            'matched': False,
            'live': False,
            'msg': f'Server error: {str(e)}',
            'layer': 'exception'
        }), 500
    finally:
        cleanup(*temp_paths)
        if reg_is_temp:
            cleanup(reg_path)


@app.route('/face/auto-verify', methods=['POST'])
def auto_verify():
    temp_paths = []
    reg_path = None
    reg_is_temp = False
    try:
        d = request.json or {}
        uid = (d.get('userId') or '').strip()
        frames = d.get('framesBase64') or []
        photo_url = (d.get('photoURL') or '').strip()  # 🆕 Firebase se aane wali Cloudinary URL

        if not uid:
            return jsonify({
                'success': False,
                'matched': False,
                'live': False,
                'msg': 'userId required'
            }), 400

        if len(frames) < 3:
            return jsonify({
                'success': False,
                'matched': False,
                'live': False,
                'msg': 'Need 3 frames'
            }), 400

        # 🆕 Cloudinary photoURL se ya local se registered face nikalo
        reg_path, reg_is_temp, err = get_registered_face(uid, photo_url)
        if err:
            return jsonify({
                'success': False,
                'matched': False,
                'live': False,
                'msg': err
            }), 400

        # 🆕 Registered photo me face hai ya nahi check karo
        if len(detect_faces(reg_path)) == 0:
            return jsonify({
                'success': False,
                'matched': False,
                'live': False,
                'msg': 'No face found in your profile photo. Contact admin.'
            }), 400

        for i, b64 in enumerate(frames[:3]):
            p = tmp_path(f'auto_{i}')
            save_b64(b64, p)
            temp_paths.append(p)

        face_boxes = []
        for i, p in enumerate(temp_paths):
            try:
                faces = DeepFace.extract_faces(
                    img_path=p,
                    detector_backend=BACKEND,
                    enforce_detection=False
                )
                valid = [f for f in faces if float(f.get('confidence', 0)) > 0.5]
                if len(valid) != 1:
                    return jsonify({
                        'success': False,
                        'matched': False,
                        'live': False,
                        'msg': f'Frame {i+1}: {len(valid)} faces found. Exactly 1 face needed.'
                    }), 400

                fa = valid[0].get('facial_area', {})
                face_boxes.append({
                    'x': fa.get('x', 0),
                    'y': fa.get('y', 0),
                    'w': fa.get('w', 0),
                    'h': fa.get('h', 0)
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'matched': False,
                    'live': False,
                    'msg': f'Frame {i+1} face detect error: {str(e)}'
                }), 500

        movements = []
        for i in range(len(face_boxes) - 1):
            dx = abs(face_boxes[i]['x'] - face_boxes[i + 1]['x'])
            dy = abs(face_boxes[i]['y'] - face_boxes[i + 1]['y'])
            dw = abs(face_boxes[i]['w'] - face_boxes[i + 1]['w'])
            dh = abs(face_boxes[i]['h'] - face_boxes[i + 1]['h'])
            movements.append(dx + dy + dw + dh)

        avg_movement = sum(movements) / len(movements) if movements else 0
        print(f"Auto-verify: avg_movement={avg_movement}")

        if avg_movement < 1.0:
            return jsonify({
                'success': True,
                'matched': False,
                'live': False,
                'msg': 'Static image detected. Please hold phone naturally and do not use photos.'
            }), 200

        lap_scores = [laplacian_variance(p) for p in temp_paths]
        avg_lap = sum(lap_scores) / len(lap_scores)
        print(f"Auto-verify: avg_lap={avg_lap}")

        if avg_lap < 30.0:
            return jsonify({
                'success': True,
                'matched': False,
                'live': False,
                'msg': 'Flat image detected. Real person required.'
            }), 200

        best_frame = temp_paths[1] if len(temp_paths) >= 2 else temp_paths[0]
        result = DeepFace.verify(
            img1_path=reg_path,  # 🆕 ab Cloudinary se downloaded ya local image
            img2_path=best_frame,
            model_name=MODEL,
            detector_backend=BACKEND,
            enforce_detection=False,
        )

        matched = bool(result.get('verified', False))
        distance = float(result.get('distance', 0))
        threshold = float(result.get('threshold', 1))
        confidence = max(0, min(100, (1 - distance / threshold) * 100)) if threshold > 0 else 0

        if not matched:
            return jsonify({
                'success': True,
                'matched': False,
                'live': True,
                'confidence': round(confidence, 2),
                'msg': f'Face NOT matched ❌ ({round(confidence,1)}%). Please contact admin to re-register your face.'
            }), 200

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
        print(f"AUTO_VERIFY CRASH: {str(e)}")
        return jsonify({
            'success': False,
            'matched': False,
            'live': False,
            'msg': f'Server error: {str(e)}'
        }), 500
    finally:
        cleanup(*temp_paths)
        if reg_is_temp:
            cleanup(reg_path)


@app.route('/face/delete', methods=['POST'])
def delete():
    try:
        uid = (request.json or {}).get('userId', '').strip()
        d = os.path.join(FACES_DIR, uid)
        if os.path.exists(d):
            shutil.rmtree(d)
            return jsonify({'success': True, 'msg': 'Deleted'})
        return jsonify({'success': False, 'msg': 'Not found'}), 404
    except Exception as e:
        print(f"delete error: {e}")
        return jsonify({'success': False, 'msg': str(e)}), 500


@app.route('/face/list')
def list_faces():
    try:
        users = []
        if os.path.exists(FACES_DIR):
            for uid in os.listdir(FACES_DIR):
                m = os.path.join(FACES_DIR, uid, 'meta.json')
                if os.path.exists(m):
                    with open(m, 'r', encoding='utf-8') as f:
                        users.append(json.load(f))
        return jsonify({'success': True, 'count': len(users), 'users': users})
    except Exception as e:
        print(f"list_faces error: {e}")
        return jsonify({'success': False, 'msg': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("=" * 60)
    print(" FACE + STRONG LIVENESS SERVER v5.0 (Cloudinary) ")
    print(f" Port: {port}")
    print(f" Liveness: {'ON' if MEDIAPIPE_OK else 'OFF'}")
    print(f" Storage: Cloudinary photoURL + Local fallback")
    print("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=False)