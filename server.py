from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import base64
import os
import json
import shutil
import uuid
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ========= CONFIG =========
FACES_DIR = 'registered_faces'
TMP_DIR = 'temp_files'
MODEL_NAME = 'VGG-Face'
DETECTOR_BACKEND = 'opencv'

os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)


# ========= HELPERS =========
def save_base64_image(base64_string, filepath):
    """Save base64 image to file"""
    if not base64_string:
        raise ValueError("Empty image data")

    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]

    image_data = base64.b64decode(base64_string)
    with open(filepath, 'wb') as f:
        f.write(image_data)


def get_image_from_request(data):
    """Accept both image and imageBase64"""
    return data.get('imageBase64') or data.get('image') or ''


def get_name_from_request(data):
    """Accept both userName and name"""
    return data.get('userName') or data.get('name') or ''


def count_faces(image_path):
    """
    Return exact detected face count.
    enforce_detection=False so we can safely handle no-face case,
    but we'll manually validate count.
    """
    faces = DeepFace.extract_faces(
        img_path=image_path,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=False
    )

    valid_faces = [f for f in faces if float(f.get('confidence', 0)) > 0.5]
    return len(valid_faces), valid_faces


def build_temp_path(prefix='temp'):
    return os.path.join(TMP_DIR, f'{prefix}_{uuid.uuid4().hex}.jpg')


# ========= ROUTES =========
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'success': True,
        'status': 'running',
        'service': 'Face Recognition API (DeepFace)',
        'version': '2.1.0',
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'success': True,
        'message': 'server running',
        'service': 'Face Recognition API',
        'version': '2.1.0',
        'time': datetime.now().isoformat(),
    })


@app.route('/detect', methods=['POST'])
def detect_face():
    """Check if exactly one face exists in photo"""
    temp_path = None
    try:
        data = request.json or {}
        image_base64 = get_image_from_request(data)

        if not image_base64:
            return jsonify({
                'success': False,
                'faceDetected': False,
                'faceCount': 0,
                'message': 'No image provided'
            }), 400

        temp_path = build_temp_path('detect')
        save_base64_image(image_base64, temp_path)

        face_count, faces = count_faces(temp_path)

        if face_count == 0:
            return jsonify({
                'success': False,
                'faceDetected': False,
                'faceCount': 0,
                'message': 'No face detected'
            })

        if face_count > 1:
            return jsonify({
                'success': False,
                'faceDetected': True,
                'faceCount': face_count,
                'message': f'{face_count} faces detected. Sirf ek face chahiye.'
            })

        face = faces[0]
        return jsonify({
            'success': True,
            'faceDetected': True,
            'faceCount': 1,
            'confidence': round(float(face.get('confidence', 0)) * 100, 2),
            'faceLocation': face.get('facial_area', {}),
            'message': 'Face detected ✅'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/register', methods=['POST'])
@app.route('/face/register', methods=['POST'])
def register_face():
    """Register employee/manager face"""
    try:
        data = request.json or {}
        user_id = data.get('userId', '').strip()
        user_name = get_name_from_request(data).strip()
        image_base64 = get_image_from_request(data)

        if not user_id or not image_base64:
            return jsonify({
                'success': False,
                'message': 'userId and image required'
            }), 400

        user_dir = os.path.join(FACES_DIR, user_id)
        os.makedirs(user_dir, exist_ok=True)

        face_path = os.path.join(user_dir, 'face.jpg')
        save_base64_image(image_base64, face_path)

        face_count, _ = count_faces(face_path)

        if face_count == 0:
            if os.path.exists(face_path):
                os.remove(face_path)
            return jsonify({
                'success': False,
                'message': 'No face detected in photo'
            })

        if face_count > 1:
            if os.path.exists(face_path):
                os.remove(face_path)
            return jsonify({
                'success': False,
                'message': 'Multiple faces detected. Sirf ek face hona chahiye.'
            })

        meta_path = os.path.join(user_dir, 'meta.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({
                'userId': user_id,
                'userName': user_name,
                'registeredAt': datetime.now().isoformat(),
            }, f, ensure_ascii=False, indent=2)

        return jsonify({
            'success': True,
            'message': f'{user_name or user_id} ka face register ho gaya ✅',
            'userId': user_id,
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500


@app.route('/verify', methods=['POST'])
@app.route('/face/verify', methods=['POST'])
def verify_face():
    """Verify face for attendance"""
    temp_path = None
    try:
        data = request.json or {}
        user_id = data.get('userId', '').strip()
        image_base64 = get_image_from_request(data)

        if not user_id or not image_base64:
            return jsonify({
                'success': False,
                'matched': False,
                'message': 'userId and image required'
            }), 400

        registered_path = os.path.join(FACES_DIR, user_id, 'face.jpg')
        if not os.path.exists(registered_path):
            return jsonify({
                'success': False,
                'matched': False,
                'message': 'Face not registered'
            })

        temp_path = build_temp_path(f'verify_{user_id}')
        save_base64_image(image_base64, temp_path)

        face_count, _ = count_faces(temp_path)

        if face_count == 0:
            return jsonify({
                'success': False,
                'matched': False,
                'message': 'No face detected in live photo'
            })

        if face_count > 1:
            return jsonify({
                'success': False,
                'matched': False,
                'message': 'Multiple faces detected in live photo'
            })

        result = DeepFace.verify(
            img1_path=registered_path,
            img2_path=temp_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
        )

        matched = bool(result.get('verified', False))
        distance = float(result.get('distance', 0))
        threshold = float(result.get('threshold', 1))

        if threshold > 0:
            confidence = max(0, min(100, (1 - (distance / threshold)) * 100))
        else:
            confidence = 0

        meta_path = os.path.join(FACES_DIR, user_id, 'meta.json')
        user_name = ''
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                user_name = meta.get('userName', '')

        return jsonify({
            'success': True,
            'matched': matched,
            'confidence': round(confidence, 2),
            'distance': round(distance, 4),
            'threshold': round(threshold, 4),
            'userName': user_name,
            'message': f'Face matched ✅ ({round(confidence, 2)}%)' if matched else f'Face NOT matched ❌ ({round(confidence, 2)}%)',
            'timestamp': datetime.now().isoformat(),
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'matched': False,
            'message': f'Error: {str(e)}'
        }), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/delete', methods=['POST'])
@app.route('/face/delete', methods=['POST'])
def delete_face():
    """Delete registered face"""
    try:
        data = request.json or {}
        user_id = data.get('userId', '').strip()
        user_dir = os.path.join(FACES_DIR, user_id)

        if os.path.exists(user_dir):
            shutil.rmtree(user_dir)
            return jsonify({
                'success': True,
                'message': 'Face deleted'
            })

        return jsonify({
            'success': False,
            'message': 'User not found'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500


@app.route('/list', methods=['GET'])
def list_users():
    """List registered users"""
    users = []
    if os.path.exists(FACES_DIR):
      for uid in os.listdir(FACES_DIR):
          meta_path = os.path.join(FACES_DIR, uid, 'meta.json')
          if os.path.exists(meta_path):
              with open(meta_path, 'r', encoding='utf-8') as f:
                  meta = json.load(f)
                  users.append(meta)

    return jsonify({
        'success': True,
        'users': users,
        'count': len(users)
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', '1') == '1'
    print(f"🚀 Face API running on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)